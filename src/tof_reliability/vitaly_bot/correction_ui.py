"""Telegram selection UI; callback payloads carry audit revision, never raw text."""
import json
from aiogram import F
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, ReplyParameters
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from .corrections import items, item_at, LABELS
from .models import StateError

PAGE_SIZE = 6


def keyboard(audit, page=None):
    prefix = 'vc:' + audit['id'] + ':' + str(audit['revision']) + ':'
    def button(label, action):
        return InlineKeyboardButton(text=label, callback_data=prefix + action)
    rows = []
    if page is None:
        rows = [[button('Исправить пункт', 'page:0')], [button('Актуальный отчёт', 'view')]]
    else:
        choices = items(json.loads(audit['current_json']))
        page = max(0, min(page, max(0, (len(choices)-1)//PAGE_SIZE)))
        for key, item in choices[page*PAGE_SIZE:(page+1)*PAGE_SIZE]:
            rows.append([button(key + ' · ' + str(item.get('title') or item.get('text'))[:55], 'pick:' + key)])
        nav = []
        if page > 0:
            nav.append(button('← Назад', 'page:' + str(page-1)))
        if (page+1)*PAGE_SIZE < len(choices):
            nav.append(button('Далее →', 'page:' + str(page+1)))
        if nav:
            rows.append(nav)
        rows.append([button('Отмена', 'cancel')])
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def show_report(runtime, case_id, chat_id):
    from .telegram import format_review, send_with_retry
    corrections = runtime.service.corrections
    audit = corrections.ensure(case_id)
    report = corrections.effective(case_id)
    # Marks are presentation-only. Original and effective JSON retain their fields.
    for key, decision in json.loads(audit['decisions_json']).items():
        item = item_at(report, key)
        field = 'text' if key.startswith('Q') else 'title'
        item[field] = '[' + LABELS[decision['decision']] + '] ' + item[field]
    text = format_review(report)
    if audit['revision']:
        text += '\n\nВерсия правок: ' + str(audit['revision']) + '. Общий балл, вердикт и список подготовки не пересчитывались.'
    message_ids = await send_with_retry(runtime.bot, chat_id, text)
    for mid in message_ids:
        corrections.record_message(audit['id'], chat_id, mid, 'report')
    message_ids = await send_with_retry(runtime.bot, chat_id, 'Действия с этим аудитом' +
                    (' — есть исправления.' if audit['revision'] else '.'), reply_markup=keyboard(audit))
    for mid in message_ids:
        corrections.record_message(audit['id'], chat_id, mid)


async def show_result(runtime, job, result):
    from .telegram import send_with_retry
    corrections = runtime.service.corrections
    audit = corrections.get(job['payload']['audit_id'], job['chat_id'])
    text = LABELS[result['decision']] + ' · ' + job['payload']['item_id'] + '\n' + result['explanation']
    replacement = result.get('replacement')
    if replacement:
        text += '\n\n' + (replacement.get('title') or replacement.get('text', ''))
        text += '\n' + replacement.get('detail', replacement.get('expected_answer', ''))
    if result.get('references'):
        text += '\nОснования:\n' + '\n'.join(result['references'])
    if result['decision'] == 'clarification':
        text += '\n\n' + result['clarifying_question'] + '\nОтветьте следующим сообщением или нажмите «Отмена».'
    else:
        text += '\n\nОстальные пункты и общий балл не изменены.'
    source_messages = corrections.messages(audit['id'])
    for entry in source_messages:
        if entry['kind'] != 'control':
            continue
        try:
            await runtime.bot.edit_message_text('У этого аудита есть результаты перепроверки. Откройте актуальную версию.',
                    chat_id=entry['chat_id'], message_id=entry['message_id'], reply_markup=keyboard(audit))
        except (TelegramBadRequest, TelegramNetworkError):
            pass
    reports = [entry for entry in source_messages if entry['kind'] == 'report']
    reply = {'reply_parameters': ReplyParameters(message_id=reports[0]['message_id'], allow_sending_without_reply=True)} if reports else {}
    await send_with_retry(runtime.bot, job['chat_id'], text, **reply)
    markup = keyboard(audit)
    if result['decision'] == 'clarification':
        markup.inline_keyboard.append([InlineKeyboardButton(text='Отмена', callback_data=f"vc:{audit['id']}:{audit['revision']}:cancel")])
    message_ids = await send_with_retry(runtime.bot, job['chat_id'], 'Действия с аудитом:', reply_markup=markup)
    for mid in message_ids:
        corrections.record_message(audit['id'], job['chat_id'], mid)


def register(router, runtime):
    @router.callback_query(F.data.startswith('vc:'))
    async def callback(query: CallbackQuery):
        from .telegram import user_error, send_with_retry
        if not query.message:
            await query.answer('Откройте аудит в чате с ботом.')
            return
        try:
            await query.answer()
        except (TelegramBadRequest, TelegramNetworkError):
            pass  # A stale spinner must not prevent processing the actual action.
        chat_id = query.message.chat.id
        try:
            await runtime.case_for_user(chat_id)  # Apply the normal four-day expiration first.
            _, audit_id, revision, action, *argument = query.data.split(':')
            corrections = runtime.service.corrections
            audit = corrections.get(audit_id, chat_id, idle=True)
            if action == 'view':
                await show_report(runtime, audit['case_id'], chat_id)
                return
            if int(revision) != audit['revision']:
                raise StateError('Отчёт обновился. Нажмите «Актуальный отчёт».')
            if action == 'page':
                await send_with_retry(runtime.bot, chat_id, 'Выберите пункт для перепроверки:', reply_markup=keyboard(audit, int(argument[0])))
            elif action == 'pick':
                key = argument[0]
                corrections.select(audit_id, int(revision), key, chat_id)
                selected = item_at(json.loads(audit['current_json']), key)
                await send_with_retry(runtime.bot, chat_id, key + '\n' + (selected.get('title') or selected.get('text', ''))
                                      + '\n' + selected.get('detail', selected.get('expected_answer', '')))
                await send_with_retry(runtime.bot, chat_id, 'Что здесь неверно? Укажите слайд и объясните правильный смысл.',
                    reply_markup=InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='Отмена', callback_data=f'vc:{audit_id}:{revision}:cancel')]]))
            elif action == 'cancel':
                corrections.select(audit_id, int(revision), None, chat_id)
                await send_with_retry(runtime.bot, chat_id, 'Исправление отменено. Аудит не изменён.')
        except Exception as error:
            await send_with_retry(runtime.bot, chat_id, user_error(error))
