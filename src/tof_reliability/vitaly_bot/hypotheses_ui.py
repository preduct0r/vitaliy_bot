"""Private Telegram presentation of saved hypothesis results."""
from __future__ import annotations

from typing import Any

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup


def _value(value: Any) -> str:
    if isinstance(value, list):
        return '; '.join(map(str, value)) or 'Нет данных'
    return str(value or 'Нет данных')


def render(card: dict, *, plan: bool = False) -> str:
    lines = ['H' + str(card['number']) + ('. План проверки' if plan else '. Гипотеза')]
    fields = [('Управленческий вопрос', 'management_question'), ('Решение', 'decision'),
              ('Периметр', 'scope'), ('Показатели, сравнения и метод', 'metrics_method')]
    if not plan:
        for label, key in fields:
            lines.append(label + ': ' + _value(card.get(key)))
        interest = card.get('meeting_interest') or {}
        attribution = {'interpretation': 'интерпретация', 'direct_request': 'прямой запрос',
                       'third_party_paraphrase': 'пересказ третьего лица',
                       'third_party': 'свидетельство третьего лица'}.get(interest.get('attribution'), 'атрибуция не определена')
        lines.append('Интерес по источнику: ' + attribution + '; ' + _value(interest.get('reference')))
    else:
        lines.append('Показатели, сравнения и метод: ' + _value(card.get('metrics_method')))
    readiness = {'partial': 'частичная', 'available': 'доступны по указанному маршруту', 'unavailable': 'не подтверждена'}.get(
        card.get('source_readiness'), 'не определена')
    brief = _value(card.get('analyst_brief'))
    if 'анализ пока не выполнять' in brief.casefold():
        brief = ('Подготовить проверку: ' + _value(card.get('metrics_method')) +
                 '. Уточнить периметр: ' + _value(card.get('scope')) +
                 '. Проверить источники и восполнить пробелы: ' + _value(card.get('missing_evidence')) +
                 '. Результаты и ограничения представить для решения; анализ ещё не выполнен.')
    source_references = ((card.get('source_references') or []) +
                         (card.get('case_material_references') or []))
    source_label = _value(source_references) if source_references else 'Источники не подтверждены'
    lines.extend(['Доступные источники: ' + source_label,
                  'Готовность источников: ' + readiness,
                  'Пробелы: ' + _value(card.get('missing_evidence')),
                  'Опровержение: ' + _value(card.get('falsification')),
                  'Сложные вопросы: ' + _value(card.get('hard_questions')),
                  'Поручение аналитику: ' + brief])
    if not plan:
        lines.append('Новизна: неизвестна')
    return '\n'.join(lines)


def keyboard(run: dict, number: int | None = None) -> InlineKeyboardMarkup:
    base = 'hy:' + run['id'] + ':' + str(run['revision']) + ':'
    rows = [[InlineKeyboardButton(text='H' + str(i), callback_data=base + 'v:' + str(i))
             for i in range(1, len(run['cards']) + 1)]]
    if number is not None:
        rows += [[InlineKeyboardButton(text='План проверки', callback_data=base + 'p:' + str(number))],
                 [InlineKeyboardButton(text='Выбрали для анализа', callback_data=base + 's:' + str(number))],
                 [InlineKeyboardButton(text='Уже делали', callback_data=base + 'd:' + str(number)),
                  InlineKeyboardButton(text='Не полезно', callback_data=base + 'n:' + str(number))],
                 [InlineKeyboardButton(text='Реакция Бусько', callback_data=base + 'b:' + str(number))]]
    return InlineKeyboardMarkup(inline_keyboard=rows)


async def send(bot, chat_id: int, text: str, markup=None) -> None:
    # Telegram measures UTF-16 code units. Plain text avoids HTML/Markdown injection.
    chunks, part, length = [], [], 0
    for char in text:
        units = 2 if ord(char) > 0xffff else 1
        if length + units > 3900:
            chunks.append(''.join(part))
            part, length = [], 0
        part.append(char)
        length += units
    if part:
        chunks.append(''.join(part))
    for index, chunk in enumerate(chunks):
        await bot.send_message(chat_id, chunk, parse_mode=None,
                               reply_markup=markup if index == len(chunks) - 1 else None)
