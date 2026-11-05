"""Offline dispatcher contracts for navigating a saved seven-card run.

All messages, cards and storage are synthetic.  No poller or model is started.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone

import pytest

from test_hypotheses_telegram_integration import Harness, cards, isolated_dir  # noqa: F401


def seven_cards():
    result = cards()
    seventh = dict(result[-1])
    seventh.update(number=7, management_question='Вопрос 7: синтетический пример')
    result.append(seventh)
    return result


def keyboard_since(h, start):
    """Only accept an actually delivered keyboard from this specific response."""
    for call in reversed(h.session.calls[start:]):
        markup = getattr(call, 'reply_markup', None)
        if getattr(markup, 'inline_keyboard', None):
            return {button.text: button.callback_data
                    for row in markup.inline_keyboard for button in row}
    pytest.fail('ответ должен содержать новую клавиатуру навигации')


def assert_keyboard(keyboard, run, selected):
    assert all(f'H{n}' in keyboard for n in range(1, 8))
    for number in range(1, 8):
        assert keyboard[f'H{number}'] == f"hy:{run['id']}:{run['revision']}:v:{number}"
    for label in ('План проверки', 'Выбрали для анализа', 'Реакция Бусько'):
        assert keyboard[label].endswith(f':{selected}')
        assert keyboard[label].startswith(f"hy:{run['id']}:{run['revision']}:")


def event_count(h, run_id):
    with h.runtime.store._connect() as db:
        return db.execute('SELECT COUNT(*) FROM hypotheses_feedback_events WHERE run_id=?',
                          (run_id,)).fetchone()[0]


def jobs_count(h):
    with h.runtime.store._connect() as db:
        return db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0]


def forbid_model(h, monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail('навигация и обратная связь не должны вызывать модель')
    monkeypatch.setattr(h.runtime.service.runner, 'run', forbidden)


def test_text_feedback_replay_multiple_events_and_cancel_keep_navigation(isolated_dir, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            forbid_model(h, monkeypatch)
            case, job, original = await h.completed(content=seven_cards())
            count = jobs_count(h)
            await h.open_first_card()
            await h.click(h.button('План проверки'))
            assert 'Метод 1' in h.text()
            await h.click(h.button('Реакция Бусько'))
            old_keyboard = keyboard_since(h, 0)
            after_decision = h.run_row(job)
            assert after_decision['revision'] == original['revision'] + 1

            mark = len(h.session.calls)
            await h.message('Синтетическая реакция один')
            message_id = h.number
            saved = h.run_row(job)
            assert saved['revision'] == after_decision['revision'] + 1
            assert saved['cards'][0]['feedback']['text'] == 'Синтетическая реакция один'
            assert event_count(h, saved['id']) == 2
            fresh = keyboard_since(h, mark)
            assert_keyboard(fresh, saved, 1)

            # The same Telegram message must not become a second event or case material.
            h.number = message_id - 1
            await h.message('Синтетическая реакция один')
            assert h.run_row(job)['revision'] == saved['revision']
            assert event_count(h, saved['id']) == 2
            assert not h.runtime.store.artifacts(case['id'])
            assert not h.runtime.store.turns(case['id'])

            # The pre-text keyboard is genuinely stale; a foreign user cannot use the fresh one.
            mark = len(h.session.calls)
            await h.click(old_keyboard['H2'])
            assert re.search(r'устар|недоступ|ист[её]к', h.text(mark), re.I)
            mark = len(h.session.calls)
            await h.click(fresh['H2'], user=202)
            assert re.search(r'устар|недоступ|другому|найден', h.text(mark), re.I)
            assert h.run_row(job)['revision'] == saved['revision']

            mark = len(h.session.calls)
            await h.click(fresh['H2'])
            assert 'Вопрос 2' in h.text(mark)
            card_two = keyboard_since(h, mark)
            assert_keyboard(card_two, saved, 2)
            mark = len(h.session.calls)
            await h.click(card_two['Выбрали для анализа'], callback_id='synthetic-select-two')
            selected = h.run_row(job)
            assert selected['cards'][1]['feedback']['decision'] == 'selected'
            assert selected['revision'] == saved['revision'] + 1
            assert event_count(h, saved['id']) == 3
            selected_keyboard = keyboard_since(h, mark)
            assert_keyboard(selected_keyboard, selected, 2)
            await h.click(card_two['Выбрали для анализа'], callback_id='synthetic-select-two')
            assert h.run_row(job)['revision'] == selected['revision']
            assert event_count(h, saved['id']) == 3

            mark = len(h.session.calls)
            await h.click(selected_keyboard['H3'])
            assert 'Вопрос 3' in h.text(mark)
            card_three = keyboard_since(h, mark)
            await h.click(card_three['Реакция Бусько'])
            awaiting = h.run_row(job)
            assert awaiting['revision'] == selected['revision'] + 1
            assert event_count(h, saved['id']) == 4
            mark = len(h.session.calls)
            await h.message('/cancel')
            assert re.search(r'отмен', h.text(mark), re.I)
            cancelled_keyboard = keyboard_since(h, mark)
            assert_keyboard(cancelled_keyboard, awaiting, 3)
            assert h.run_row(job)['revision'] == awaiting['revision']
            assert event_count(h, saved['id']) == 4
            mark = len(h.session.calls)
            await h.click(cancelled_keyboard['H4'])
            assert 'Вопрос 4' in h.text(mark)
            assert jobs_count(h) == count
            assert h.runtime.store.next_queued_work() is None
        finally:
            await h.close()
    asyncio.run(scenario())


def test_feedback_confirmation_send_failure_and_redelivery_are_idempotent(isolated_dir, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            forbid_model(h, monkeypatch)
            case, job, _ = await h.completed(content=seven_cards())
            count = jobs_count(h)
            await h.open_first_card()
            await h.click(h.button('Реакция Бусько'))
            request = h.session.make_request
            failed = False

            async def fail_one_send(bot, method, timeout=None):
                nonlocal failed
                if not failed and method.__api_method__ == 'sendMessage':
                    failed = True
                    raise RuntimeError('synthetic confirmation delivery failure')
                return await request(bot, method, timeout=timeout)

            monkeypatch.setattr(h.session, 'make_request', fail_one_send)
            await h.message('Синтетическая реакция после сбоя')
            message_id = h.number
            saved = h.run_row(job)
            assert failed
            assert saved['cards'][0]['feedback']['text'] == 'Синтетическая реакция после сбоя'
            assert event_count(h, saved['id']) == 2
            mark = len(h.session.calls)
            h.number = message_id - 1
            await h.message('Синтетическая реакция после сбоя')
            assert h.run_row(job)['revision'] == saved['revision']
            assert event_count(h, saved['id']) == 2
            assert not h.runtime.store.artifacts(case['id'])
            keyboard = keyboard_since(h, mark)
            assert_keyboard(keyboard, saved, 1)
            mark = len(h.session.calls)
            await h.click(keyboard['H2'])
            assert 'Вопрос 2' in h.text(mark)
            assert jobs_count(h) == count
            assert h.runtime.store.next_queued_work() is None
        finally:
            await h.close()
    asyncio.run(scenario())


def test_old_text_replay_uses_current_revision_and_does_not_revive_old_case(isolated_dir, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            forbid_model(h, monkeypatch)
            case, job, _ = await h.completed(content=seven_cards())
            await h.open_first_card()
            await h.click(h.button('Реакция Бусько'))
            await h.message('Первая синтетическая реакция')
            first_message_id = h.number
            await h.click(h.button('Выбрали для анализа'))
            current = h.run_row(job)
            count = event_count(h, current['id'])
            mark = len(h.session.calls)
            h.number = first_message_id - 1
            await h.message('Первая синтетическая реакция')
            assert_keyboard(keyboard_since(h, mark), current, 1)
            assert event_count(h, current['id']) == count
            await h.message('/new')
            mark = len(h.session.calls)
            h.number = first_message_id - 1
            await h.message('Первая синтетическая реакция')
            assert not any(getattr(getattr(call, 'reply_markup', None), 'inline_keyboard', None)
                           for call in h.session.calls[mark:])
            assert event_count(h, current['id']) == count
            assert not h.runtime.store.artifacts(case['id'])
        finally:
            await h.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('invalidation', ['new', 'expired'])
def test_refreshed_keyboard_still_respects_case_lifetime(isolated_dir, monkeypatch, invalidation):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            forbid_model(h, monkeypatch)
            case, job, _ = await h.completed(content=seven_cards())
            count = jobs_count(h)
            await h.open_first_card()
            await h.click(h.button('Реакция Бусько'))
            mark = len(h.session.calls)
            await h.message('Синтетическая реакция')
            saved = h.run_row(job)
            keyboard = keyboard_since(h, mark)
            assert_keyboard(keyboard, saved, 1)
            if invalidation == 'new':
                await h.message('/new')
                assert h.runtime.service.active_case(101)['id'] != case['id']
            else:
                h.runtime.store._update_case(case['id'], last_user_activity_at=(
                    datetime.now(timezone.utc) - timedelta(days=5)).isoformat())
            mark = len(h.session.calls)
            await h.click(keyboard['H2'])
            assert re.search(r'устар|ист[её]к|найден|доступ|кейс', h.text(mark), re.I)
            assert 'Вопрос 2' not in h.text(mark)
            assert h.run_row(job)['revision'] == saved['revision']
            assert event_count(h, saved['id']) == 2
            assert jobs_count(h) == count
            assert h.runtime.store.next_queued_work() is None
        finally:
            await h.close()
    asyncio.run(scenario())
