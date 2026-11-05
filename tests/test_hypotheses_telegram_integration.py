"""S4b red-first Telegram contracts: synthetic data and an offline aiogram bot.

No poller, Telegram network, production credentials, server, or model is used.
The real S4a store/service and the existing router/FIFO are exercised together.
"""
from __future__ import annotations

import asyncio
import html
import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.types import Chat, Message, Update

from tof_reliability.vitaly_bot.hypotheses_service import HypothesesService
from tof_reliability.vitaly_bot.models import BotConfig, EvidenceError
from tof_reliability.vitaly_bot.service import JobOutcome
from tof_reliability.vitaly_bot.telegram import TelegramRuntime, build_router


class FakeSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.calls = []

    async def close(self):
        pass

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b''

    async def make_request(self, bot, method, timeout=None):
        self.calls.append(method)
        if method.__api_method__ in ('sendMessage', 'editMessageText'):
            return Message(message_id=getattr(method, 'message_id', None) or len(self.calls),
                           date=datetime.now(timezone.utc),
                           chat=Chat(id=method.chat_id, type='private'), text=method.text)
        return True


class Knowledge:
    def load(self, profile_version, knowledge_package_id):
        return {'knowledge_version': 'synthetic-v1', 'interest_context': {'version': 'i1'},
                'dimension_projection': {}}


@pytest.fixture
def isolated_dir():
    root = Path(os.environ['WORKFLOW_TEST_DB_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='s4b-', dir=root) as directory:
        yield Path(directory)


def cards():
    return [{
        'number': n, 'capability_id': 'AN-REPAIR-EPISODES',
        'management_question': f'Вопрос {n}: отличаются ли повторные ремонты?',
        'decision': f'Решение {n}: выбрать оборудование для проверки',
        'scope': f'Периметр {n}: насосы',
        'metrics_method': f'Метод {n}: сравнить интервалы между ремонтами',
        'meeting_interest': {'interest_id': f'interest-{n}', 'topic_id': 'repairs',
                             'dimension_id': f'dimension-{n}', 'attribution': 'interpretation',
                             'reference': 'Встреча 2030-01-01, свидетельство 1234',
                             'fragment': 'PRIVATE_MEETING_QUOTE_DO_NOT_SHOW'},
        'source_references': ['ledger:synthetic_repairs'],
        'case_material_references': [], 'source_readiness': 'partial',
        'missing_evidence': ['GAP_MISSING_EPISODE_KEYS'],
        'falsification': f'Опровержение {n}: различий нет',
        'hard_questions': [f'Сложный вопрос {n}: как определить эпизод?'],
        'analyst_brief': f'Поручение аналитику {n}: проверить ключи и период',
        'novelty': 'unknown',
        'ranking': {'utility': 4, 'feasibility': 3, 'interest': 4, 'novelty': 0},
    } for n in range(1, 7)]


class Harness:
    def __init__(self, directory):
        self.session = FakeSession()
        self.bot = Bot('123456:TEST_TOKEN', session=self.session)
        config = BotConfig(token='test', state_dir=directory / 'state',
                           knowledge_dir=directory / 'knowledge',
                           codex_bin=directory / 'no-codex', min_free_bytes=0)
        self.runtime = TelegramRuntime(self.bot, config)
        self.hypotheses = HypothesesService(store=self.runtime.store, config=config,
            runner=self.runtime.service.runner, coverage_provider=None,
            knowledge_loader=Knowledge(), trusted_store=None)
        self.runtime.service.hypotheses_service = self.hypotheses
        self.dispatcher = Dispatcher()
        self.dispatcher.include_router(build_router(self.runtime))
        self.number = 0

    async def close(self):
        for task in list(self.runtime._progress_tasks.values()):
            task.cancel()
        await asyncio.gather(*self.runtime._progress_tasks.values(), return_exceptions=True)
        await self.bot.session.close()

    async def message(self, text, chat=101, user=None):
        self.number += 1
        message = {'message_id': self.number, 'date': 1, 'chat': {'id': chat, 'type': 'private'},
                   'from': {'id': chat if user is None else user, 'is_bot': False, 'first_name': 'Test'},
                   'text': text}
        await self.dispatcher.feed_update(self.bot, Update.model_validate(
            {'update_id': self.number, 'message': message}))
        await asyncio.sleep(0)

    async def click(self, data, chat=101, user=None, callback_id=None):
        self.number += 1
        update = {'update_id': self.number, 'callback_query': {
            'id': callback_id or str(self.number),
            'from': {'id': chat if user is None else user, 'is_bot': False, 'first_name': 'Test'},
            'chat_instance': 'synthetic', 'data': data,
            'message': {'message_id': 99, 'date': 1, 'chat': {'id': chat, 'type': 'private'}}}}
        await self.dispatcher.feed_update(self.bot, Update.model_validate(update))
        await asyncio.sleep(0)

    def text(self, start=0):
        return '\n'.join(str(getattr(call, 'text', '') or '') for call in self.session.calls[start:])

    def button(self, *needles):
        for call in reversed(self.session.calls):
            markup = getattr(call, 'reply_markup', None)
            for row in getattr(markup, 'inline_keyboard', []) or []:
                for button in row:
                    if any(needle.casefold() in button.text.casefold() for needle in needles):
                        assert button.callback_data, 'кнопка должна передавать callback_data'
                        assert len(button.callback_data.encode('utf-8')) <= 64
                        return button.callback_data
        pytest.fail('не найдена кнопка: ' + ', '.join(needles))

    def run_row(self, job, owner=101):
        return self.hypotheses.get_run(user_id=owner, job_id=job['id'])

    async def completed(self, case=None, content=None, owner=101):
        case = case or self.runtime.service.new_case(owner)
        job = self.hypotheses.submit(case_id=case['id'], chat_id=owner, user_id=owner)
        run = self.run_row(job, owner)
        content = cards() if content is None else content
        with self.runtime.store._connect() as db:
            db.execute('UPDATE hypotheses_runs SET cards_json=?, generation_id=?, coverage_json=? WHERE id=?',
                       (json.dumps(content, ensure_ascii=False), 'synthetic-generation',
                        json.dumps({'generation': {'run_id': 'synthetic-generation'}}), run['id']))
        self.runtime.store.finish_job(job['id'])
        await self.runtime._completed(job, JobOutcome('hypotheses', case['id'],
                                                      {'run_id': run['id'], 'cards': content}))
        return case, job, self.run_row(job, owner)

    async def open_first_card(self):
        await self.click(self.button('H1', 'Н1', '1'))


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize('description', ['', 'насосы цеха А'])
def test_command_starts_without_files_in_shared_fifo_and_preserves_description(isolated_dir, description):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            await h.message('/hypotheses' + (' ' + description if description else ''))
            case = h.runtime.service.active_case(101)
            job = h.runtime.store.next_queued_work()
            assert job is not None and job['kind'] == 'hypotheses', 'команда должна сразу ставить задачу'
            assert h.run_row(job)['user_request'] == description
            assert h.runtime.store.artifacts(case['id']) == []
            assert re.search(r'материал|файл', h.text(), re.I), 'нужно сообщить об использованных материалах'
            assert re.search(r'нет|без|0|отсутств', h.text(), re.I)
            await h.message('/status')
            assert re.search(r'очеред|гипотез|ожида', h.text(), re.I)
        finally:
            await h.close()
    run(scenario())


def test_current_material_is_reported_and_expired_material_is_excluded(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            await h.message('CURRENT_CASE_SYNTHETIC_MATERIAL')
            case = h.runtime.service.active_case(101)
            artifacts = h.runtime.store.artifacts(case['id'])
            assert len(artifacts) == 1
            start = len(h.session.calls)
            await h.message('/hypotheses детали')
            job = h.runtime.store.next_queued_work()
            assert job and job['kind'] == 'hypotheses'
            assert json.loads(h.run_row(job)['artifact_snapshot_json'])[0]['id'] == artifacts[0]['id']
            assert artifacts[0]['original_name'] in h.text(start), 'показать фактически взятый материал'
            h.runtime.store.finish_job(job['id'])
            h.runtime.store._update_case(case['id'], last_user_activity_at=(
                datetime.now(timezone.utc) - timedelta(days=5)).isoformat())
            await h.message('/hypotheses')
            new_case = h.runtime.service.active_case(101)
            assert new_case['id'] != case['id']
            new_job = h.runtime.store.next_queued_work()
            assert new_job and new_job['case_id'] == new_case['id']
            assert json.loads(h.run_row(new_job)['artifact_snapshot_json']) == []
        finally:
            await h.close()
    run(scenario())


def test_rehearsal_requires_stop_and_busy_admission_is_atomic(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            case = h.runtime.service.new_case(101)
            h.runtime.store.set_question(case['id'], {'text': 'Где данные?'}, 0, 'grilling')
            await h.message('/hypotheses')
            assert '/stop' in h.text()
            assert h.runtime.store.next_queued_work() is None
            assert h.runtime.store.get_case(case['id'])['mode'] == 'grilling'
            await h.message('/stop')
            await h.message('/hypotheses')
            first = h.runtime.store.next_queued_work()
            assert first and first['kind'] == 'hypotheses'
            await asyncio.gather(h.message('/hypotheses один'), h.message('/hypotheses два'))
            with h.runtime.store._connect() as db:
                assert db.execute('SELECT COUNT(*) FROM hypotheses_runs WHERE case_id=?', (case['id'],)).fetchone()[0] == 1
                assert db.execute("SELECT COUNT(*) FROM jobs WHERE case_id=? AND status IN ('queued','running')", (case['id'],)).fetchone()[0] == 1
            assert re.search(r'занят|выполня|очеред|подожд', h.text(), re.I)
        finally:
            await h.close()
    run(scenario())


def test_fifo_recovery_and_failure_leave_retry_possible(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            await h.message('/hypotheses первый', 101)
            first = h.runtime.store.next_queued_work()
            assert first and first['kind'] == 'hypotheses'
            await h.message('/hypotheses второй', 202)
            second_case = h.runtime.service.active_case(202)
            second = h.runtime.store.get_job(second_case['active_job_id'])
            assert h.runtime.store.queue_position(second['id']) == 2
            h.runtime.store.mark_job_running(first['id'])
            h.runtime.store.recover_jobs()
            assert h.runtime.store.next_queued_work()['id'] == first['id']
            h.runtime.store.finish_job(first['id'], error='tof_unavailable')
            await h.runtime._failed(first, EvidenceError('tof_unavailable', 'synthetic'))
            assert re.search(r'недоступ|ошиб|не удалось', h.text(), re.I)
            assert h.runtime.store.next_queued_work()['id'] == second['id']
            await h.message('/hypotheses снова', 101)
            assert h.runtime.service.active_case(101)['active_job_id'] not in (None, first['id'])
        finally:
            await h.close()
    run(scenario())


def test_late_delivery_does_not_clear_new_active_job(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            case = h.runtime.service.new_case(101)
            first = h.hypotheses.submit(case_id=case['id'], chat_id=101, user_id=101)
            result = h.run_row(first)
            with h.runtime.store._connect() as db:
                db.execute('UPDATE hypotheses_runs SET cards_json=? WHERE id=?',
                           (json.dumps(cards(), ensure_ascii=False), result['id']))
            h.runtime.store.finish_job(first['id'])
            second = h.hypotheses.submit(case_id=case['id'], chat_id=101, user_id=101)
            await h.runtime._completed(first, JobOutcome('hypotheses', case['id'], {'run_id': result['id']}))
            assert h.runtime.store.get_case(case['id'])['active_job_id'] == second['id']
            assert h.runtime.store.get_case(case['id'])['activity'] == 'queued'
        finally:
            await h.close()
    run(scenario())


def test_cards_render_saved_content_with_attribution_gaps_and_unknown_novelty(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            _, _, result = await h.completed()
            assert 5 <= len(result['cards']) <= 7
            await h.open_first_card()
            output = html.unescape(h.text())
            for value in ('Вопрос 1', 'Решение 1', 'Периметр 1', 'Метод 1',
                          'Встреча 2030-01-01', 'ledger:synthetic_repairs',
                          'GAP_MISSING_EPISODE_KEYS', 'Опровержение 1',
                          'Сложный вопрос 1', 'Поручение аналитику 1'):
                assert value in output, f'в карточке нет {value}'
            assert re.search(r'частич|не полн', output, re.I)
            assert re.search(r'новизн\w*[^\n]{0,60}(неизвест|не определ|не подтвержд)', output, re.I)
            assert re.search(r'интерпретац|предполож|косвен', output, re.I)
            assert 'PRIVATE_MEETING_QUOTE_DO_NOT_SHOW' not in output
            assert not re.search(r'\b\d+\s*%', output)
        finally:
            await h.close()
    run(scenario())


def test_navigation_and_saved_plan_do_not_enqueue_model_work(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            _, job, before = await h.completed()
            await h.open_first_card()
            mark = len(h.session.calls)
            await h.click(h.button('План проверки', 'План'))
            plan = h.text(mark)
            assert 'Метод 1' in plan and 'Поручение аналитику 1' in plan
            assert 'GAP_MISSING_EPISODE_KEYS' in plan
            assert h.runtime.store.next_queued_work() is None
            assert h.run_row(job)['revision'] == before['revision']
            # A page or card selector must let the user reach a different card.
            await h.click(h.button('H2', 'Н2', '2', 'далее', 'следующ'))
            assert re.search(r'H2|Н2|Вопрос 2|Решение 2', h.text(mark), re.I)
            assert h.runtime.store.next_queued_work() is None
        finally:
            await h.close()
    run(scenario())


@pytest.mark.parametrize(('label', 'decision'), [
    ('Выбрали для анализа', 'selected'), ('Уже делали', 'already_done'),
    ('Не полезно', 'not_useful'), ('Реакция Бусько', 'busko_reaction'),
])
def test_feedback_buttons_update_current_run_and_persist(isolated_dir, label, decision):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            case, job, original = await h.completed()
            await h.open_first_card()
            callback = h.button(label)
            await h.click(callback)
            saved = h.run_row(job)
            feedback = saved['cards'][0].get('feedback', {})
            key = 'reaction' if decision == 'busko_reaction' else 'decision'
            assert feedback.get(key) == decision
            assert saved['revision'] == original['revision'] + 1
            reopened = Harness(isolated_dir)
            try:
                assert reopened.run_row(job)['cards'][0]['feedback'][key] == decision
            finally:
                await reopened.close()
            assert not h.runtime.store.artifacts(case['id'])
            assert not h.runtime.store.turns(case['id'])
            assert h.runtime.store.next_queued_work() is None
        finally:
            await h.close()
    run(scenario())


def test_feedback_text_survives_restart_is_isolated_from_case_and_can_be_cancelled(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            case, job, _ = await h.completed()
            await h.open_first_card()
            await h.click(h.button('Реакция Бусько'))
            reopened = Harness(isolated_dir)
            try:
                await reopened.message('PRIVATE_CASE_FEEDBACK_SYNTHETIC')
                result = reopened.run_row(job)
                assert result['cards'][0]['feedback']['text'] == 'PRIVATE_CASE_FEEDBACK_SYNTHETIC'
                assert not reopened.runtime.store.artifacts(case['id'])
                assert not reopened.runtime.store.turns(case['id'])
                assert reopened.runtime.store.next_queued_work() is None
                await reopened.runtime._completed(job, JobOutcome('hypotheses', case['id'],
                    {'run_id': result['id'], 'cards': result['cards']}))
                await reopened.open_first_card()
                await reopened.click(reopened.button('Реакция Бусько'))
                await reopened.message('/cancel')
                assert re.search(r'отмен', reopened.text(), re.I)
                assert reopened.run_row(job)['cards'][0]['feedback']['text'] == 'PRIVATE_CASE_FEEDBACK_SYNTHETIC'
                with reopened.runtime.store._connect() as db:
                    assert db.execute('SELECT COUNT(*) FROM hypotheses_feedback_input WHERE user_id=101').fetchone()[0] == 0
            finally:
                await reopened.close()
        finally:
            await h.close()
    run(scenario())


def test_replayed_feedback_message_never_becomes_case_material(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            case, job, _ = await h.completed()
            await h.open_first_card()
            await h.click(h.button('Реакция Бусько'))
            await h.message('SYNTHETIC_PRIVATE_REPLAY')
            message_id = h.number
            h.number = message_id - 1
            await h.message('SYNTHETIC_PRIVATE_REPLAY')
            assert h.run_row(job)['cards'][0]['feedback']['text'] == 'SYNTHETIC_PRIVATE_REPLAY'
            assert not h.runtime.store.artifacts(case['id'])
            assert not h.runtime.store.turns(case['id'])
        finally:
            await h.close()
    run(scenario())


def test_feedback_input_cannot_be_overlapped_by_meeting_or_other_buttons(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            _, job, _ = await h.completed()
            await h.open_first_card()
            await h.click(h.button('Реакция Бусько'))
            selected = h.button('Выбрали для анализа')
            before = h.run_row(job)['revision']
            await h.message('/meeting')
            assert h.runtime.store.pending_meeting(101, 101) is None
            await h.click(selected)
            assert h.run_row(job)['revision'] == before
            await h.message('текст реакции')
            assert h.run_row(job)['cards'][0]['feedback']['text'] == 'текст реакции'
        finally:
            await h.close()
    run(scenario())


def test_queue_wakes_even_if_material_confirmation_fails(isolated_dir, monkeypatch):
    from tof_reliability.vitaly_bot import hypotheses_ui
    async def failing_send(*args, **kwargs):
        raise RuntimeError('synthetic delivery failure')
    async def scenario():
        h = Harness(isolated_dir)
        try:
            monkeypatch.setattr(hypotheses_ui, 'send', failing_send)
            await h.message('/hypotheses')
            job = h.runtime.store.next_queued_work()
            assert job and job['kind'] == 'hypotheses'
            assert h.runtime.worker._wake.is_set()
        finally:
            await h.close()
    run(scenario())


@pytest.mark.parametrize('invalidation', ['foreign', 'double', 'new_case', 'expired', 'new_run'])
def test_stale_foreign_and_duplicate_callbacks_do_not_mutate_run(isolated_dir, invalidation):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            case, job, initial = await h.completed()
            await h.open_first_card()
            callback = h.button('Выбрали для анализа')
            if invalidation == 'foreign':
                await h.click(callback, user=202)
            elif invalidation == 'double':
                await h.click(callback, callback_id='same-tap')
                once = h.run_row(job)
                await h.click(callback, callback_id='same-tap')
                await h.click(callback, callback_id='another-tap')
                assert h.run_row(job)['revision'] == once['revision'] == initial['revision'] + 1
                return
            elif invalidation == 'new_case':
                h.runtime.service.new_case(101)
                await h.click(callback)
            elif invalidation == 'expired':
                h.runtime.store._update_case(case['id'], last_user_activity_at=(
                    datetime.now(timezone.utc) - timedelta(days=5)).isoformat())
                await h.click(callback)
            else:
                newer = h.hypotheses.submit(case_id=case['id'], chat_id=101, user_id=101)
                h.runtime.store.finish_job(newer['id'])
                await h.click(callback)
            assert h.run_row(job)['revision'] == initial['revision']
            assert not h.run_row(job)['cards'][0].get('feedback')
            assert re.search(r'устар|ист[её]к|найден|доступ|кейс|кнопк', h.text(), re.I)
        finally:
            await h.close()
    run(scenario())


class StrictHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack and self.stack.pop() == tag, 'некорректная HTML-разметка Telegram'


def test_telegram_length_escaping_and_callback_sizes_are_safe(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            content = cards()
            content[0]['management_question'] = 'Сравнить <насос & участок> ' + ('длинные данные ' * 800)
            await h.completed(content=content)
            await h.open_first_card()
            for call in h.session.calls:
                text = getattr(call, 'text', None)
                if not text:
                    continue
                assert len(text.encode('utf-16-le')) // 2 <= 4096
                if str(getattr(call, 'parse_mode', '')).lower() == 'html':
                    parser = StrictHTML()
                    parser.feed(text)
                    parser.close()
                    assert not parser.stack
                    assert '<насос & участок>' not in text
                markup = getattr(call, 'reply_markup', None)
                for row in getattr(markup, 'inline_keyboard', []) or []:
                    for button in row:
                        if button.callback_data:
                            assert len(button.callback_data.encode()) <= 64
        finally:
            await h.close()
    run(scenario())


@pytest.mark.parametrize(('attribution', 'expected'), [
    ('direct_request', 'прямой запрос'), ('third_party_paraphrase', 'пересказ'),
    ('third_party', 'треть'), ('interpretation', 'интерпретац'),
])
def test_card_reports_actual_attribution_and_source_availability(attribution, expected):
    from tof_reliability.vitaly_bot.hypotheses_ui import render
    card = cards()[0]
    card['meeting_interest']['attribution'] = attribution
    card['source_readiness'] = 'available'
    value = render(card).lower()
    assert expected in value
    assert re.search(r'источник\w*: (доступ|подтвержд)', value)


def test_hypothesis_feedback_enters_separate_learning_lane_without_raw_case_facts(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            case, job, _ = await h.completed()
            await h.open_first_card()
            await h.click(h.button('Реакция Бусько'))
            await h.message('PRIVATE_BUSINESS_FACT_SYNTHETIC: проверяли на конкретном участке')
            saved = h.run_row(job)
            assert saved['cards'][0]['feedback']['text'].startswith('PRIVATE_BUSINESS_FACT_SYNTHETIC')
            assert not h.runtime.store.artifacts(case['id']) and not h.runtime.store.turns(case['id'])
            # The automatic 04:00 cycle must have an actionable, separate path:
            # a quality-gated learning source or its queued assessment. Merely
            # retaining a hypotheses_feedback_events row is not integration.
            with h.runtime.store._connect() as db:
                queued = [dict(row) for row in db.execute(
                    "SELECT * FROM learning_jobs WHERE status IN ('queued','running','scheduled')")]
            sources = h.runtime.store.eligible_learning_sources()
            assert queued or sources, 'обратная связь должна попасть в отдельный проверяемый цикл обучения'
            assert 'PRIVATE_BUSINESS_FACT_SYNTHETIC' not in json.dumps(sources, ensure_ascii=False)
            assert not re.search(r'уже (?:выуч|добавлен|учт[её]н[оа] в общ)', h.text(), re.I)
            assert h.runtime.config.learning_run_hour == 4
        finally:
            await h.close()
    run(scenario())


def test_legacy_help_and_review_without_materials_still_work(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            await h.message('/help')
            assert '/review' in h.text() and '/grill' in h.text()
            await h.message('/review')
            assert re.search(r'файл|материал', h.text(), re.I)
            assert h.runtime.store.next_queued_work() is None
        finally:
            await h.close()
    run(scenario())


def test_daily_feedback_assessment_exports_only_closed_safe_rules(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            _, _, _ = await h.completed()
            await h.open_first_card()
            await h.click(h.button('Реакция Бусько'))
            await h.message('PRIVATE_BUSINESS_FACT_SYNTHETIC: следует разделять факт и сценарий')
            manager = h.runtime.service.learning
            prompts = []
            async def model(job, kind, prompt, schema, images=None):
                prompts.append((kind, prompt))
                if kind == 'hypotheses_feedback_assess':
                    return {'transfer_key': 'separate_fact_from_scenario'}
                assert kind == 'hypotheses_feedback_verify'
                return {'supported': True}
            manager._model = model
            await manager._prepare_hypothesis_feedback({'id': 'synthetic-job'})
            sources = h.runtime.store.eligible_learning_sources()
            assert sources
            assert 'PRIVATE_BUSINESS_FACT_SYNTHETIC' not in json.dumps(sources, ensure_ascii=False)
            assert 'separate_fact_from_scenario' in json.dumps(sources, ensure_ascii=False)
            assert len(prompts) >= 2
        finally:
            await h.close()
    run(scenario())


def test_feedback_cycle_runs_at_most_once_per_day_and_respects_pause(isolated_dir):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            await h.completed()
            await h.open_first_card()
            await h.click(h.button('Реакция Бусько'))
            await h.message('описание методического замечания')
            now = datetime(2031, 3, 1, 4, 0, tzinfo=timezone.utc)
            h.runtime.store.set_learning_paused(True, actor_user_id=101)
            assert not h.runtime.store.resume_scheduled_learning(4, now=now)
            h.runtime.store.set_learning_paused(False, actor_user_id=101)
            assert h.runtime.store.resume_scheduled_learning(4, now=now)
            assert not h.runtime.store.due_learning_cycle(4, now=now)
            with h.runtime.store._connect() as db:
                db.execute("UPDATE learning_jobs SET status='completed' WHERE kind='knowledge_generate'")
                h.runtime.store._schedule_hypotheses_learning(db)
            assert not h.runtime.store.resume_scheduled_learning(4, now=now)
            assert h.runtime.store.resume_scheduled_learning(4, now=now + timedelta(days=1))
        finally:
            await h.close()
    run(scenario())


def test_development_service_is_explicitly_opted_in_and_shares_runner(isolated_dir, monkeypatch):
    from tof_reliability.vitaly_bot.config import load_config
    from tof_reliability.vitaly_bot import hypotheses_knowledge
    class Loader:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def load(self, profile_version, knowledge_package_id):
            return {'knowledge_version': 'development', 'interest_context': {'version': 'i1'},
                    'dimension_projection': {}}
    monkeypatch.setattr(hypotheses_knowledge, 'VerifiedKnowledgeLoader', Loader)
    env = {'VITALY_BOT_ENV_FILE': str(isolated_dir / 'absent'),
           'VITALY_BOT_STATE_DIR': str(isolated_dir / 'dev-state'),
           'VITALY_BOT_HYPOTHESES_DEVELOPMENT': '1',
           'VITALY_BOT_HYPOTHESES_CORPUS_DIR': str(isolated_dir / 'corpus'),
           'VITALY_BOT_HYPOTHESES_ORIGINAL_DIR': str(isolated_dir / 'original'),
           'VITALY_BOT_HYPOTHESES_ADJUDICATION_DIR': str(isolated_dir / 'adjudication'),
           'VITALY_BOT_HYPOTHESES_MANIFEST_SHA256': 'a' * 64}
    config = load_config(env, require_token=False)
    session = FakeSession()
    bot = Bot('123456:TEST_TOKEN', session=session)
    runtime = TelegramRuntime(bot, config)
    assert runtime.service.hypotheses_service is not None
    assert runtime.service.hypotheses_service.runner is runtime.service.runner
    case = runtime.service.new_case(101)
    job = runtime.service.hypotheses_service.submit(case_id=case['id'], chat_id=101, user_id=101)
    assert job['kind'] == 'hypotheses'
    run(bot.session.close())


def test_hypothesis_feedback_passes_existing_candidate_and_verification_gate(isolated_dir):
    from test_vitaly_learning_runtime import LearningRunner
    async def scenario():
        h = Harness(isolated_dir)
        try:
            case, _, _ = await h.completed()
            await h.open_first_card()
            await h.click(h.button('Реакция Бусько'))
            await h.message('PRIVATE_BUSINESS_FACT_SYNTHETIC: сравнивать определения до вывода о противоречии')
            manager = h.runtime.service.learning
            runner = LearningRunner()
            async def model(job, kind, prompt, schema, images=None):
                if kind == 'hypotheses_feedback_assess':
                    return {'transfer_key': 'compare_definition_before_conflict'}
                if kind == 'hypotheses_feedback_verify':
                    return {'supported': True}
                assert 'PRIVATE_BUSINESS_FACT_SYNTHETIC' not in prompt
                sources = h.runtime.store.eligible_learning_sources()
                if kind == 'knowledge_generate':
                    runner.feedback_id = sources[0]['id']
                result = (await runner.run(prompt, schema)).value
                if kind == 'knowledge_generate':
                    result['patch']['analysis_rules'][0]['evidence_references'][0]['source_document_id'] = sources[0]['result']['source_document_id']
                return result
            manager._model = model
            with h.runtime.store._connect() as db:
                job_id = db.execute("SELECT id FROM learning_jobs WHERE kind='knowledge_generate'").fetchone()[0]
            generated = await manager.execute(h.runtime.store.get_learning_job(job_id))
            assert generated.kind == 'knowledge_candidate'
            verification = await manager.execute(generated.payload['verification_job'])
            assert verification.kind == 'knowledge_enabled'
            new_case = h.runtime.service.new_case(202)
            rules = manager.analysis_rules_for_case(new_case['id'])
            assert any(rule['id'] == 'compare_definition_before_conflict' for rule in rules)
            assert 'PRIVATE_BUSINESS_FACT_SYNTHETIC' not in json.dumps(rules, ensure_ascii=False)
            assert not h.runtime.store.artifacts(case['id'])
        finally:
            await h.close()
    run(scenario())
