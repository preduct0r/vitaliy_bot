import asyncio
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from tof_reliability.vitaly_bot.corrections import Corrections, item_at, correct_point
from tof_reliability.vitaly_bot.storage import VitalyStore
from tof_reliability.vitaly_bot.models import StateError, ModelError, BotConfig, ModelResult, ArtifactInput
from tof_reliability.vitaly_bot.profile import load_default_profile
from tof_reliability.vitaly_bot.service import VitalyService


def report():
    q = dict(text='Где число?', topic='evidence_scope', likelihood=3, severity=4,
             trigger='Не видно', material_reference='input.txt', expected_answer='Число 30',
             expected_answer_evidence_level='material', expected_answer_references=['input.txt'],
             followups=[], profile_evidence=['2026-06-03'], confidence='high')
    return dict(readiness_score=55, verdict='Нужна проверка',
                critical_gaps=[dict(title='Нет числа', detail='Не найдено', material_reference='input.txt')],
                contradictions=[dict(title='Разные периоды', detail='Уточнить', material_reference='input.txt')],
                questions=[q, dict(q, text='Кто проверит?')], prep_checklist=['Проверить цифры'])


@pytest.fixture
def setup(tmp_path):
    store = VitalyStore(tmp_path / 'state')
    config = BotConfig('test', store.state_dir, Path('/none'))
    service = VitalyService(store, load_default_profile(), config)
    case = service.new_case(101)
    source = tmp_path / 'input.txt'
    source.write_text('Показатель 30. Период — январь. Источник указан в таблице.')
    service.documents.ingest(case['id'], ArtifactInput('file', 'input.txt', source, source.stat().st_size))
    store.set_review(case['id'], report())
    audit = service.corrections.ensure(case['id'])
    return store, service, case, audit


def submit(setup, key='G1'):
    store, service, case, audit = setup
    audit = service.corrections.ensure(case['id'])
    service.corrections.select(audit['id'], audit['revision'], key, 101)
    return service.corrections.submit(audit['id'], 101, 'Число 30 уже указано')


def test_telegram_selection_answer_and_current_report(setup):
    pytest.importorskip('aiogram')
    from aiogram import Bot, Dispatcher
    from aiogram.client.session.base import BaseSession
    from aiogram.types import Update, Message, Chat
    from tof_reliability.vitaly_bot.telegram import TelegramRuntime, build_router
    from datetime import datetime, timezone

    class Session(BaseSession):
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
            if method.__api_method__ == 'sendMessage':
                return Message(message_id=len(self.calls), date=datetime.now(timezone.utc),
                               chat=Chat(id=101, type='private'), text=method.text)
            return True

    async def scenario():
        store, service, case, audit = setup
        session = Session()
        bot = Bot('123456:TEST_TOKEN', session=session)
        runtime = TelegramRuntime(bot, service.config)
        async def progress(*args):
            pass
        runtime._ensure_progress = progress
        dp = Dispatcher()
        dp.include_router(build_router(runtime))
        user = dict(id=101, is_bot=False, first_name='Test')
        message = dict(message_id=1, date=1, chat=dict(id=101, type='private'), **{'from': user})
        async def click(action, revision=0):
            await dp.feed_update(bot, Update.model_validate(dict(update_id=len(session.calls)+1,
                callback_query=dict(id='test', **{'from': user}, chat_instance='test',
                    message=message, data=f"vc:{audit['id']}:{revision}:{action}"))))
        await click('page:0')
        assert any(getattr(call, 'reply_markup', None) for call in session.calls)
        await click('pick:G1')
        assert runtime.service.corrections.pending(case['id'])['pending_item'] == 'G1'
        for i in range(2):
            await dp.feed_update(bot, Update.model_validate(dict(update_id=100+i,
                message=dict(message, text='В файле уже есть число 30'))))
        assert store.queue_summary()['queued'] == 1
        assert store.artifact_count(case['id']) == 1
        assert not store.turns(case['id'])
        job = store.next_queued_job()
        store.mark_job_running(job['id'])
        runtime.service.corrections.apply(job, result())
        store.finish_job(job['id'])
        store.settle_job(case['id'], mode='ready')
        await click('pick:G1')
        assert 'Отчёт обновился' in session.calls[-1].text
        await click('view')
        assert any('Показатель подтверждён' in (getattr(call, 'text', '') or '') for call in session.calls)
        assert any('не пересчитывались' in (getattr(call, 'text', '') or '') for call in session.calls)
        await bot.session.close()
    asyncio.run(scenario())


def test_telegram_general_feedback_and_meeting_are_separate_from_case(setup):
    pytest.importorskip('aiogram')
    from aiogram import Bot, Dispatcher
    from aiogram.client.session.base import BaseSession
    from aiogram.types import Update, Message, Chat
    from datetime import datetime, timezone
    from tof_reliability.vitaly_bot.telegram import TelegramRuntime, build_router

    class Session(BaseSession):
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
            if method.__api_method__ == 'sendMessage':
                return Message(message_id=len(self.calls), date=datetime.now(timezone.utc),
                               chat=Chat(id=101, type='private'), text=method.text)
            return True

    async def scenario():
        store, service, case, audit = setup
        session = Session()
        bot = Bot('123456:TEST_TOKEN', session=session)
        runtime = TelegramRuntime(bot, service.config)
        runtime._ensure_learning_typing = lambda job: None
        dp = Dispatcher()
        dp.include_router(build_router(runtime))
        user = dict(id=101, is_bot=False, first_name='Test')
        base = dict(message_id=1, date=1, chat=dict(id=101, type='private'), **{'from': user})

        await dp.feed_update(bot, Update.model_validate(dict(
            update_id=1,
            callback_query=dict(id='feedback', **{'from': user}, chat_instance='test', message=base,
                                data=f"vc:{audit['id']}:{audit['revision']}:feedback"),
        )))
        assert store.pending_feedback(101, 101) is not None
        # While a feedback explanation is pending, any non-text update must be
        # rejected before it can reach the ordinary case-ingest path.
        await dp.feed_update(bot, Update.model_validate(dict(update_id=11, message=dict(base, message_id=11,
            document=dict(file_id='blocked-document', file_unique_id='blocked-document-unique',
                          file_name='blocked.pdf', mime_type='application/pdf', file_size=10)))))
        await dp.feed_update(bot, Update.model_validate(dict(update_id=12, message=dict(base, message_id=12,
            photo=[dict(file_id='blocked-photo', file_unique_id='blocked-photo-unique', width=1, height=1, file_size=10)]))))
        assert store.pending_feedback(101, 101) is not None
        assert store.artifact_count(case['id']) == 1
        await dp.feed_update(bot, Update.model_validate(dict(update_id=2, message=dict(base, message_id=2,
            text='На двух слайдах разные определения показателя, это не конфликт.'))))
        assert store.pending_feedback(101, 101) is None
        assert store.get_learning_job(store.get_feedback(store.own_learning_status(101, 101)['feedback'][0]['id'])['learning_job_id'])['kind'] == 'feedback_analyze'
        assert store.artifact_count(case['id']) == 1
        await dp.feed_update(bot, Update.model_validate(dict(update_id=21, message=dict(base, message_id=21,
            text='Это повтор: не превращайте его в материал.'))))
        assert store.artifact_count(case['id']) == 1
        assert len(store.own_learning_status(101, 101)['feedback']) == 1

        # The transcript intake itself is not allowed to start until an already
        # accepted feedback item has left the foreground user queue.
        accepted_feedback = store.own_learning_status(101, 101)['feedback'][0]
        store.set_feedback_result(accepted_feedback['id'], 'case_only', {
            'result': 'case_only', 'reason': 'Тестовая развязка очереди.',
            'source_document_id': 'audit', 'fragment': 'G1', 'expected_improvement': '',
        })
        store.finish_learning_job(accepted_feedback['learning_job_id'])
        await dp.feed_update(bot, Update.model_validate(dict(update_id=4, message=dict(base, message_id=4, text='/meeting'))))
        for update_id, text in enumerate(('2026-09-16', 'Новая встреча', 'Виталий — руководитель'), start=5):
            await dp.feed_update(bot, Update.model_validate(dict(update_id=update_id, message=dict(base, message_id=update_id, text=text))))
        await dp.feed_update(bot, Update.model_validate(dict(update_id=8, message=dict(base, message_id=8,
            text='Виталий: покажите источник и срок. Коллега: подготовим завтра.'))))
        meeting = store.pending_meeting(101, 101)
        assert meeting is None
        own = store.own_learning_status(101, 101)
        assert own['meetings'][0]['status'] == 'received'
        assert store.artifact_count(case['id']) == 1
        await dp.feed_update(bot, Update.model_validate(dict(update_id=22, message=dict(base, message_id=22,
            text='Это повтор транскрипта.'))))
        assert store.artifact_count(case['id']) == 1
        assert len(store.own_learning_status(101, 101)['meetings']) == 1
        group = dict(message_id=23, date=1, chat=dict(id=-100, type='group'), **{'from': user}, text='/new')
        await dp.feed_update(bot, Update.model_validate(dict(update_id=23, message=group)))
        assert any('только в личном чате' in (getattr(call, 'text', '') or '') for call in session.calls)
        with store._connect() as connection:
            assert connection.execute('SELECT id FROM cases WHERE chat_id = -100').fetchone() is None
        await bot.session.close()

    asyncio.run(scenario())


def result(decision='corrected'):
    return dict(decision=decision, replacement=dict(title='Показатель подтверждён', detail='Указано 30', material_reference='input.txt') if decision == 'corrected' else None,
                explanation='Проверено по input.txt', references=['input.txt'],
                clarifying_question='Какой слайд?' if decision == 'clarification' else '',
                suppressed_question_ids=['Q1'] if decision in {'corrected', 'removed'} else [])


@pytest.mark.parametrize('decision', ['corrected', 'removed', 'upheld', 'clarification'])
def test_point_only_and_immutable_original(setup, decision):
    store, service, case, audit = setup
    before = copy.deepcopy(store.get_case(case['id'])['review'])
    job = submit(setup)
    service.corrections.apply(job, result(decision))
    after = service.corrections.effective(case['id'])
    expected = copy.deepcopy(before)
    if decision == 'corrected':
        expected['critical_gaps'][0] = result()['replacement']
    assert after == expected
    assert store.get_case(case['id'])['review'] == before
    assert bool(service.corrections.pending(case['id'])) == (decision == 'clarification')
    count = 1 if decision in {'removed', 'corrected'} else 2
    assert len(service.corrections.effective(case['id'], rehearsal=True)['questions']) == count
    # A recovered already-applied job does not add another revision.
    service.corrections.apply(job, result(decision))
    assert len(service.corrections.history(audit['id'])) == 1


def test_duplicate_answer_is_one_job_and_no_turn_or_artifact(setup):
    store, service, case, audit = setup
    service.corrections.select(audit['id'], 0, 'G1', 101)
    def send(_):
        try:
            return service.corrections.submit(audit['id'], 101, 'Проверьте')
        except StateError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        answers = list(pool.map(send, range(2)))
    assert sum(a is not None for a in answers) == 1
    assert store.turns(case['id']) == []
    assert store.artifact_count(case['id']) == 1


def test_restart_selection_cancel_and_old_case(setup):
    store, service, case, audit = setup
    service.corrections.select(audit['id'], 0, 'C1', 101)
    restarted = Corrections(VitalyStore(store.state_dir))
    assert restarted.pending(case['id'])['pending_item'] == 'C1'
    restarted.cancel(case['id'])
    assert restarted.pending(case['id']) is None
    store.create_case(101, 'v1')
    with pytest.raises(StateError, match='Старый кейс'):
        restarted.select(audit['id'], 0, 'G1', 101)


def test_stale_revision_invalid_schema_and_ownership(setup):
    store, service, case, audit = setup
    with pytest.raises(StateError):
        service.corrections.select(audit['id'], 0, 'G1', 999)
    job = submit(setup)
    bad = result()
    bad['replacement']['unexpected'] = 'must not reach report'
    with pytest.raises(ModelError):
        service.corrections.apply(job, bad)
    assert service.corrections.get(audit['id'])['revision'] == 0
    service.corrections.apply(job, result())
    store.finish_job(job['id'])
    store.settle_job(case['id'])
    with pytest.raises(StateError, match='обновился'):
        service.corrections.select(audit['id'], 0, 'G1', 101)


def test_changed_material_and_active_grill_reject_correction(setup):
    store, service, case, audit = setup
    store._update_case(case['id'], mode='grilling')
    with pytest.raises(StateError, match='/stop'):
        service.corrections.select(audit['id'], 0, 'G1', 101)
    store._update_case(case['id'], mode='ready', review_json=None)
    with pytest.raises(StateError):
        service.corrections.select(audit['id'], 0, 'G1', 101)


def test_model_failure_keeps_original_and_pending_for_retry(setup):
    store, service, case, audit = setup
    class Runner:
        async def run(self, *args):
            raise ModelError('model_output_invalid_json')
    service.runner = Runner()
    job = submit(setup)
    with pytest.raises(ModelError):
        asyncio.run(correct_point(service, job))
    assert service.corrections.effective(case['id']) == report()
    assert service.corrections.get(audit['id'])['revision'] == 0
    assert service.corrections.pending(case['id'])


def test_source_text_and_evidence_are_passed_without_full_review_run(setup):
    store, service, case, audit = setup
    class Runner:
        async def run(self, prompt, schema, images):
            assert 'Показатель 30' in prompt
            assert 'readiness_score' not in schema['properties']
            return ModelResult(result(), '', '')
    service.runner = Runner()
    job = submit(setup)
    outcome = asyncio.run(service.execute(job))
    assert outcome.kind == 'review_correction'
    assert service.corrections.effective(case['id'])['critical_gaps'][0]['title'] == 'Показатель подтверждён'


def test_missing_source_keeps_report(setup):
    store, service, case, audit = setup
    Path(store.artifacts(case['id'])[0]['local_path']).unlink()
    with pytest.raises(StateError, match='Исходники удалены'):
        asyncio.run(correct_point(service, submit(setup)))
    assert service.corrections.effective(case['id']) == report()


def test_keyboard_pagination_and_short_callbacks(setup):
    pytest.importorskip('aiogram')
    from tof_reliability.vitaly_bot.correction_ui import keyboard
    _, _, _, audit = setup
    review = report()
    review['critical_gaps'] *= 20
    audit['current_json'] = json.dumps(review)
    first, second = keyboard(audit, 0), keyboard(audit, 1)
    assert any('Далее' in b.text for row in first.inline_keyboard for b in row)
    assert any('Назад' in b.text for row in second.inline_keyboard for b in row)
    assert all(len(b.callback_data.encode()) <= 64 for row in second.inline_keyboard for b in row)


def test_exact_slide_outside_initial_image_sample(setup, tmp_path):
    store, service, case, audit = setup
    artifact = store.artifacts(case['id'])[0]
    images = []
    for i in range(20):
        path = tmp_path / f'page-{i+1}.png'
        path.touch()
        images.append(str(path))
    with store._connect() as db:
        db.execute('UPDATE artifacts SET rendered_pages_json=? WHERE id=?', (json.dumps(images), artifact['id']))
    class Runner:
        async def run(self, prompt, schema, image_paths):
            assert image_paths == [Path(images[15])]
            return ModelResult(result(), '', '')
    service.runner = Runner()
    service.corrections.select(audit['id'], 0, 'G1', 101)
    job = service.corrections.submit(audit['id'], 101, 'Слайд 16: число уже есть')
    asyncio.run(correct_point(service, job))


def test_timeout_recovery_requeues_one_job(setup):
    store, service, case, audit = setup
    job = submit(setup)
    store.mark_job_running(job['id'])
    reopened = VitalyStore(store.state_dir)
    reopened.recover_jobs()
    assert reopened.next_queued_job()['id'] == job['id']
    assert Corrections(reopened).pending(case['id'])['pending_item'] == 'G1'


def test_point_correction_learning_example_keeps_selected_item_and_before_after(setup):
    store, service, case, audit = setup
    record = {
        'item_id': 'G1',
        'explanation': 'На слайде уже есть нужная подпись.',
        'before': {'title': 'Ложное противоречие', 'detail': 'x', 'material_reference': 'slide 2'},
        'result': {'decision': 'removed', 'explanation': 'Подпись снимает противоречие.'},
    }
    captured = store.capture_point_correction(
        case['id'], 'point-job-grounded', {'decision': 'removed', 'explanation': 'Подпись снимает противоречие.'}, record
    )
    assert captured is not None
    payload = captured['result']
    assert payload['item_id'] == 'G1'
    assert payload['before'] == record['before']
    assert payload['after'] == record['result']
    assert payload['source_document_id'] == 'audit:{}:G1'.format(case['id'])


def test_expired_case_callback_is_rejected(setup):
    from datetime import datetime, timedelta, timezone
    store, service, case, audit = setup
    store._update_case(case['id'], last_user_activity_at=(datetime.now(timezone.utc)-timedelta(days=5)).isoformat())
    fresh, expired = store.activate_case_for_user(101, 'v1', 4)
    assert expired == case['id']
    with pytest.raises(StateError):
        service.corrections.select(audit['id'], 0, 'G1', 101)


def test_release_refuses_active_job_and_does_not_signal(setup, tmp_path, monkeypatch):
    from scripts import vitaly_release
    store, _, _, _ = setup
    submit(setup)
    signals = []
    monkeypatch.setattr(vitaly_release.os, 'kill', lambda *args: signals.append(args))
    with pytest.raises(RuntimeError, match='занят'):
        vitaly_release.switch(tmp_path/'live', tmp_path/'next', store.state_dir, tmp_path/'backup')
    assert signals == []


def test_snapshot_resolves_only_configured_knowledge_path(tmp_path, monkeypatch):
    from scripts.vitaly_snapshot import resolve_knowledge_dir

    state = tmp_path / "state"
    state.mkdir()
    archive = tmp_path / "private-knowledge"
    (state / "config.env").write_text(
        "VITALY_BOT_TOKEN=must-not-be-read-or-printed\n"
        f"VITALY_BOT_KNOWLEDGE_DIR={archive}\n"
    )
    monkeypatch.delenv("VITALY_BOT_KNOWLEDGE_DIR", raising=False)
    monkeypatch.delenv("VITALY_BOT_ENV_FILE", raising=False)

    assert resolve_knowledge_dir(state) == archive
    assert resolve_knowledge_dir(state, tmp_path / "explicit") == tmp_path / "explicit"
    monkeypatch.setenv("VITALY_BOT_KNOWLEDGE_DIR", str(tmp_path / "from-env"))
    assert resolve_knowledge_dir(state) == tmp_path / "from-env"
    external_env = tmp_path / "outside-state.env"
    external_env.write_text(f"VITALY_BOT_KNOWLEDGE_DIR={tmp_path / 'from-config-path'}\n")
    monkeypatch.delenv("VITALY_BOT_KNOWLEDGE_DIR")
    monkeypatch.setenv("VITALY_BOT_ENV_FILE", str(external_env))
    assert resolve_knowledge_dir(state) == tmp_path / "from-config-path"


def test_external_tof_client_uses_its_own_interpreter(setup, tmp_path, monkeypatch):
    from tof_reliability.vitaly_bot.evidence import RemoteLedgerEvidenceProvider
    from tof_reliability.vitaly_bot.models import EvidenceError
    root = tmp_path / 'external-tof'
    monkeypatch.setenv('VITALY_BOT_TOF_ROOT', str(root))
    commands = []
    async def launch(*args, **kwargs):
        commands.append((args, kwargs))
        raise OSError('test transport unavailable')
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', launch)
    with pytest.raises(EvidenceError):
        asyncio.run(RemoteLedgerEvidenceProvider(setup[1].config).build('test', []))
    assert commands[0][0][0] == str(root / '.venv/bin/python')
    assert commands[0][0][1] == str(root / 'scripts/tof_remote_workspace.py')
    assert commands[0][1]['cwd'] == str(root)


def test_release_restores_previous_plist_on_failed_bootstrap(setup, tmp_path, monkeypatch):
    from scripts import vitaly_release
    from types import SimpleNamespace
    store, _, _, _ = setup
    live, candidate = tmp_path/'live.plist', tmp_path/'candidate.plist'
    live.write_bytes(b'previous')
    candidate.write_bytes(b'candidate')
    knowledge = tmp_path / "separate-knowledge"
    knowledge.mkdir()
    (knowledge / "meeting.txt").write_text("private transcript")
    pids = iter([123, None, None])
    monkeypatch.setattr(vitaly_release, 'pid', lambda: next(pids, None))
    monkeypatch.setattr(vitaly_release.os, 'kill', lambda *a: None)
    monkeypatch.setattr(vitaly_release.time, 'sleep', lambda *a: None)
    attempts = []
    def run(command, **kwargs):
        if command[1] == 'bootstrap':
            attempts.append(live.read_bytes())
            if len(attempts) == 1:
                raise RuntimeError('candidate launch failed')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(vitaly_release.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='candidate launch failed'):
        vitaly_release.switch(live, candidate, store.state_dir, tmp_path/'backup', knowledge)
    assert live.read_bytes() == b'previous'
    assert attempts == [b'candidate', b'previous']
    assert (tmp_path/'backup/vitaly_bot.sqlite3').exists()
    import tarfile
    with tarfile.open(tmp_path / "backup" / "knowledge.tar.gz") as archive:
        assert archive.getnames() == ["knowledge", "knowledge/meeting.txt"]
