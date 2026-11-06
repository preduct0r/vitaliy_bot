"""Red-first integration contracts for the automatic interest lifecycle.

All users, meetings, credentials and model outputs are synthetic.  The proposed
``configure_hypotheses_interest_learning`` composition seam supplies the real
Telegram runtime with isolated dependencies; the implementation may change
that seam while preserving these observable contracts.  A scheduler tick and
the existing FIFO consumer, rather than a direct lifecycle ``tick``, must drive
the work.  The comparison and package-manager spies below wrap the real s1 and
s2 implementations, never replace their decisions with a supplied verdict.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from test_hypotheses_auto_atomic_package import manager, package_lab, sha, write
from test_hypotheses_dimensions_pilot import dimension_case, topic_case
from test_hypotheses_scenario_comparison import lab as scenario_lab
from test_hypotheses_service_runtime import FakeCoverage, FakeRunner
from test_hypotheses_telegram_integration import Harness, isolated_dir
from tof_reliability.vitaly_bot import hypotheses_scenario_comparison as comparison
from tof_reliability.vitaly_bot import storage as storage_module
from tof_reliability.vitaly_bot.hypotheses_auto_atomic_package import AtomicReleaseManager
from tof_reliability.vitaly_bot.hypotheses_pilot import TrustedRequestStore
from tof_reliability.vitaly_bot.models import ModelResult


WINDOW = datetime(2030, 11, 6, 4, 0, tzinfo=timezone(timedelta(hours=3)))
PRIVATE_A = 'SYNTHETIC_USER_A_PRIVATE_834721'
PRIVATE_B = 'SYNTHETIC_USER_B_PRIVATE_912637'


class ReleaseLoader:
    def __init__(self, releases):
        self.releases = releases

    def load(self, profile_version, knowledge_package_id):
        return self.releases.release_store.load(self.releases.current()['version'])

    def load_version(self, version, profile_version, knowledge_package_id):
        return self.releases.release_store.load(version)


class SharedRunner(FakeRunner):
    """A model double producing content and independent traces, not a score."""

    def __init__(self, cards, proposal):
        super().__init__(cards)
        self.proposal = proposal
        self.order = []
        self.inject_foreground = None
        self.prompts = []

    async def run(self, prompt, schema, image_paths=None):
        properties = schema.get('properties', {})
        self.prompts.append(prompt)
        if 'decision' in properties and 'evidence' in properties:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                self.order.append('classifier')
                if self.inject_foreground:
                    callback, self.inject_foreground = self.inject_foreground, None
                    await callback()
                answer = {'decision': 'candidate', 'interest_id': self.proposal['id'],
                          'dimension_id': 'repair_repeat_interval',
                          'requirement': self.proposal['requirement'],
                          'scope': self.proposal['scope'],
                          'evidence': copy.deepcopy(self.proposal['evidence'])}
                return ModelResult(answer, json.dumps({'thread_id': 'synthetic-classifier', 'answer': answer}), '')
            finally:
                self.active -= 1
        if 'verdicts' in properties:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                self.order.append('verifier')
                rows = [{**entry, 'interest_id': self.proposal['id'],
                         'dimension_id': 'repair_repeat_interval', 'supported': True,
                         'confidence': 'high', 'conflicting': False}
                        for entry in self.proposal['evidence']]
                answer = {'verdicts': rows}
                return ModelResult(answer, json.dumps({'thread_id': 'synthetic-verifier', 'answer': answer}), '')
            finally:
                self.active -= 1
        self.order.append('foreground')
        return await super().run(prompt, schema, image_paths)


def add_accepted_meetings(store, inventory):
    """Exercise the actual accepted-meeting store adapter with original bytes."""
    rows = []
    for number, source in enumerate(inventory['meetings'], 1):
        meeting = store.begin_meeting(400 + number, 400 + number)
        if source.get('date'):
            store.request_meeting_date_edit(meeting['id'], 400 + number, 400 + number)
            store.submit_meeting_metadata(meeting['id'], source['date'])
        path = Path(source['path'])
        accepted, job = store.accept_meeting_source(
            meeting['id'], source_name=path.name, source_path=path, normalized_path=path,
            source_sha256=sha(path), normalized_sha256=sha(path))
        assert job is not None
        store.set_meeting_result(accepted['id'], {'status': 'accepted'})
        store.finish_learning_job(job['id'])
        rows.append({**source, 'id': accepted['id'], 'meeting_id': accepted['id']})
    return rows


async def wait_for(predicate, message, timeout=8):
    async def poll():
        while not predicate():
            await asyncio.sleep(.01)
    try:
        await asyncio.wait_for(poll(), timeout=timeout)
    except asyncio.TimeoutError:
        pytest.fail(message)


async def setup(h, lab, monkeypatch, *, mode='improved'):
    """Configure only inputs and doubles; no candidate or assessment is injected."""
    runtime = h.runtime
    configure = getattr(runtime, 'configure_hypotheses_interest_learning', None)
    assert callable(configure), '04:00 scheduler and FIFO do not connect hypotheses learning'
    lab['scenario']['mode']['value'] = mode
    inventory = copy.deepcopy(lab['inventory'])
    inventory['meetings'] = add_accepted_meetings(runtime.store, inventory)
    proposal = copy.deepcopy(lab['proposal'])
    for evidence, meeting in zip(proposal['evidence'], inventory['meetings']):
        evidence['source_id'] = meeting['id']
    write(lab['root'] / 'inventory.json', inventory)
    (lab['root'] / 'inventory.json').chmod(0o600)
    package = manager(lab['store'], lab['anchor'])
    # Use a real pilot request captured by the package fixture. The scenario
    # generator requires dimensions; the earlier abbreviated setup never
    # reached the lifecycle under test once its composition seam existed.
    initial_request = copy.deepcopy(lab['scenario']['calls'][0])
    initial_request['scenario'] = {'kind': 'repair_episodes'}
    runner = SharedRunner(lab['scenario']['generator'](initial_request), proposal)
    runtime.service.runner = runner
    h.hypotheses.runner = runner
    h.hypotheses.knowledge_loader = ReleaseLoader(package)
    h.hypotheses.coverage_provider = FakeCoverage(lab['scenario']['coverage'])
    h.hypotheses.trusted_store = TrustedRequestStore(lab['root'] / 'bot-trusted.sqlite')

    # Add a genuine pending feedback event, including an idempotent replay.
    _, _, run = await h.completed(owner=101)
    h.hypotheses.begin_feedback_input(run_id=run['id'], user_id=101, card_number=1,
                                     expected_revision=run['revision'])
    feedback = h.hypotheses.submit_feedback_text(user_id=101, text=PRIVATE_A,
                                                 event_id='synthetic-feedback-event')
    assert h.hypotheses.submit_feedback_text(user_id=101, text=PRIVATE_A,
                                              event_id='synthetic-feedback-event') == feedback
    await h.message(PRIVATE_B, chat=202)
    monkeypatch.setattr(storage_module, '_local_now', lambda now=None: now or WINDOW)

    evaluations, promotions = [], []
    real_evaluate = comparison.evaluate_scenarios
    real_promote = AtomicReleaseManager.promote

    def observe_evaluation(**kwargs):
        result = real_evaluate(**kwargs)
        evaluations.append(copy.deepcopy(result))
        return result

    def observe_promotion(self, **kwargs):
        promotions.append(copy.deepcopy(kwargs))
        return real_promote(self, **kwargs)

    monkeypatch.setattr(comparison, 'evaluate_scenarios', observe_evaluation)
    monkeypatch.setattr(AtomicReleaseManager, 'promote', observe_promotion)
    engine = configure(release_manager=package, inventory_path=lab['root'] / 'inventory.json',
        scenario_coverage_provider=lab['scenario']['coverage_provider'],
        scenario_generator=lab['scenario']['generator'],
        scenario_verifier=lab['scenario']['verifier'], authority=lab['authority'],
        audit_dir=lab['scenario']['audit_dir'])
    return engine, runner, evaluations, promotions, inventory


async def drive(h, engine):
    await h.runtime.start()
    await wait_for(lambda: engine.status().get('status') in
                   {'published', 'rejected', 'no_change', 'failed'},
                   'scheduled interest cycle did not reach a durable outcome', timeout=60)


def public_files(root):
    result = b''
    for path in Path(root).rglob('*'):
        if path.is_file() and path.suffix in {'.json', '.txt', '.md'}:
            result += path.read_bytes()
    return result


@pytest.mark.parametrize('mode,published', [('improved', True), ('tie', False), ('regression', False)])
def test_scheduler_runs_real_comparison_and_verified_release(isolated_dir, package_lab, monkeypatch, mode, published):
    async def scenario():
        h = Harness(isolated_dir)
        lab = package_lab
        try:
            engine, runner, evaluations, promotions, inventory = await setup(h, lab, monkeypatch, mode=mode)
            baseline = lab['store'].current()
            await drive(h, engine)
            assert evaluations, 'the real s1 evaluator must execute baseline and candidate pilots'
            assessment = evaluations[-1]
            assert assessment['activation_allowed'] is published
            assert assessment['scenarios']
            assert all(row['baseline']['output'] and row['candidate']['output']
                       and row['baseline']['observations'] and row['candidate']['observations']
                       for row in assessment['scenarios'])
            assert lab['scenario']['calls'] and lab['scenario']['pilots']
            if published:
                assert promotions, 'promotion must go through the verified s2 materializer'
                assert lab['store'].current()['version'] != baseline['version']
                assert any(row['comparison']['improved'] for row in assessment['scenarios'])
                active = lab['store'].current()
                manifest, values = lab['store']._validate(
                    lab['store'].root / 'releases' / active['version'], active['manifest_sha256'])
                assert set(manifest['source_ids']) == {x['id'] for x in inventory['historical_sources'] + inventory['meetings']}
                assert values['classifier_trace']['thread_id'] != values['verifier_trace']['thread_id']
                assert PRIVATE_A.encode() not in public_files(lab['store'].root / 'releases' / active['version'])
                assert PRIVATE_B.encode() not in public_files(lab['store'].root / 'releases' / active['version'])
            else:
                assert lab['store'].current() == baseline
            assert runner.max_active <= 1
            assert PRIVATE_A not in h.text(0) and PRIVATE_B not in h.text(0)
            assert all(PRIVATE_A not in prompt and PRIVATE_B not in prompt for prompt in runner.prompts)
        finally:
            await h.runtime.stop()
            await h.close()
    asyncio.run(scenario())


def test_foreground_preempts_between_stages_and_pins_survive_rollback(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            lab = package_lab
            engine, runner, evaluations, promotions, _ = await setup(h, lab, monkeypatch)
            old_version = lab['store'].current()['version']
            accepted = {}

            async def enqueue_user():
                await h.message('/hypotheses насосы', chat=303)
                case = h.runtime.service.active_case(303)
                accepted['job'] = h.runtime.store.get_job(case['active_job_id'])
                accepted['run'] = h.hypotheses.get_run(user_id=303, job_id=accepted['job']['id'])

            runner.inject_foreground = enqueue_user
            await drive(h, engine)
            assert runner.order.index('classifier') < runner.order.index('foreground') < runner.order.index('verifier')
            assert runner.max_active == 1
            assert accepted['run']['knowledge_version'] == old_version
            assert h.runtime.store.get_job(accepted['job']['id'])['status'] == 'completed'
            active = lab['store'].current()
            assert active['version'] != old_version
            await h.message('/hypotheses насосы', chat=404)
            case = h.runtime.service.active_case(404)
            new_job = h.runtime.store.get_job(case['active_job_id'])
            new_run = h.hypotheses.get_run(user_id=404, job_id=new_job['id'])
            assert new_run['knowledge_version'] == active['version']
            await wait_for(lambda: h.runtime.store.get_job(new_job['id'])['status'] not in ('queued', 'running'),
                           'new hypotheses job did not finish')
            assert h.runtime.store.get_job(new_job['id'])['status'] == 'completed'
            old = h.hypotheses.get_run(user_id=303, job_id=accepted['job']['id'])
            new = h.hypotheses.get_run(user_id=404, job_id=new_job['id'])
            assert old['knowledge_version'] == old_version and new['knowledge_version'] == active['version']
            assert any(row['baseline']['output'] != row['candidate']['output'] for row in evaluations[-1]['scenarios'])
            manager(lab['store'], lab['anchor']).rollback(old_version)
            assert lab['store'].current()['version'] == old_version
            assert h.hypotheses.get_run(user_id=404, job_id=new_job['id'])['knowledge_version'] == active['version']
            assert runner.max_active == 1
        finally:
            await h.runtime.stop()
            await h.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('defect', ['deleted_source', 'server_outage', 'ambiguous', 'disk_error'])
def test_unavailable_evidence_or_interruption_never_replaces_current(isolated_dir, package_lab, monkeypatch, defect):
    async def scenario():
        h = Harness(isolated_dir)
        lab = package_lab
        try:
            engine, runner, _, _, inventory = await setup(h, lab, monkeypatch)
            baseline = lab['store'].current()
            if defect == 'deleted_source':
                Path(inventory['meetings'][0]['path']).unlink()
            elif defect == 'server_outage':
                lab['scenario']['mode']['outage'] = True
            elif defect == 'ambiguous':
                runner.proposal['evidence'][0]['attribution'] = 'ambiguous'
            else:
                replace = os.replace
                def fail_pointer(src, dst, *args, **kwargs):
                    if Path(dst).name == 'current.json':
                        raise OSError('synthetic disk full')
                    return replace(src, dst, *args, **kwargs)
                monkeypatch.setattr(os, 'replace', fail_pointer)
            await drive(h, engine)
            assert lab['store'].current() == baseline
            assert engine.status().get('published') is not True
            await h.runtime.stop()
            reopened = manager(type(lab['store'])(lab['store'].root), lab['anchor'])
            assert reopened.current() == baseline
        finally:
            await h.close()
    asyncio.run(scenario())


def test_cancel_and_duplicate_feedback_remain_user_scoped(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, _, _, _, _ = await setup(h, package_lab, monkeypatch)
            _, _, run = await h.completed(owner=202)
            h.hypotheses.begin_feedback_input(run_id=run['id'], user_id=202, card_number=1,
                                             expected_revision=run['revision'])
            await h.message('/cancel', chat=202)
            assert h.runtime.store.pop_hypotheses_feedback_input(202) is None
            pending = h.runtime.store.pending_hypotheses_learning(limit=100)
            assert sum(row['event_id'] == 'synthetic-feedback-event' for row in pending) == 1
            assert not any(row['user_id'] == 202 for row in pending)
            assert package_lab['store'].current() == package_lab['baseline']
        finally:
            await h.close()
    asyncio.run(scenario())


@pytest.mark.parametrize('mismatch', [False, True])
def test_materializer_binds_original_classifier_trace(package_lab, mismatch):
    """The generated proposal's original trace must survive normalization."""
    lab = package_lab
    raw = {'decision': 'candidate', 'interest_id': lab['proposal']['id'],
           'dimension_id': 'repair_repeat_interval', 'requirement': lab['proposal']['requirement'],
           'scope': lab['proposal']['scope'], 'evidence': copy.deepcopy(lab['proposal']['evidence'])}
    if mismatch:
        raw['requirement'] = 'A different synthetic requirement'
    trace = {'thread_id': 'synthetic-classifier', 'answer': raw}
    args = {name: copy.deepcopy(lab[name]) for name in
            ('inventory', 'candidate', 'proposal', 'classifier', 'verifier', 'verifier_trace')}
    args.update(classifier_trace=trace, assessment_path=lab['assessment'])
    if mismatch:
        with pytest.raises(ValueError):
            manager(lab['store'], lab['anchor']).promote(**args)
        assert lab['store'].current() == lab['baseline']
    else:
        result = manager(lab['store'], lab['anchor']).promote(**args)
        assert result['published'] is True
        active = lab['store'].current()
        _, values = lab['store']._validate(lab['store'].root / 'releases' / active['version'],
                                            active['manifest_sha256'])
        assert values['classifier_trace']['new_trace'] == trace


def test_comparison_is_resumable_at_each_model_step(package_lab):
    lab = package_lab
    calls, checks = lab['scenario']['calls'], lab['scenario']['checks']
    checkpoint = lab['root'] / 'comparison-checkpoint.json'
    args = dict(checkpoint=checkpoint, authority=lab['authority'],
                baseline=lab['baseline_payload'], candidate=lab['candidate'], scope='насосы',
                coverage_provider=lab['scenario']['coverage_provider'],
                generator=lab['scenario']['generator'], private_verifier=lab['scenario']['verifier'],
                audit_dir=lab['scenario']['audit_dir'])
    for number in range(500):
        # Reconstruct on every step: no in-memory generator or coroutine can
        # be required to resume a partially completed comparison.
        before = len(calls) + len(checks)
        result = comparison.ScenarioComparisonStepper(**args).step()
        assert len(calls) + len(checks) - before <= 1
        if result['status'] == 'complete':
            break
        assert result['status'] == 'pending'
    else:
        pytest.fail('comparison did not complete')
    assert result['assessment']['activation_allowed'] is True
    assert any(row['comparison']['improved'] for row in result['assessment']['scenarios'])
    assert len(calls) and len(checks)


def test_comparison_checkpoint_tampering_fails_closed(package_lab):
    lab = package_lab
    checkpoint = lab['root'] / 'comparison-checkpoint.json'
    args = dict(checkpoint=checkpoint, authority=lab['authority'],
                baseline=lab['baseline_payload'], candidate=lab['candidate'], scope='насосы',
                coverage_provider=lab['scenario']['coverage_provider'],
                generator=lab['scenario']['generator'], private_verifier=lab['scenario']['verifier'],
                audit_dir=lab['scenario']['audit_dir'])
    assert comparison.ScenarioComparisonStepper(**args).step()['status'] == 'pending'
    saved = json.loads(checkpoint.read_text(encoding='utf-8'))
    saved['responses'][0]['output'] = []
    write(checkpoint, saved)
    before = len(lab['scenario']['calls']) + len(lab['scenario']['checks'])
    with pytest.raises(ValueError):
        comparison.ScenarioComparisonStepper(**args).step()
    assert len(lab['scenario']['calls']) + len(lab['scenario']['checks']) == before
    assert lab['store'].current() == lab['baseline']


def test_comparison_checks_server_generation_between_steps(package_lab):
    lab = package_lab
    args = dict(checkpoint=lab['root'] / 'comparison-checkpoint.json', authority=lab['authority'],
                baseline=lab['baseline_payload'], candidate=lab['candidate'], scope='насосы',
                coverage_provider=lab['scenario']['coverage_provider'],
                generator=lab['scenario']['generator'], private_verifier=lab['scenario']['verifier'],
                audit_dir=lab['scenario']['audit_dir'])
    assert comparison.ScenarioComparisonStepper(**args).step()['status'] == 'pending'
    before = len(lab['scenario']['calls']) + len(lab['scenario']['checks'])
    lab['scenario']['mode']['drift'] = True
    with pytest.raises((ValueError, RuntimeError)):
        comparison.ScenarioComparisonStepper(**args).step()
    assert len(lab['scenario']['calls']) + len(lab['scenario']['checks']) == before


def test_replay_reads_server_coverage_once_per_step(package_lab):
    lab = package_lab
    calls = []
    def coverage():
        calls.append(1)
        return lab['scenario']['coverage_provider']()
    args = dict(checkpoint=lab['root'] / 'comparison-checkpoint.json', authority=lab['authority'],
                baseline=lab['baseline_payload'], candidate=lab['candidate'], scope='насосы',
                coverage_provider=coverage, generator=lab['scenario']['generator'],
                private_verifier=lab['scenario']['verifier'], audit_dir=lab['scenario']['audit_dir'])
    for _ in range(3):
        calls.clear()
        assert comparison.ScenarioComparisonStepper(**args).step()['status'] == 'pending'
        assert len(calls) == 1


def test_restart_recovers_completed_pointer_without_promoting_twice(isolated_dir, package_lab, monkeypatch):
    from tof_reliability.vitaly_bot.hypotheses_learning_cycle import AutomaticInterestCycle
    lab = package_lab
    h = Harness(isolated_dir)
    package = manager(lab['store'], lab['anchor'])
    engine = AutomaticInterestCycle(store=h.runtime.store, runner=h.runtime.service.runner,
        release_manager=package, inventory_path=lab['root'] / 'inventory.json',
        scenario_coverage_provider=lab['scenario']['coverage_provider'],
        scenario_generator=lab['scenario']['generator'], scenario_verifier=lab['scenario']['verifier'],
        authority=lab['authority'], audit_dir=lab['scenario']['audit_dir'], run_hour=4)
    try:
        monkeypatch.setattr(storage_module, '_local_now', lambda now=None: now or WINDOW)
        state = {'day': WINDOW.date().isoformat(), 'status': 'pending', 'stage': 'promote',
                 'published': False, 'baseline': lab['baseline'], 'candidate': lab['candidate']}
        engine._save(state)
        args = {name: copy.deepcopy(lab[name]) for name in
                ('inventory', 'candidate', 'proposal', 'classifier', 'verifier',
                 'classifier_trace', 'verifier_trace')}
        package.promote(**args, assessment_path=lab['assessment'])
        asyncio.run(engine.step())
        assert engine.status()['status'] == 'published'
        assert package.current()['version'] == lab['candidate']['knowledge_version']
    finally:
        asyncio.run(h.close())


def test_production_loader_accepts_only_authenticated_promoted_versions(isolated_dir, package_lab):
    from dataclasses import replace
    from tof_reliability.vitaly_bot.telegram import TelegramRuntime

    h = Harness(isolated_dir)
    lab = package_lab
    try:
        package = manager(lab['store'], lab['anchor'])
        args = {name: copy.deepcopy(lab[name]) for name in
                ('inventory', 'candidate', 'proposal', 'classifier', 'verifier',
                 'classifier_trace', 'verifier_trace')}
        package.promote(**args, assessment_path=lab['assessment'])
        config = replace(h.runtime.config, hypotheses_production={
            'directory': str(lab['store'].root), 'manifest_sha256': lab['anchor'],
            'authority_path': str(lab['root'] / 'authority.key'),
            'authority_fingerprint': sha(lab['root'] / 'authority.key'),
            'state_path': str(lab['root'] / 'trusted.sqlite'),
            'inventory_path': str(lab['root'] / 'inventory.json'),
            'audit_dir': str(lab['scenario']['audit_dir']),
        })
        runtime = TelegramRuntime(h.bot, config)
        assert runtime.worker.interest_cycle is not None
        assert runtime.worker.interest_cycle.auto_inventory is True
        service = runtime.service.hypotheses_service
        active = runtime.store.active_knowledge_package()
        version = active['package']['profile']['version']
        assert service.knowledge_loader.load(version, active['id'])['knowledge_version'] == lab['candidate']['knowledge_version']
        assert service.knowledge_loader.load_version(lab['baseline']['version'], version, active['id'])['knowledge_version'] == lab['baseline']['version']
        with pytest.raises(ValueError):
            service.knowledge_loader.load_version('untrusted', version, active['id'])
    finally:
        asyncio.run(h.close())


def test_automatic_release_configuration_is_complete_or_rejected(package_lab):
    from tof_reliability.vitaly_bot.config import load_config
    from tof_reliability.vitaly_bot.models import InputError

    lab = package_lab
    env = {'VITALY_BOT_ENV_FILE': str(lab['root'] / 'missing.env'),
           'VITALY_BOT_HYPOTHESES_PRODUCTION': '1',
           'VITALY_BOT_HYPOTHESES_RELEASE_DIR': str(lab['store'].root),
           'VITALY_BOT_HYPOTHESES_RELEASE_SHA256': lab['anchor'],
           'VITALY_BOT_HYPOTHESES_AUTHORITY_PATH': str(lab['root'] / 'authority.key'),
           'VITALY_BOT_HYPOTHESES_AUTHORITY_SHA256': sha(lab['root'] / 'authority.key'),
           'VITALY_BOT_HYPOTHESES_STATE_PATH': str(lab['root'] / 'trusted.sqlite'),
           'VITALY_BOT_HYPOTHESES_INVENTORY_PATH': str(lab['root'] / 'inventory.json'),
           'VITALY_BOT_HYPOTHESES_AUDIT_DIR': str(lab['scenario']['audit_dir'])}
    config = load_config(env, require_token=False)
    assert config.hypotheses_production['authority_fingerprint'] == env['VITALY_BOT_HYPOTHESES_AUTHORITY_SHA256']
    assert config.hypotheses_production['inventory_path'] == env['VITALY_BOT_HYPOTHESES_INVENTORY_PATH']
    del env['VITALY_BOT_HYPOTHESES_STATE_PATH']
    with pytest.raises(InputError):
        load_config(env, require_token=False)


def test_job_acceptance_serializes_with_release_promotion(isolated_dir, package_lab, monkeypatch):
    import threading
    from dataclasses import replace
    from tof_reliability.vitaly_bot.telegram import TelegramRuntime

    h, lab = Harness(isolated_dir), package_lab
    package = manager(lab['store'], lab['anchor'])
    config = replace(h.runtime.config, hypotheses_production={
        'directory': str(lab['store'].root), 'manifest_sha256': lab['anchor'],
        'authority_path': str(lab['root'] / 'authority.key'),
        'authority_fingerprint': sha(lab['root'] / 'authority.key'),
        'state_path': str(lab['root'] / 'trusted.sqlite'),
        'inventory_path': str(lab['root'] / 'inventory.json'),
        'audit_dir': str(lab['scenario']['audit_dir'])})
    runtime = TelegramRuntime(h.bot, config)
    service = runtime.service.hypotheses_service
    case = runtime.service.new_case(101)
    started, finished, errors = threading.Event(), threading.Event(), []
    args = {name: copy.deepcopy(lab[name]) for name in
            ('inventory', 'candidate', 'proposal', 'classifier', 'verifier',
             'classifier_trace', 'verifier_trace')}
    def promote():
        started.set()
        try:
            package.promote(**args, assessment_path=lab['assessment'])
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()
    thread = threading.Thread(target=promote)
    original = runtime.store.begin_hypotheses_run
    def accept(*args, **kwargs):
        thread.start()
        assert started.wait(2)
        assert not finished.wait(.25), 'release changed before job acceptance committed'
        return original(*args, **kwargs)
    monkeypatch.setattr(runtime.store, 'begin_hypotheses_run', accept)
    try:
        job = service.submit(case_id=case['id'], chat_id=101, user_id=101, description='насосы')
        thread.join(10)
        assert finished.is_set() and not errors
        assert service.get_run(user_id=101, job_id=job['id'])['knowledge_version'] == lab['baseline']['version']
        assert package.current()['version'] == lab['candidate']['knowledge_version']
    finally:
        if thread.ident:
            thread.join(10)
        asyncio.run(h.close())


def test_interest_cycle_never_admits_while_user_work_is_queued_or_running(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, _, _, _, _ = await setup(h, package_lab, monkeypatch)
            case = h.runtime.service.new_case(303)
            job = h.hypotheses.submit(case_id=case['id'], chat_id=303, user_id=303)
            assert not engine.ready()
            h.runtime.store.mark_job_running(job['id'])
            assert not engine.ready()
            h.runtime.store.finish_job(job['id'])
            assert engine.ready()
        finally:
            await h.close()
    asyncio.run(scenario())


def test_cycle_rejects_inventory_omitting_an_accepted_meeting(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, runner, _, _, _ = await setup(h, package_lab, monkeypatch)
            path = package_lab['root'] / 'third-meeting.txt'
            path.write_text('A third synthetic meeting with distinct evidence.', encoding='utf-8')
            add_accepted_meetings(h.runtime.store, {'meetings': [{'path': str(path)}]})
            await engine.step()
            assert engine.status()['status'] in {'failed', 'rejected'}
            assert 'classifier' not in runner.order
            assert package_lab['store'].current() == package_lab['baseline']
        finally:
            await h.close()
    asyncio.run(scenario())


def test_cycle_rejects_a_meeting_with_changed_date(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, runner, _, _, inventory = await setup(h, package_lab, monkeypatch)
            inventory['meetings'][0]['date'] = '2029-01-01'
            write(package_lab['root'] / 'inventory.json', inventory)
            await engine.step()
            assert engine.status()['status'] in {'failed', 'rejected'}
            assert 'classifier' not in runner.order
            assert package_lab['store'].current() == package_lab['baseline']
        finally:
            await h.close()
    asyncio.run(scenario())


def test_damaged_learning_checkpoint_does_not_kill_user_worker(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, _, _, _, _ = await setup(h, package_lab, monkeypatch)
            engine.path.parent.mkdir(parents=True, exist_ok=True)
            engine.path.write_text('{}', encoding='utf-8')
            await h.runtime.start()
            await asyncio.sleep(.1)
            await h.message('/hypotheses насосы', chat=303)
            case = h.runtime.service.active_case(303)
            job = h.runtime.store.get_job(case['active_job_id'])
            await wait_for(lambda: h.runtime.store.get_job(job['id'])['status'] == 'completed',
                           'damaged background checkpoint stopped the user worker')
            assert not h.runtime.worker._task.done()
            assert package_lab['store'].current() == package_lab['baseline']
        finally:
            await h.runtime.stop()
            await h.close()
    asyncio.run(scenario())


def test_new_daily_cycle_does_not_reuse_previous_model_responses(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, _, _, _, _ = await setup(h, package_lab, monkeypatch)
            await engine.step()  # classifier
            await engine.step()  # verifier
            await engine.step()  # first scenario generator
            assert engine.status()['stage'] == 'evaluate'
            previous_calls = len(package_lab['scenario']['calls'])
            state = engine.status()
            state.update(status='rejected', stage='complete')
            engine._save(state)
            monkeypatch.setattr(storage_module, '_local_now', lambda now=None: now or WINDOW + timedelta(days=1))
            await engine.step()
            await engine.step()
            await engine.step()
            assert engine.status()['stage'] == 'evaluate'
            assert len(package_lab['scenario']['calls']) == previous_calls + 1
        finally:
            await h.close()
    asyncio.run(scenario())


def test_terminal_daily_cycle_is_not_restarted_same_day(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, runner, _, _, _ = await setup(h, package_lab, monkeypatch)
            engine._save({'day': WINDOW.date().isoformat(), 'status': 'rejected',
                          'stage': 'complete', 'published': False})
            await engine.step()
            assert engine.status()['status'] == 'rejected'
            assert not runner.order
        finally:
            await h.close()
    asyncio.run(scenario())


def test_resumed_cycle_uses_its_daily_window_without_starting_another(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, runner, _, _, _ = await setup(h, package_lab, monkeypatch)
            answer = {'decision': 'none'}
            async def no_change(prompt, schema, image_paths=None):
                return ModelResult(answer, json.dumps({'thread_id': 'synthetic-no-change', 'answer': answer}), '')
            runner.run = no_change
            engine._save({'day': (WINDOW - timedelta(days=1)).date().isoformat(),
                          'status': 'pending', 'stage': 'classify', 'published': False})
            await engine.step()
            assert engine.status()['status'] == 'no_change'
            assert not engine.ready()
            await engine.step()
            assert engine.status()['status'] == 'no_change'
        finally:
            await h.close()
    asyncio.run(scenario())


def test_transient_server_outage_resumes_saved_stage_next_day(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, _, _, _, _ = await setup(h, package_lab, monkeypatch)
            await engine.step()
            await engine.step()
            package_lab['scenario']['mode']['outage'] = True
            await engine.step()
            assert engine.status()['status'] == 'failed'
            assert engine.status()['stage'] == 'evaluate'
            assert package_lab['store'].current() == package_lab['baseline']
            package_lab['scenario']['mode']['outage'] = False
            monkeypatch.setattr(storage_module, '_local_now', lambda now=None: now or WINDOW + timedelta(days=1))
            await engine.step()
            assert engine.status()['status'] == 'pending'
            assert engine.status()['stage'] == 'evaluate'
        finally:
            await h.close()
    asyncio.run(scenario())


def test_worker_does_not_recover_jobs_when_another_worker_holds_lock(isolated_dir):
    import fcntl
    h = Harness(isolated_dir)
    lock = h.runtime.store.state_dir / 'model-worker.lock'
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(Exception):
            asyncio.run(h.runtime.worker.start())
        assert h.runtime.worker._task is None
    finally:
        os.close(fd)
        asyncio.run(h.close())


def test_user_queued_during_source_check_preempts_next_model_call(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, runner, _, _, _ = await setup(h, package_lab, monkeypatch)
            original = engine._inputs
            def inputs_with_new_user():
                data = original()
                case = h.runtime.service.new_case(303)
                h.hypotheses.submit(case_id=case['id'], chat_id=303, user_id=303)
                return data
            monkeypatch.setattr(engine, '_inputs', inputs_with_new_user)
            await engine.step()
            assert not runner.order
            assert engine.status()['status'] == 'pending'
            assert engine.status()['stage'] == 'classify'
        finally:
            await h.close()
    asyncio.run(scenario())


def test_user_queued_during_coverage_preempts_scenario_model(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, _, _, _, _ = await setup(h, package_lab, monkeypatch)
            await engine.step()
            await engine.step()
            before = len(package_lab['scenario']['calls']) + len(package_lab['scenario']['checks'])
            original = engine.coverage
            def coverage_with_new_user():
                data = original()
                case = h.runtime.service.new_case(303)
                h.hypotheses.submit(case_id=case['id'], chat_id=303, user_id=303)
                return data
            engine.coverage = coverage_with_new_user
            await engine.step()
            assert len(package_lab['scenario']['calls']) + len(package_lab['scenario']['checks']) == before
            assert engine.status()['status'] == 'pending'
            assert engine.status()['stage'] == 'evaluate'
        finally:
            await h.close()
    asyncio.run(scenario())


def test_inventory_adapter_adds_accepted_meetings_and_preserves_originals(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, _, _, _, inventory = await setup(h, package_lab, monkeypatch)
            path = package_lab['root'] / 'third-meeting.txt'
            path.write_text('A third synthetic accepted meeting.', encoding='utf-8')
            source = {'path': str(path), 'date': '2030-11-03'}
            row = add_accepted_meetings(h.runtime.store, {'meetings': [source]})[0]
            engine.refresh_inventory()
            result = json.loads((package_lab['root'] / 'inventory.json').read_text(encoding='utf-8'))
            assert result['historical_sources'] == inventory['historical_sources']
            assert {r['meeting_id'] for r in result['meetings']} == {r['id'] for r in inventory['meetings']} | {row['id']}
            added = next(r for r in result['meetings'] if r['id'] == row['id'])
            assert added['date'] == source['date'] and added['sha256'] == sha(path)
            assert Path(added['original_path']).read_bytes() == path.read_bytes()
            before = (package_lab['root'] / 'inventory.json').read_bytes()
            path.unlink()
            with pytest.raises((OSError, ValueError)):
                engine.refresh_inventory()
            assert (package_lab['root'] / 'inventory.json').read_bytes() == before
        finally:
            await h.close()
    asyncio.run(scenario())


def test_new_verified_meetings_update_existing_attribution_counts(package_lab):
    from test_hypotheses_auto_atomic_package import release_package
    from tof_reliability.vitaly_bot.hypotheses_knowledge import VerifiedReleaseStore
    from tof_reliability.vitaly_bot.hypotheses_learning_cycle import AutomaticInterestCycle

    lab = package_lab
    baseline = copy.deepcopy(lab['baseline_payload'])
    candidate = copy.deepcopy(lab['candidate'])
    for payload in (baseline, candidate):
        for dimension in payload['dimension_projection']['dimensions']:
            counts = {}
            for evidence in dimension['evidence']:
                counts[evidence['attribution']] = counts.get(evidence['attribution'], 0) + 1
            dimension['attribution_meeting_counts'] = counts
    raw = {'decision': 'candidate', 'interest_id': lab['proposal']['id'],
           'dimension_id': 'repair_repeat_interval', 'requirement': lab['proposal']['requirement'],
           'scope': lab['proposal']['scope'], 'evidence': copy.deepcopy(lab['proposal']['evidence'])}
    _, generated = AutomaticInterestCycle._candidate(None, baseline, lab['inventory'], raw)
    actual = next(row for row in generated['dimension_projection']['dimensions']
                  if row['dimension_id'] == raw['dimension_id'])
    assert actual['attribution_meeting_counts'] == {'interpretation': 1, 'direct_request': 2}
    source = release_package(lab['root'] / 'counted-baseline', baseline['interest_context'],
        baseline['dimension_projection'], lab['inventory']['historical_sources'], baseline['knowledge_version'])
    store = VerifiedReleaseStore(lab['root'] / 'counted-store')
    anchor = sha(source / 'manifest.json')
    store.install(source, manifest_sha256=anchor)
    release = AtomicReleaseManager(release_store=store, trusted_manifest_sha256=anchor,
        authority_path=lab['root'] / 'authority.key', authority_fingerprint=sha(lab['root'] / 'authority.key'),
        state_path=lab['root'] / 'counted-state.sqlite', inventory_path=lab['root'] / 'inventory.json')
    assessment = comparison.evaluate_scenarios(baseline=baseline, candidate=candidate, scope='насосы',
        coverage_provider=lab['scenario']['coverage_provider'], generator=lab['scenario']['generator'],
        private_verifier=lab['scenario']['verifier'], scenarios=comparison.fixed_scenarios(),
        audit_dir=lab['scenario']['audit_dir'], authority=lab['authority'])
    assert assessment['activation_allowed'] is True
    args = {name: copy.deepcopy(lab[name]) for name in
            ('inventory', 'proposal', 'classifier', 'verifier', 'classifier_trace', 'verifier_trace')}
    assert release.promote(**args, candidate=candidate, assessment_path=assessment['audit_path'])['published']


def test_automatic_cycle_snapshots_newly_accepted_meeting(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, runner, _, _, _ = await setup(h, package_lab, monkeypatch)
            engine.auto_inventory = True
            path = package_lab['root'] / 'new-accepted.txt'
            path.write_text('A new synthetic meeting accepted by the store.', encoding='utf-8')
            added = add_accepted_meetings(h.runtime.store, {'meetings': [
                {'path': str(path), 'date': '2030-11-03'}]})[0]
            await engine.step()
            assert engine.status()['stage'] == 'verify'
            assert added['id'] in runner.prompts[-1]
            assert PRIVATE_A not in runner.prompts[-1] and PRIVATE_B not in runner.prompts[-1]
        finally:
            await h.close()
    asyncio.run(scenario())


def test_checkpoint_failure_does_not_lose_user_queue_notification(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        job = {}
        try:
            await setup(h, package_lab, monkeypatch)
            class FailingCycle:
                used = False
                def ready(self):
                    return not self.used
                async def step(self):
                    self.used = True
                    case = h.runtime.service.new_case(303)
                    job.update(h.hypotheses.submit(case_id=case['id'], chat_id=303, user_id=303))
                    h.runtime.worker.notify()
                    raise OSError('synthetic checkpoint write failure')
            h.runtime.worker.interest_cycle = FailingCycle()
            await h.runtime.worker.start()
            await wait_for(lambda: job and h.runtime.store.get_job(job['id'])['status'] == 'completed',
                           'queued user work was not awakened after checkpoint failure')
        finally:
            await h.runtime.stop()
            await h.close()
    asyncio.run(scenario())


def test_promotion_waits_when_coverage_read_crosses_daily_window(isolated_dir, package_lab, monkeypatch):
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, _, _, _, _ = await setup(h, package_lab, monkeypatch)
            await engine.step()
            await engine.step()
            state = engine.status()
            baseline = engine._inputs()[2]
            assessment = comparison.evaluate_scenarios(baseline=baseline, candidate=state['candidate'], scope='насосы',
                coverage_provider=package_lab['scenario']['coverage_provider'], generator=package_lab['scenario']['generator'],
                private_verifier=package_lab['scenario']['verifier'], scenarios=comparison.fixed_scenarios(),
                audit_dir=package_lab['scenario']['audit_dir'], authority=package_lab['authority'])
            assert assessment['activation_allowed'] is True
            state.update(stage='promote', assessment_path=assessment['audit_path'])
            engine._save(state)
            clock = [WINDOW]
            monkeypatch.setattr(storage_module, '_local_now', lambda now=None: now or clock[0])
            def coverage_after_window():
                result = package_lab['scenario']['coverage_provider']()
                clock[0] = WINDOW + timedelta(hours=1)
                return result
            engine.coverage = coverage_after_window
            await engine.step()
            assert engine.status()['status'] == 'pending'
            assert engine.status()['stage'] == 'promote'
            assert package_lab['store'].current() == package_lab['baseline']
        finally:
            await h.close()
    asyncio.run(scenario())


def test_candidate_updates_explicit_profile_and_corpus_version_pins(package_lab):
    from tof_reliability.vitaly_bot.hypotheses_learning_cycle import AutomaticInterestCycle
    from tof_reliability.vitaly_bot import hypotheses_pilot as pilot

    lab = package_lab
    baseline = copy.deepcopy(lab['baseline_payload'])
    old = baseline['knowledge_version']
    baseline['interest_context']['corpus_version'] = old
    baseline['dimension_projection']['profile_version'] = old
    baseline['dimension_projection']['corpus_version'] = old
    raw = {'decision': 'candidate', 'interest_id': lab['proposal']['id'],
           'dimension_id': 'repair_repeat_interval', 'requirement': lab['proposal']['requirement'],
           'scope': lab['proposal']['scope'], 'evidence': copy.deepcopy(lab['proposal']['evidence'])}
    _, candidate = AutomaticInterestCycle._candidate(None, baseline, lab['inventory'], raw)
    version = candidate['knowledge_version']
    assert candidate['interest_context']['version'] == version
    assert candidate['interest_context']['corpus_version'] == version
    assert candidate['dimension_projection']['profile_version'] == version
    assert candidate['dimension_projection']['corpus_version'] == version
    pilot._dimension_bindings(candidate['dimension_projection'], pilot._interests(candidate['interest_context']),
                              lab['scenario']['coverage'])


def test_production_composition_runs_real_callbacks_with_synthetic_runner(isolated_dir, package_lab, monkeypatch):
    from dataclasses import replace
    from tof_reliability.vitaly_bot import service as service_module
    from tof_reliability.vitaly_bot.coverage import RemoteCoverageProvider
    from tof_reliability.vitaly_bot.telegram import TelegramRuntime

    async def scenario():
        h, lab = Harness(isolated_dir), package_lab
        runtime = None
        try:
            class SyntheticProductionRunner(SharedRunner):
                async def run(self, prompt, schema, image_paths=None):
                    properties = schema['properties']
                    if 'cards' in properties:
                        self.active += 1
                        self.max_active = max(self.max_active, self.active)
                        try:
                            data = json.loads(prompt.split('Контекст:\n', 1)[1])
                            return ModelResult({'cards': lab['scenario']['generator'](data)}, '', '')
                        finally:
                            self.active -= 1
                    if 'supported' in properties and 'confidence' in properties:
                        self.active += 1
                        self.max_active = max(self.max_active, self.active)
                        try:
                            data = json.loads(prompt.split('\n', 1)[1])
                            return ModelResult({**lab['scenario']['verifier'](data), 'reason': ''}, '', '')
                        finally:
                            self.active -= 1
                    return await super().run(prompt, schema, image_paths)
            runner = SyntheticProductionRunner([], copy.deepcopy(lab['proposal']))
            monkeypatch.setattr(service_module, 'CodexRunner', lambda config: runner)
            async def coverage(self, request):
                return lab['scenario']['coverage_provider']()
            monkeypatch.setattr(RemoteCoverageProvider, 'build_for_pilot', coverage)
            config = replace(h.runtime.config, hypotheses_production={
                'directory': str(lab['store'].root), 'manifest_sha256': lab['anchor'],
                'authority_path': str(lab['root'] / 'authority.key'),
                'authority_fingerprint': sha(lab['root'] / 'authority.key'),
                'state_path': str(lab['root'] / 'trusted.sqlite'),
                'inventory_path': str(lab['root'] / 'inventory.json'),
                'audit_dir': str(lab['scenario']['audit_dir'])})
            runtime = TelegramRuntime(h.bot, config)
            meetings = add_accepted_meetings(runtime.store, lab['inventory'])
            for evidence, meeting in zip(runner.proposal['evidence'], meetings):
                evidence['source_id'] = meeting['id']
            monkeypatch.setattr(storage_module, '_local_now', lambda now=None: now or WINDOW)
            await runtime.start()
            engine = runtime.worker.interest_cycle
            await wait_for(lambda: engine.status()['status'] in {'published', 'rejected', 'failed', 'no_change'},
                           'production composition did not finish', timeout=60)
            assert engine.status()['status'] == 'published'
            assert manager(lab['store'], lab['anchor']).current()['version'] != lab['baseline']['version']
            assert runner.max_active == 1
        finally:
            if runtime is not None:
                await runtime.stop()
            await h.close()
    asyncio.run(scenario())


def test_new_evidence_cannot_count_a_known_historical_meeting_twice(package_lab):
    from test_hypotheses_auto_atomic_package import release_package
    from tof_reliability.vitaly_bot.hypotheses_auto_atomic_package import validate_source_evidence
    from tof_reliability.vitaly_bot.hypotheses_knowledge import VerifiedReleaseStore

    lab = package_lab
    inventory = copy.deepcopy(lab['inventory'])
    inventory['historical_sources'][0]['meeting_id'] = inventory['meetings'][0]['meeting_id']
    source = release_package(lab['root'] / 'overlap-baseline', lab['baseline_payload']['interest_context'],
        lab['baseline_payload']['dimension_projection'], inventory['historical_sources'],
        lab['baseline_payload']['knowledge_version'])
    store = VerifiedReleaseStore(lab['root'] / 'overlap-store')
    store.install(source, manifest_sha256=sha(source / 'manifest.json'))
    args = {name: copy.deepcopy(lab[name]) for name in
            ('candidate', 'proposal', 'classifier', 'verifier', 'classifier_trace', 'verifier_trace')}
    with pytest.raises(ValueError):
        validate_source_evidence(release_store=store, inventory=inventory, **args)


def test_same_day_without_historical_meeting_identity_is_ambiguous(package_lab):
    from test_hypotheses_auto_atomic_package import release_package
    from tof_reliability.vitaly_bot.hypotheses_auto_atomic_package import validate_source_evidence
    from tof_reliability.vitaly_bot.hypotheses_knowledge import VerifiedReleaseStore

    lab = package_lab
    inventory = copy.deepcopy(lab['inventory'])
    source_id = inventory['historical_sources'][0]['id']
    when = inventory['meetings'][0]['date']
    inventory['historical_sources'][0]['date'] = when
    baseline, candidate = copy.deepcopy(lab['baseline_payload']), copy.deepcopy(lab['candidate'])
    for payload in (baseline, candidate):
        for item in payload['interest_context']['interests']:
            for evidence in item['evidence']:
                if evidence['source_id'] == source_id:
                    evidence['date'] = when
        for dimension in payload['dimension_projection']['dimensions']:
            for evidence in dimension['evidence']:
                if evidence['source_id'] == source_id:
                    evidence['date'] = when
    source = release_package(lab['root'] / 'ambiguous-baseline', baseline['interest_context'],
        baseline['dimension_projection'], inventory['historical_sources'], baseline['knowledge_version'])
    store = VerifiedReleaseStore(lab['root'] / 'ambiguous-store')
    store.install(source, manifest_sha256=sha(source / 'manifest.json'))
    args = {name: copy.deepcopy(lab[name]) for name in
            ('proposal', 'classifier', 'verifier', 'classifier_trace', 'verifier_trace')}
    with pytest.raises(ValueError):
        validate_source_evidence(release_store=store, inventory=inventory, candidate=candidate, **args)


def test_cycle_checkpoint_syncs_parent_directory(isolated_dir, package_lab, monkeypatch):
    import stat
    async def scenario():
        h = Harness(isolated_dir)
        try:
            engine, _, _, _, _ = await setup(h, package_lab, monkeypatch)
            original = os.fsync
            synced = []
            def capture(fd):
                synced.append(stat.S_ISDIR(os.fstat(fd).st_mode))
                return original(fd)
            monkeypatch.setattr(os, 'fsync', capture)
            engine._save({'day': WINDOW.date().isoformat(), 'status': 'pending',
                          'stage': 'classify', 'published': False})
            assert True in synced, 'checkpoint rename must be durable across restart'
        finally:
            await h.close()
    asyncio.run(scenario())
