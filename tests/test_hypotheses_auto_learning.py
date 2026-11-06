"""Offline contracts for the 04:00 hypotheses interest-map lifecycle.

The existing release format and queue are exercised directly.  For the missing
lifecycle this file proposes an injectable seam, ``HypothesesInterestLearning``
(in learning_runtime), with ``tick(now=...)``, ``status()`` and
``rollback(version)``.  ``tick`` is async and returns a mapping containing a
status; a candidate is never considered published until the release pointer is
verified.  The constructor takes the existing store and release store, the
bot's runner, an explicitly supplied complete corpus inventory, and fixed
source-backed evaluation scenarios.  This seam is a test proposal, not an
assertion that the current implementation already has such an API.

All content, users, timestamps and traces here are invented.  No Telegram,
network, model process, real corpus, ledger, or production state is used.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from test_hypotheses_production_knowledge import _install, _package, _sha
from tof_reliability.vitaly_bot import learning_runtime
from tof_reliability.vitaly_bot.hypotheses_knowledge import VerifiedReleaseStore
from tof_reliability.vitaly_bot.models import ModelResult
from tof_reliability.vitaly_bot.storage import VitalyStore


ZONE = timezone(timedelta(hours=3))
WINDOW = datetime(2030, 11, 6, 4, 0, tzinfo=ZONE)
PRIVATE = 'SYNTHETIC_PRIVATE_FEEDBACK_DO_NOT_EXPORT'
MEETING_TEXT = 'Synthetic speaker: please compare the maintenance intervals.'


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class OfflineRunner:
    """One shared runner; only closed synthetic responses, never real calls."""

    def __init__(self):
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.fail_on = None
        self.same_thread = False
        self.conflicting = False
        self.proposal = {
            'decision': 'candidate',
            'interest_id': 'compare-maintenance-intervals',
            'requirement': 'Compare maintenance intervals before recommending action.',
            'scope': 'broad',
            'evidence': [
                {'source_id': 'new-meeting-a', 'locator': 'chars:0-55', 'attribution': 'speaker'},
                {'source_id': 'new-meeting-b', 'locator': 'chars:0-55', 'attribution': 'speaker'},
            ],
        }

    async def run(self, prompt, schema, image_paths=None):
        assert not image_paths
        assert schema['type'] == 'object' and schema.get('additionalProperties') is False
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)
            properties = schema.get('properties', {})
            role = ('verify' if 'verdicts' in properties or 'supported' in properties
                    else 'generate')
            self.calls.append({'role': role, 'prompt': prompt})
            if self.fail_on == role:
                raise RuntimeError('synthetic model failure')
            if role == 'verify':
                answer = {'verdicts': [
                    {**entry, 'interest_id': self.proposal['interest_id'],
                     'supported': True, 'confidence': 'high', 'conflicting': self.conflicting}
                    for entry in self.proposal['evidence']
                ]}
                if 'supported' in properties:
                    answer = {'supported': not self.conflicting}
            else:
                answer = copy.deepcopy(self.proposal)
            thread = 'synthetic-generator-thread' if role == 'generate' or self.same_thread else 'synthetic-verifier-thread'
            # stdout deliberately carries the trace; the verifier must not be
            # passed generator reasoning or accept an identical thread as independent.
            return ModelResult(answer, json.dumps({'thread_id': thread, 'answer': answer}), '')
        finally:
            self.active -= 1


def _source(source_id, text, path, date):
    return {'id': source_id, 'date': date, 'sha256': digest(text), 'path': str(path),
            'status': 'accepted', 'kind': 'meeting', 'text': text}


@pytest.fixture
def laboratory(tmp_path):
    """Private release/store paths live in the runner's isolated output area."""
    output = Path(os.environ.get('WORKFLOW_OUTPUT_DIR', str(tmp_path)))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='interest-tests-', dir=output) as name:
        root = Path(name)
        releases = VerifiedReleaseStore(root / 'releases')
        _install(releases, _package(root / 'baseline'))
        store = VitalyStore(root / 'state', root / 'knowledge')
        paths = [root / 'meeting-a.txt', root / 'meeting-b.txt']
        texts = [MEETING_TEXT, MEETING_TEXT + ' A second independent meeting.']
        meetings = []
        for index, (path, text) in enumerate(zip(paths, texts)):
            path.write_text(text, encoding='utf-8')
            source_id = 'new-meeting-' + 'ab'[index]
            meetings.append(_source(source_id, text, path, '2030-11-0' + str(index + 2)))
        # Historical entries reflect the exact pre-existing release inventory;
        # their source records must not be silently dropped or regenerated.
        inventory = {'complete': True, 'baseline_version': 'synthetic-release-1',
                     'historical_source_ids': ['meeting-early', 'meeting-late'],
                     'meetings': meetings,
                     'feedback': [{'event_id': 'feedback-a', 'owner_id': 101,
                                   'kind': 'text', 'text': PRIVATE,
                                   'generalized_issue': 'missing interval comparison'}]}
        scenarios = [
            {'id': 'missing-interval-comparison', 'source_ids': ['new-meeting-a', 'new-meeting-b'],
             'expected_interest_ids': ['compare-maintenance-intervals'], 'critical': False},
            {'id': 'preserve-early', 'source_ids': ['meeting-early'],
             'expected_interest_ids': ['interest-early'], 'critical': True},
            {'id': 'preserve-late', 'source_ids': ['meeting-late'],
             'expected_interest_ids': ['interest-late'], 'critical': True},
        ]
        yield {'root': root, 'store': store, 'releases': releases, 'inventory': inventory,
               'scenarios': scenarios, 'runner': OfflineRunner()}


def lifecycle(lab):
    cls = getattr(learning_runtime, 'HypothesesInterestLearning', None)
    assert cls is not None, '04:00 interest-map candidate lifecycle is missing'
    return cls(store=lab['store'], release_store=lab['releases'], runner=lab['runner'],
               inventory=lab['inventory'], scenarios=lab['scenarios'])


def tick(engine, when=WINDOW):
    return asyncio.run(engine.tick(now=when))


def current(lab):
    return lab['releases'].current()


def assert_not_published(lab, baseline, result):
    assert current(lab) == baseline
    assert result['status'] in {'pending', 'rejected', 'no_change', 'failed', 'skipped'}
    assert result.get('published') is not True


def test_existing_04_scheduler_skips_busy_user_and_resumes_only_next_window(laboratory, monkeypatch):
    store = laboratory['store']
    monkeypatch.setattr(store, 'learning_sources', lambda: [{'id': 'synthetic-source'}])
    case = store.create_case(101, 'synthetic-profile', None)
    foreground = store.begin_job(case['id'], 101, 'review')
    assert not store.due_learning_cycle(4, now=WINDOW - timedelta(minutes=1))
    assert not store.due_learning_cycle(4, now=WINDOW)
    assert store.next_queued_work(automatic_run_hour=4, now=WINDOW)['id'] == foreground['id']
    store.finish_job(foreground['id'])
    store.settle_job(case['id'])
    assert not store.due_learning_cycle(4, now=WINDOW + timedelta(hours=1))
    assert store.due_learning_cycle(4, now=WINDOW + timedelta(days=1))
    cycle = store.queue_learning_cycle(4, now=WINDOW + timedelta(days=1))
    assert cycle is not None
    assert not store.due_learning_cycle(4, now=WINDOW + timedelta(days=1, minutes=30))


def test_existing_release_binding_survives_activation_restart_and_rollback(laboratory):
    releases = laboratory['releases']
    old = releases.bind(profile_version='p', knowledge_package_id='g', case_id='c', run_id='old')
    _install(releases, _package(laboratory['root'] / 'second', 'synthetic-release-2'))
    new = releases.bind(profile_version='p', knowledge_package_id='g', case_id='c', run_id='new')
    reopened = VerifiedReleaseStore(laboratory['root'] / 'releases')
    assert reopened.load_bound(old)['knowledge_version'] == 'synthetic-release-1'
    assert reopened.load_bound(new)['knowledge_version'] == 'synthetic-release-2'
    reopened.activate('synthetic-release-1')
    assert reopened.load_bound(new)['knowledge_version'] == 'synthetic-release-2'
    assert reopened.current()['version'] == 'synthetic-release-1'


def test_interest_cycle_is_due_once_and_never_claims_queue_is_publication(laboratory):
    engine = lifecycle(laboratory)
    baseline = current(laboratory)
    assert tick(engine, WINDOW - timedelta(minutes=1))['status'] == 'skipped'
    result = tick(engine)
    if result['status'] == 'queued':
        assert result.get('published') is not True
        assert current(laboratory) == baseline
    assert tick(engine, WINDOW + timedelta(minutes=30))['status'] in {'skipped', 'pending'}
    assert tick(engine, WINDOW + timedelta(hours=1))['status'] in {'skipped', 'pending'}
    assert laboratory['runner'].max_active <= 1


def test_candidate_uses_complete_inventory_and_independent_trace(laboratory):
    engine = lifecycle(laboratory)
    outcome = tick(engine)
    status = engine.status()
    assert outcome['status'] in {'pending', 'published'}
    audit = status['audit']
    assert set(audit['source_ids']) == {'meeting-early', 'meeting-late', 'new-meeting-a', 'new-meeting-b'}
    assert audit['classifier_trace']['thread_id'] != audit['verifier_trace']['thread_id']
    assert audit['classifier_trace']['answer'] == audit['classifier']['answer']
    assert audit['verifier_trace']['answer'] == audit['verifier']['answer']
    for entry in laboratory['inventory']['meetings']:
        assert audit['source_sha256'][entry['id']] == entry['sha256']
    assert {e['locator'] for e in audit['new_evidence']} == {'chars:0-55'}
    assert PRIVATE not in json.dumps(audit['public_context'], ensure_ascii=False)
    assert laboratory['runner'].max_active == 1
    if outcome['status'] == 'published':
        assert current(laboratory)['manifest_sha256'] == outcome['manifest_sha256']
        assert current(laboratory)['version'] == outcome['version']
        assert laboratory['releases'].load(outcome['version'])['knowledge_version'] == outcome['version']


@pytest.mark.parametrize('defect', ['incomplete_inventory', 'missing_historical_source', 'changed_source',
                                   'single_meeting', 'same_meeting_twice', 'conflicting_verdict', 'same_trace'])
def test_unverifiable_evidence_cannot_activate(laboratory, defect):
    inv, runner = laboratory['inventory'], laboratory['runner']
    if defect == 'incomplete_inventory':
        inv['complete'] = False
    elif defect == 'missing_historical_source':
        inv['historical_source_ids'].remove('meeting-late')
    elif defect == 'changed_source':
        Path(inv['meetings'][0]['path']).write_text('changed after hashing', encoding='utf-8')
    elif defect == 'single_meeting':
        inv['meetings'].pop()
        runner.proposal['evidence'].pop()
    elif defect == 'same_meeting_twice':
        runner.proposal['evidence'][1] = copy.deepcopy(runner.proposal['evidence'][0])
    elif defect == 'conflicting_verdict':
        runner.conflicting = True
    else:
        runner.same_thread = True
    baseline = current(laboratory)
    assert_not_published(laboratory, baseline, tick(lifecycle(laboratory)))


@pytest.mark.parametrize('mode', ['feedback_only', 'negative_mark', 'no_change', 'no_effect', 'critical_regression'])
def test_feedback_and_quality_gate_cannot_fabricate_improvement(laboratory, mode):
    runner, inv = laboratory['runner'], laboratory['inventory']
    if mode == 'feedback_only':
        inv['meetings'] = []
        runner.proposal['evidence'] = [{'source_id': 'feedback-a', 'locator': 'event:feedback-a'}]
    elif mode == 'negative_mark':
        inv['feedback'] = [{'event_id': 'negative-a', 'owner_id': 101, 'kind': 'negative'}]
        inv['meetings'] = []
    elif mode == 'no_change':
        runner.proposal['decision'] = 'none'
    elif mode == 'no_effect':
        laboratory['scenarios'][0]['expected_interest_ids'] = ['interest-early']
    else:
        # The proposed change must preserve the existing release's critical capability.
        runner.proposal['remove_interest_ids'] = ['interest-late']
    baseline = current(laboratory)
    result = tick(lifecycle(laboratory))
    assert_not_published(laboratory, baseline, result)


@pytest.mark.parametrize('failed_role', ['generate', 'verify'])
def test_model_failure_keeps_previous_release_and_has_resumable_status(laboratory, failed_role):
    laboratory['runner'].fail_on = failed_role
    engine = lifecycle(laboratory)
    baseline = current(laboratory)
    result = tick(engine)
    assert_not_published(laboratory, baseline, result)
    saved = engine.status()
    assert saved['status'] in {'failed', 'pending', 'rejected'}
    assert saved.get('published') is not True
    laboratory['runner'].fail_on = None
    restarted = lifecycle(laboratory)
    assert tick(restarted, WINDOW + timedelta(hours=2))['status'] in {'skipped', 'pending'}
    resumed = tick(restarted, WINDOW + timedelta(days=1))
    assert resumed['status'] in {'pending', 'published', 'rejected', 'no_change'}
    assert laboratory['runner'].max_active <= 1


def test_feedback_owner_and_raw_mixed_source_content_never_reach_new_run_context(laboratory):
    engine = lifecycle(laboratory)
    result = tick(engine)
    status = engine.status()
    exported = json.dumps(status.get('public_context', status.get('audit', {}).get('public_context', {})), ensure_ascii=False)
    assert PRIVATE not in exported
    assert MEETING_TEXT not in exported
    assert 'owner_id' not in exported and 'feedback-a' not in exported
    if result['status'] == 'published':
        context = laboratory['releases'].load(result['version'])['interest_context']
        assert PRIVATE not in json.dumps(context, ensure_ascii=False)
        assert MEETING_TEXT not in json.dumps(context, ensure_ascii=False)


def test_no_duplicate_meeting_is_added_on_retry_and_active_bindings_survive_rollback(laboratory):
    releases = laboratory['releases']
    bound = releases.bind(profile_version='p', knowledge_package_id='g', case_id='c', run_id='r')
    engine = lifecycle(laboratory)
    first = tick(engine)
    tick(lifecycle(laboratory), WINDOW + timedelta(days=1))
    status = lifecycle(laboratory).status()
    source_ids = status.get('audit', {}).get('source_ids', [])
    assert len(source_ids) == len(set(source_ids))
    assert releases.load_bound(bound)['knowledge_version'] == 'synthetic-release-1'
    if first['status'] == 'published':
        new_binding = releases.bind(profile_version='p', knowledge_package_id='g', case_id='c', run_id='r2')
        engine.rollback('synthetic-release-1')
        assert releases.current()['version'] == 'synthetic-release-1'
        assert releases.load_bound(new_binding)['knowledge_version'] == first['version']


def test_invalid_locator_is_recorded_and_cannot_be_published(laboratory):
    laboratory['runner'].proposal['evidence'][0]['locator'] = 'chars:900-1000'
    baseline = current(laboratory)
    engine = lifecycle(laboratory)
    result = tick(engine)
    assert_not_published(laboratory, baseline, result)
    assert 'invalid_locator' in engine.status().get('blockers', [])


def test_model_trace_must_match_response(laboratory):
    runner = laboratory['runner']
    original = runner.run

    async def forged(prompt, schema, image_paths=None):
        result = await original(prompt, schema, image_paths)
        trace = json.loads(result.stdout)
        trace['answer'] = {'forged': True}
        return ModelResult(result.value, json.dumps(trace), '')

    runner.run = forged
    baseline = current(laboratory)
    engine = lifecycle(laboratory)
    assert_not_published(laboratory, baseline, tick(engine))
    assert engine.status()['status'] in {'rejected', 'failed'}


def test_existing_foreground_job_prevents_model_calls(laboratory):
    store = laboratory['store']
    case = store.create_case(101, 'synthetic-profile', None)
    store.begin_job(case['id'], 101, 'review')
    baseline = current(laboratory)
    result = tick(lifecycle(laboratory))
    assert_not_published(laboratory, baseline, result)
    assert laboratory['runner'].calls == []


def test_verifier_failure_resumes_without_regenerating_candidate(laboratory):
    runner = laboratory['runner']
    runner.fail_on = 'verify'
    engine = lifecycle(laboratory)
    assert tick(engine)['status'] == 'failed'
    runner.fail_on = None
    assert tick(lifecycle(laboratory), WINDOW + timedelta(days=1))['status'] == 'pending'
    assert [call['role'] for call in runner.calls] == ['generate', 'verify', 'verify']


def test_user_job_arriving_between_model_calls_defers_verifier(laboratory):
    runner = laboratory['runner']
    original = runner.run
    async def enqueue_after_generate(prompt, schema, image_paths=None):
        result = await original(prompt, schema, image_paths)
        if len(runner.calls) == 1:
            store = laboratory['store']
            case = store.create_case(101, 'synthetic-profile', None)
            store.begin_job(case['id'], 101, 'review')
        return result
    runner.run = enqueue_after_generate
    assert tick(lifecycle(laboratory))['status'] == 'pending'
    assert [call['role'] for call in runner.calls] == ['generate']


def test_codex_jsonl_trace_is_parsed_without_fabricating_a_thread(laboratory):
    runner = laboratory['runner']
    original = runner.run
    async def jsonl(prompt, schema, image_paths=None):
        result = await original(prompt, schema, image_paths)
        trace = json.loads(result.stdout)
        events = [{'type': 'thread.started', 'thread_id': trace['thread_id']},
                  {'type': 'item.completed', 'item': {'type': 'agent_message',
                   'text': json.dumps(result.value)}}, {'type': 'turn.completed'}]
        return ModelResult(result.value, '\n'.join(json.dumps(e) for e in events), '')
    runner.run = jsonl
    engine = lifecycle(laboratory)
    assert tick(engine)['status'] == 'pending'
    assert engine.status()['audit']['classifier_trace']['thread_id'] == 'synthetic-generator-thread'


def test_same_meeting_content_under_two_ids_is_not_independent(laboratory):
    meetings = laboratory['inventory']['meetings']
    meetings[1]['text'] = meetings[0]['text']
    Path(meetings[1]['path']).write_text(meetings[1]['text'], encoding='utf-8')
    meetings[1]['sha256'] = meetings[0]['sha256']
    baseline = current(laboratory)
    assert_not_published(laboratory, baseline, tick(lifecycle(laboratory)))
    assert lifecycle(laboratory).status()['status'] == 'rejected'


def test_process_interruption_after_checkpoint_resumes_verifier(laboratory):
    runner = laboratory['runner']
    runner.fail_on = 'verify'
    engine = lifecycle(laboratory)
    assert tick(engine)['status'] == 'failed'
    checkpoint = json.loads(engine.path.read_text(encoding='utf-8'))
    checkpoint.update(status='pending', stage='verify', blockers=[])
    engine._save(checkpoint)
    runner.fail_on = None
    assert tick(lifecycle(laboratory), WINDOW + timedelta(days=1))['status'] == 'pending'
    assert [call['role'] for call in runner.calls] == ['generate', 'verify', 'verify']


def test_checkpoint_remains_resumable_if_interrupted_during_input_revalidation(laboratory):
    runner = laboratory['runner']
    runner.fail_on = 'verify'
    engine = lifecycle(laboratory)
    assert tick(engine)['status'] == 'failed'
    checkpoint = json.loads(engine.path.read_text(encoding='utf-8'))
    checkpoint.update(status='pending', stage='inputs', blockers=[])
    engine._save(checkpoint)
    runner.fail_on = None
    assert tick(lifecycle(laboratory), WINDOW + timedelta(days=1))['status'] == 'pending'
    assert [call['role'] for call in runner.calls] == ['generate', 'verify', 'verify']
