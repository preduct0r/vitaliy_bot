"""Red-first S4a service contracts; all identities and inputs are synthetic.

The public seam specified here is deliberately small: HypothesesService is
injected into VitalyService and shares its FifoWorker.  ``submit`` returns a
persisted job; ``get_run`` and feedback methods enforce the supplied owner.
No Telegram client, network, real model, production database, or private corpus
is used.  The existing dimension fixture supplies validated synthetic cards.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from test_hypotheses_dimensions_pilot import dimension_case, topic_case  # fixtures
from test_hypotheses_adjudicated_pilot import adjudication_case, source_case  # fixtures
from tof_reliability.vitaly_bot import service as service_module
from tof_reliability.vitaly_bot.coverage import RemoteCoverageProvider
from tof_reliability.vitaly_bot.documents import DocumentProcessor
from tof_reliability.vitaly_bot.hypotheses_pilot import TrustedRequestStore
from tof_reliability.vitaly_bot.models import ArtifactInput, BotConfig, EvidenceError, ModelResult, StateError
from tof_reliability.vitaly_bot.profile import load_default_profile
from tof_reliability.vitaly_bot.service import FifoWorker, VitalyService
from tof_reliability.vitaly_bot.storage import VitalyStore


class FakeCoverage:
    def __init__(self, coverage):
        self.coverage = copy.deepcopy(coverage)
        self.calls = []
        self.failure = None

    async def build(self, request):
        self.calls.append(copy.deepcopy(request))
        if self.failure:
            raise self.failure
        return copy.deepcopy(self.coverage)


class FakeKnowledge:
    """Only synthetic fixture contents; production loading must verify audits."""
    def __init__(self, context, projection):
        self.context = copy.deepcopy(context)
        self.projection = copy.deepcopy(projection)
        self.version = 'synthetic-knowledge-v1'
        self.calls = []

    def load(self, profile_version, knowledge_package_id):
        self.calls.append((profile_version, knowledge_package_id))
        return {'knowledge_version': self.version, 'interest_context': copy.deepcopy(self.context),
                'dimension_projection': copy.deepcopy(self.projection)}


class FakeRunner:
    def __init__(self, cards):
        self.cards = copy.deepcopy(cards)
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.heartbeat = 0

    async def run(self, prompt, schema, image_paths=None):
        self.calls.append((prompt, copy.deepcopy(schema)))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            # This await requires the main loop to remain free while the
            # synchronous pilot bridges to the async runner.
            await asyncio.sleep(0)
            self.heartbeat += 1
            assert schema['type'] == 'object'
            assert schema.get('additionalProperties') is False
            assert schema.get('required')
            if 'cards' in schema['properties']:
                cards = copy.deepcopy(self.cards)
                context, _ = json.JSONDecoder().raw_decode(prompt.split('Контекст:\n', 1)[1])
                if context['scope']:
                    for card in cards:
                        card['scope'] = context['scope']
                for card in cards:
                    card['case_material_references'] = []
                if 'CURRENT_CASE_SYNTHETIC' in prompt:
                    material_json = prompt.rsplit('иначе пустой список.\n', 1)[1]
                    materials = json.loads(material_json)
                    cards[0]['case_material_references'] = [next(
                        item['reference'] for item in materials if 'CURRENT_CASE_SYNTHETIC' in item['text'])]
                return ModelResult({'cards': cards}, '', '')
            return ModelResult({'supported': True, 'confidence': 'high', 'reason': ''}, '', '')
        finally:
            self.active -= 1


def make_runtime(tmp_path, dimension_case):
    # Keep construction inside each test: the missing S4a seam is an assertion
    # failure, rather than an import/collection failure that masks other tests.
    cls = getattr(service_module, 'HypothesesService', None)
    assert cls is not None, 'S4a must provide injectable HypothesesService'
    context, coverage, projection, cards = copy.deepcopy(dimension_case)
    for card in cards:
        card['scope'] = ''
    config = BotConfig(token='synthetic-unused', state_dir=tmp_path / 'state',
                       codex_bin=tmp_path / 'unused-codex')
    store = VitalyStore(config.state_dir)
    runner = FakeRunner(cards)
    provider = FakeCoverage(coverage)
    knowledge = FakeKnowledge(context, projection)
    trusted = TrustedRequestStore(tmp_path / 'trusted.sqlite')
    hypotheses = cls(store=store, config=config, runner=runner,
                     coverage_provider=provider, knowledge_loader=knowledge,
                     trusted_store=trusted)
    service = VitalyService(store, load_default_profile(), config, runner=runner,
                            hypotheses_service=hypotheses)
    return store, service, hypotheses, runner, provider, knowledge, trusted


def submit(hypotheses, case, owner=101, description=''):
    return hypotheses.submit(case_id=case['id'], chat_id=case['chat_id'],
                             user_id=owner, description=description)


async def drain(store, service, *jobs):
    completed, failures = [], []
    async def on_completed(job, outcome):
        completed.append((job, outcome))
    async def on_failed(job, error):
        failures.append((job, error))
    worker = FifoWorker(store, service, on_completed, on_failed)
    await worker.start()
    try:
        worker.notify()
        async def finished():
            while any(store.get_job(j['id'])['status'] in ('queued', 'running') for j in jobs):
                await asyncio.sleep(.005)
        await asyncio.wait_for(finished(), timeout=3)
    finally:
        await asyncio.wait_for(worker.stop(), timeout=3)
    return completed, failures


def add_text(store, case, tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding='utf-8')
    return DocumentProcessor(store).ingest(
        case['id'], ArtifactInput(name, name, path, path.stat().st_size))


def persisted(hypotheses, job, owner=101):
    return hypotheses.get_run(job_id=job['id'], user_id=owner)


def test_new_hypothesis_run_uses_only_approved_generic_learning_rules(tmp_path, dimension_case):
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    package = store.active_knowledge_package()
    data = package['package']
    data['analysis_rules'] = [{'transfer_key': 'separate_fact_from_scenario',
                               'guidance': 'PRIVATE_OTHER_USER_FACT_SHOULD_NOT_APPEAR'}]
    with store._connect() as db:
        db.execute('UPDATE knowledge_packages SET package_json=? WHERE id=?',
                   (json.dumps(data), package['id']))
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    prompts = [prompt for prompt, schema in runner.calls if 'cards' in schema['properties']]
    assert prompts and 'separate_fact_from_scenario' in prompts[0]
    assert 'PRIVATE_OTHER_USER_FACT_SHOULD_NOT_APPEAR' not in prompts[0]


def test_hypothesis_model_stages_are_visible_in_case_status(tmp_path, dimension_case):
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    job = submit(hypotheses, case)
    original = runner.run
    stages = []
    async def capture(prompt, schema, image_paths=None):
        stages.append(store.get_case(case['id'])['activity_detail'])
        return await original(prompt, schema, image_paths)
    runner.run = capture
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    assert any('гипотез' in stage.lower() for stage in stages)
    assert any('провер' in stage.lower() for stage in stages)


def test_recovered_s4a_job_keeps_its_original_knowledge_binding(tmp_path, dimension_case):
    from tof_reliability.vitaly_bot.hypotheses_service import _digest
    store, service, hypotheses, runner, _, knowledge, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    job = submit(hypotheses, case)
    old_hash = _digest(knowledge.load(case['profile_version'], case['knowledge_package_id']))
    with store._connect() as db:
        db.execute('UPDATE hypotheses_runs SET knowledge_hash=? WHERE job_id=?', (old_hash, job['id']))
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    assert persisted(hypotheses, job)['status'] == 'completed'


def test_trusted_binding_and_sealed_result_recover_without_rebinding(tmp_path):
    store = TrustedRequestStore(tmp_path / 'requests.sqlite')
    request = {'run_id': 'run', 'case_id': 'case', 'user_id': '101', 'scope': ''}
    trusted = store.create(request)
    run = {'request_binding': trusted.token, 'scope': '', 'cards': []}
    store.seal(trusted, run)
    reopened = TrustedRequestStore(store.path)
    recovered, result = reopened.recover(request)
    assert recovered == trusted and result == run
    with pytest.raises(ValueError):
        reopened.recover({**request, 'scope': 'changed'})
    # An older pilot writes the original seven-column table positionally.
    with sqlite3.connect(store.path) as db:
        db.execute('INSERT INTO trusted_requests VALUES (?, ?, ?, ?, ?, ?, ?)',
                   ('old-run', 'old-case', '101', '', 'old-token', 'old-digest', None))


def test_private_loader_replays_full_audit_and_rejects_tampering(tmp_path, adjudication_case):
    from tof_reliability.vitaly_bot.hypotheses_knowledge import VerifiedKnowledgeLoader
    from tof_reliability.vitaly_bot.hypotheses_topic_projection import adjudicate_conflicts
    data = adjudication_case
    corpus, original, adjudicated = (tmp_path / name for name in ('corpus', 'original', 'adjudicated'))
    for directory in (corpus, original, adjudicated):
        directory.mkdir()
    def write(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding='utf-8')
    write(corpus / 'training-interest-map.json', data['profile'])
    write(corpus / 'training-index.json', data['source_index'])
    write(original / 'repair-dimensions-private.json', data['original_projection'])
    write(original / 'classifier-bounded-000.json', {'answer': {'candidates': data['original_candidates']}})
    raw_verdicts = [{**{k: v[k] for k in ('interest_id', 'dimension_id', 'source_id', 'locator')}, **v['verdict']}
                    for v in data['original_verdicts']]
    write(original / 'verifier-bounded-000.json', {'answer': {'verdicts': raw_verdicts}})
    for number, answer in enumerate(({'candidates': data['original_candidates']}, {'verdicts': raw_verdicts}), 1):
        trace = [
            {'type': 'thread.started', 'thread_id': f'synthetic-thread-{number}'},
            {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': json.dumps(answer, ensure_ascii=False)}},
            {'type': 'turn.completed'},
        ]
        (original / f'trace-{number:04}.jsonl').write_text('\n'.join(json.dumps(e, ensure_ascii=False) for e in trace), encoding='utf-8')
    write(adjudicated / 'original-candidates.json', data['original_candidates'])
    write(adjudicated / 'original-verdicts-normalized.json', data['original_verdicts'])
    write(adjudicated / 'adjudications-private.json', data['adjudications'])
    write(adjudicated / 'revised-projection-private.json', adjudicate_conflicts(**data))
    files = list(corpus.glob('*.json')) + list(original.glob('*.json')) + list(original.glob('*.jsonl'))
    outputs = list(adjudicated.glob('*.json'))
    manifest = {'version': 'conflict-adjudication-audit.v1', 'developmental_only': True,
                'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                'output_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}}
    write(adjudicated / 'audit-manifest.json', manifest)
    loader = VerifiedKnowledgeLoader(corpus_dir=corpus, original_dir=original,
                                    adjudication_dir=adjudicated, profile_version='case-profile',
                                    knowledge_package_id='package-x',
                                    manifest_sha256=hashlib.sha256((adjudicated / 'audit-manifest.json').read_bytes()).hexdigest())
    knowledge = loader.load('case-profile', 'package-x')
    assert knowledge['interest_context']['version'] == data['profile']['version']
    assert knowledge['dimension_projection']['dimensions']
    with pytest.raises(ValueError):
        loader.load('different-profile', 'package-x')
    (original / 'verifier-bounded-000.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError):
        loader.load('case-profile', 'package-x')


def test_atomic_hypotheses_admission_persists_job_and_run_together(tmp_path):
    store = VitalyStore(tmp_path / 'state')
    case = store.create_case(101, 'profile-test')
    def attempt(_):
        try:
            return store.begin_hypotheses_run(case['id'], 101, 101, '')
        except StateError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = [row for row in pool.map(attempt, range(2)) if row]
    assert len(accepted) == 1
    with store._connect() as db:
        rows = db.execute('SELECT r.job_id, r.user_id, j.status FROM hypotheses_runs r JOIN jobs j ON j.id=r.job_id').fetchall()
    assert [(row['job_id'], row['user_id'], row['status']) for row in rows] == [(accepted[0]['id'], 101, 'queued')]


def test_atomic_admission_rejects_foreign_owner_without_creating_work(tmp_path):
    store = VitalyStore(tmp_path / 'state')
    case = store.create_case(101, 'profile-test')
    with pytest.raises((StateError, ValueError, PermissionError)):
        store.begin_hypotheses_run(case['id'], 101, 202, '')
    with store._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM hypotheses_runs').fetchone()[0] == 0


def test_remote_coverage_reports_generation_drift_distinctly(monkeypatch):
    class Process:
        returncode = 1

        async def communicate(self, _program):
            return b'', b'RuntimeError: published_generation_changed_during_coverage\n'

    async def spawn(*_args, **_kwargs):
        return Process()

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    with pytest.raises(EvidenceError) as error:
        asyncio.run(RemoteCoverageProvider().build({'version': 1}))
    assert error.value.code == 'generation_drift'


def test_remote_pilot_coverage_requests_single_server_attempt(monkeypatch):
    seen = []
    class Process:
        returncode = 0
        async def communicate(self, program):
            seen.append(program.decode('utf-8'))
            return json.dumps({'coverage_version': 1, 'generation': {'run_id': 'g1'}, 'routes': []}).encode(), b''
    async def spawn(*_args, **_kwargs):
        return Process()
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    asyncio.run(RemoteCoverageProvider().build_for_pilot({'version': 1}))
    assert 'max_generation_retries=0' in seen[0]


def test_server_single_attempt_reports_drift_without_internal_retry(tmp_path, monkeypatch):
    from tof_reliability.vitaly_bot import server_coverage
    ledger, catalog = tmp_path / 'ledger', tmp_path / 'catalog'
    ledger.mkdir()
    catalog.write_text('', encoding='utf-8')
    calls = []
    def generation(_path):
        calls.append(None)
        return {'run_id': f'g{len(calls)}', 'ledger_path': str(ledger), 'catalog_path': str(catalog)}
    monkeypatch.setattr(server_coverage, '_generation', generation)
    monkeypatch.setattr(server_coverage, '_collect', lambda *_: [])
    with pytest.raises(RuntimeError, match='generation_changed'):
        server_coverage.build_coverage({'version': 1}, generation_state_path=tmp_path / 'state',
                                       root=tmp_path, max_generation_retries=0)
    assert len(calls) == 2


def test_empty_request_enters_real_fifo_and_persists_bound_snapshot(tmp_path, dimension_case):
    store, service, hypotheses, runner, provider, knowledge, trusted = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    job = submit(hypotheses, case)
    assert job['kind'] == 'hypotheses' and job['status'] == 'queued'
    assert store.next_queued_work()['id'] == job['id']
    completed, failures = asyncio.run(drain(store, service, job))
    assert not failures and len(completed) == 1
    run = persisted(hypotheses, job)
    assert run['status'] == 'completed' and run['user_id'] == 101
    assert run['case_id'] == case['id'] and run['user_request'] == ''
    assert run['artifact_hashes'] == [] and run['case_hash']
    assert run['profile_version'] == case['profile_version']
    assert run['interest_version'] == knowledge.context['version']
    assert run['knowledge_version'] == knowledge.version
    assert run['generation_id'] == provider.coverage['generation']['run_id']
    assert run['coverage'] == provider.coverage
    assert run['created_at'] and run['started_at'] and run['finished_at']
    assert 5 <= len(run['cards']) <= 7
    assert all(c['source_references'] and c['missing_evidence'] for c in run['cards'])
    interests = {item['id']: item for item in knowledge.context['interests']}
    for card in run['cards']:
        meeting = card['meeting_interest']
        evidence = interests[meeting['interest_id']]['evidence'][0]
        reference_id = hashlib.sha256((evidence['source_id'] + '\0' + evidence['locator']).encode()).hexdigest()[:16]
        assert reference_id in meeting['reference'] and 'chars:' not in meeting['reference']
    assert runner.calls and runner.heartbeat and runner.max_active == 1
    # The pilot checks the published snapshot before generation, after the
    # generator and again after independent review; it must not cache it.
    assert len(provider.calls) == 3 and knowledge.calls
    assert all(request == provider.calls[0] for request in provider.calls)
    # The request must be independently bound before generation, not trusted
    # merely because its final JSON repeats the user's scope.
    with sqlite3.connect(trusted.path) as db:
        row = db.execute('SELECT run_id, case_id, user_id, scope, result_digest FROM trusted_requests').fetchone()
    assert row == (run['id'], case['id'], '101', '', row[4]) and row[4]


def test_user_scope_current_files_hashes_and_source_references(tmp_path, dimension_case):
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    artifact = add_text(store, case, tmp_path, 'current.txt', 'CURRENT_CASE_SYNTHETIC')
    job = submit(hypotheses, case, description='насосы')
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    run = persisted(hypotheses, job)
    assert run['user_request'] == 'насосы'
    assert artifact['sha256'] in run['artifact_hashes']
    assert any('current.txt' in json.dumps(c, ensure_ascii=False) for c in run['cards'])
    assert any('CURRENT_CASE_SYNTHETIC' in prompt for prompt, _ in runner.calls)


def test_changed_extracted_text_after_admission_fails_closed(tmp_path, dimension_case):
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    artifact = add_text(store, case, tmp_path, 'current.txt', 'CURRENT_CASE_SYNTHETIC')
    job = submit(hypotheses, case)
    Path(artifact['normalized_text_path']).write_text('CHANGED_AFTER_QUEUE', encoding='utf-8')
    _, failures = asyncio.run(drain(store, service, job))
    assert failures and persisted(hypotheses, job)['status'] == 'failed'
    assert not runner.calls


def test_duplicate_filenames_have_distinct_current_case_citations(tmp_path, dimension_case):
    store, service, hypotheses, _, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    first = tmp_path / 'first.txt'
    second = tmp_path / 'second.txt'
    first.write_text('first synthetic material', encoding='utf-8')
    second.write_text('second synthetic material', encoding='utf-8')
    processor = DocumentProcessor(store)
    for ident, source in [('a', first), ('b', second)]:
        processor.ingest(case['id'], ArtifactInput(ident, 'same.txt', source, source.stat().st_size))
    job = submit(hypotheses, case)
    materials = hypotheses._materials(persisted(hypotheses, job))
    assert len(materials) == 2 and len({item['reference'] for item in materials}) == 2
    assert all('same.txt' in item['reference'] for item in materials)


def test_changed_knowledge_after_admission_fails_before_model(tmp_path, dimension_case):
    store, service, hypotheses, runner, _, knowledge, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    job = submit(hypotheses, case)
    knowledge.context['version'] = 'different-version'
    _, failures = asyncio.run(drain(store, service, job))
    assert failures and persisted(hypotheses, job)['status'] == 'failed'
    assert not runner.calls


def test_forged_case_reference_is_rejected(tmp_path, dimension_case):
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    original_run = runner.run
    async def forged(prompt, schema, image_paths=None):
        result = await original_run(prompt, schema, image_paths)
        if 'cards' in result.value:
            result.value['cards'][0]['case_material_references'] = ['Файл: чужой-кейс.txt']
        return result
    runner.run = forged
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert failures and persisted(hypotheses, job)['status'] == 'failed'
    assert not persisted(hypotheses, job)['cards']


@pytest.mark.parametrize('stage', ['generator', 'verifier'])
def test_invalid_model_schema_fails_closed(tmp_path, dimension_case, stage):
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    original_run = runner.run
    async def invalid(prompt, schema, image_paths=None):
        result = await original_run(prompt, schema, image_paths)
        if stage == 'generator' and 'cards' in result.value:
            del result.value['cards'][0]['case_material_references']
        if stage == 'verifier' and 'supported' in result.value:
            del result.value['reason']
        return result
    runner.run = invalid
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert failures and persisted(hypotheses, job)['status'] == 'failed'
    assert not persisted(hypotheses, job)['cards']


def test_sealed_result_recovery_retains_selected_case_citations(tmp_path, dimension_case):
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    add_text(store, case, tmp_path, 'current.txt', 'CURRENT_CASE_SYNTHETIC')
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    before = persisted(hypotheses, job)
    assert any(card['case_material_references'] for card in before['cards'])
    calls = len(runner.calls)
    # Simulate a crash after trusted sealing but before writing the case DB.
    with store._connect() as db:
        db.execute("UPDATE jobs SET status='running' WHERE id=?", (job['id'],))
        db.execute("UPDATE hypotheses_runs SET status='running', cards_json='[]', generation_id=NULL WHERE job_id=?", (job['id'],))
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures and len(runner.calls) == calls
    after = persisted(hypotheses, job)
    assert after['cards'] == before['cards']


def test_sealed_result_recovery_rejects_changed_generation(tmp_path, dimension_case):
    store, service, hypotheses, runner, provider, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    calls = len(runner.calls)
    with store._connect() as db:
        db.execute("UPDATE jobs SET status='running' WHERE id=?", (job['id'],))
        # The case DB may already contain the staged cards when the process dies.
        db.execute("UPDATE hypotheses_runs SET status='running' WHERE job_id=?", (job['id'],))
    provider.coverage['generation']['run_id'] += '-new'
    _, failures = asyncio.run(drain(store, service, job))
    assert failures and len(runner.calls) == calls
    run = persisted(hypotheses, job)
    assert run['status'] == 'failed' and 'generation_drift' in run['error'] and not run['cards']


def test_expired_and_other_users_files_cannot_enter_new_case(tmp_path, dimension_case):
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    old = service.new_case(101)
    add_text(store, old, tmp_path, 'old.txt', 'EXPIRED_SECRET_SYNTHETIC')
    other = service.new_case(202)
    add_text(store, other, tmp_path, 'other.txt', 'OTHER_USER_SECRET_SYNTHETIC')
    past = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    with store._connect() as db:
        db.execute('UPDATE cases SET last_user_activity_at=? WHERE id=?', (past, old['id']))
    current, expired = service.case_for_user(101)
    assert expired == old['id'] and current['id'] != old['id']
    job = submit(hypotheses, current)
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    result = json.dumps(persisted(hypotheses, job), ensure_ascii=False)
    prompts = json.dumps(runner.calls, ensure_ascii=False)
    for secret in ('EXPIRED_SECRET_SYNTHETIC', 'OTHER_USER_SECRET_SYNTHETIC', 'old.txt', 'other.txt'):
        assert secret not in result and secret not in prompts
    with pytest.raises((StateError, ValueError, PermissionError)):
        persisted(hypotheses, job, owner=202)


def test_private_fragments_and_other_case_facts_never_reach_model(tmp_path, dimension_case):
    context, _, projection, _ = dimension_case
    private = 'SYNTHETIC_PRIVATE_TRANSCRIPT_QUOTE_842'
    context['interests'][0]['evidence'][0]['fragment'] = private
    projection['dimensions'][0]['evidence'][0]['fragment'] = private
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    assert private not in json.dumps(runner.calls, ensure_ascii=False)
    assert 'chars:' not in json.dumps(runner.calls, ensure_ascii=False)


def test_unavailable_server_fails_persistently_without_model_or_fallback(tmp_path, dimension_case):
    store, service, hypotheses, runner, provider, _, _ = make_runtime(tmp_path, dimension_case)
    provider.failure = EvidenceError('tof_unavailable', 'synthetic outage')
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert len(failures) == 1 and store.get_job(job['id'])['status'] == 'failed'
    run = persisted(hypotheses, job)
    assert run['status'] == 'failed' and 'tof_unavailable' in str(run['error'])
    assert not run['cards'] and not runner.calls and len(provider.calls) == 1


def test_provider_exhausted_generation_drift_fails_without_outer_retry(tmp_path, dimension_case):
    store, service, hypotheses, runner, provider, _, _ = make_runtime(tmp_path, dimension_case)
    async def drifting(request):
        provider.calls.append(copy.deepcopy(request))
        raise EvidenceError('generation_drift', 'server retry exhausted')
    provider.build = drifting
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert len(provider.calls) == 1
    run = persisted(hypotheses, job)
    assert failures and run['status'] == 'failed' and not runner.calls
    assert 'generation_drift' in str(run['error'])


def test_pilot_owns_one_retry_when_coverage_operation_has_no_internal_retry(tmp_path, dimension_case):
    store, service, hypotheses, runner, provider, _, _ = make_runtime(tmp_path, dimension_case)
    async def single_attempt(request):
        provider.calls.append(copy.deepcopy(request))
        if len(provider.calls) == 1:
            raise EvidenceError('generation_drift', 'single attempt detected drift')
        return copy.deepcopy(provider.coverage)
    provider.build_for_pilot = single_attempt
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures and persisted(hypotheses, job)['status'] == 'completed'
    assert len(provider.calls) == 4


def test_server_and_between_stage_drifts_share_one_retry_budget(tmp_path, dimension_case):
    store, service, hypotheses, _, provider, _, _ = make_runtime(tmp_path, dimension_case)
    async def single_attempt(request):
        provider.calls.append(copy.deepcopy(request))
        if len(provider.calls) == 1:
            raise EvidenceError('generation_drift', 'single attempt detected drift')
        result = copy.deepcopy(provider.coverage)
        if len(provider.calls) >= 3:
            result['generation']['run_id'] += '-changed'
        return result
    provider.build_for_pilot = single_attempt
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert failures and persisted(hypotheses, job)['status'] == 'failed'
    assert len(provider.calls) == 3


@pytest.mark.parametrize('always_drifts', [False, True])
def test_generation_change_between_pilot_checks_retries_pilot_once(tmp_path, dimension_case, always_drifts):
    store, service, hypotheses, runner, provider, _, _ = make_runtime(tmp_path, dimension_case)
    normal = provider.build
    async def drifting(request):
        value = await normal(request)
        # First attempt observes A then B. The retry observes either B
        # throughout, or B then C for a second, terminal drift.
        call = len(provider.calls)
        if call >= 2:
            value['generation']['run_id'] += '-changed'
        if always_drifts and call >= 4:
            value['generation']['run_id'] += '-again'
        return value
    provider.build = drifting
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    run = persisted(hypotheses, job)
    if always_drifts:
        assert len(provider.calls) == 4
        assert failures and run['status'] == 'failed'
    else:
        assert len(provider.calls) == 5
        assert not failures and run['status'] == 'completed'
        assert run['generation_id'] == provider.coverage['generation']['run_id'] + '-changed'


def test_running_and_queued_jobs_recover_in_fifo_on_restart(tmp_path, dimension_case):
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    a, b = service.new_case(101), service.new_case(202)
    first, second = submit(hypotheses, a), submit(hypotheses, b, owner=202)
    store.mark_job_running(first['id'])
    # Construct a new store and service as a process restart would do.
    restarted = VitalyStore(store.state_dir)
    hypotheses.store = restarted
    service.store = restarted
    completed, failures = asyncio.run(drain(restarted, service, first, second))
    assert not failures and [job['id'] for job, _ in completed] == [first['id'], second['id']]
    assert persisted(hypotheses, first)['status'] == 'completed'
    assert persisted(hypotheses, second, owner=202)['status'] == 'completed'
    assert runner.max_active == 1


def test_duplicate_commands_race_accepts_exactly_one_job(tmp_path, dimension_case):
    store, service, hypotheses, _, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    def attempt(_):
        try:
            return submit(hypotheses, case)
        except StateError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = [j for j in pool.map(attempt, range(2)) if j]
    assert len(accepted) == 1
    with store._connect() as db:
        rows = db.execute("SELECT id FROM jobs WHERE case_id=? AND status IN ('queued','running')", (case['id'],)).fetchall()
    assert [r['id'] for r in rows] == [accepted[0]['id']]


def test_grill_and_existing_operations_block_hypotheses_without_losing_state(tmp_path, dimension_case):
    store, service, hypotheses, _, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    store.set_question(case['id'], {'text': 'synthetic question'}, 0, 'grilling')
    with pytest.raises(StateError, match='/stop'):
        submit(hypotheses, case)
    assert store.get_case(case['id'])['current_question']['text'] == 'synthetic question'
    store.set_question(case['id'], None, 0, 'collecting')
    first = store.begin_job(case['id'], 101, 'review')
    with pytest.raises(StateError):
        submit(hypotheses, case)
    assert store.get_job(first['id'])['status'] == 'queued'


def test_model_choice_is_user_default_and_override_survives(tmp_path):
    base = BotConfig(token='unused', state_dir=tmp_path, codex_bin=tmp_path / 'unused')
    override = BotConfig(token='unused', state_dir=tmp_path, codex_bin=tmp_path / 'unused',
                         model='gpt-6-luna', reasoning_effort='xhigh')
    assert (base.model, base.reasoning_effort) == ('gpt-6.1-sol', 'high')
    assert (override.model, override.reasoning_effort) == ('gpt-6-luna', 'xhigh')


def test_feedback_ownership_revision_and_double_tap_are_transactional(tmp_path, dimension_case):
    store, service, hypotheses, _, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    run = persisted(hypotheses, job)
    args = dict(run_id=run['id'], user_id=101, card_number=1,
                decision='selected', expected_revision=run['revision'], event_id='tap-1')
    first = hypotheses.record_feedback(**args)
    assert hypotheses.record_feedback(**args) == first
    assert first['revision'] == run['revision'] + 1
    with pytest.raises((StateError, ValueError, PermissionError)):
        hypotheses.record_feedback(**{**args, 'event_id': 'stale-tap'})
    with pytest.raises((StateError, ValueError, PermissionError)):
        hypotheses.record_feedback(**{**args, 'user_id': 202, 'event_id': 'foreign-tap',
                                      'expected_revision': first['revision']})
    updated = persisted(hypotheses, job)
    assert updated['cards'][0]['feedback']['decision'] == 'selected'
    assert updated['revision'] == first['revision']


def test_feedback_input_is_not_case_material_or_grill_and_no_learning_claim(tmp_path, dimension_case):
    store, service, hypotheses, _, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    run = persisted(hypotheses, job)
    hypotheses.begin_feedback_input(run_id=run['id'], user_id=101, card_number=1,
                                    expected_revision=run['revision'])
    result = hypotheses.submit_feedback_text(user_id=101, text='PERSONAL_CASE_FACT_SYNTHETIC',
                                             event_id='message-1')
    assert result == hypotheses.submit_feedback_text(user_id=101, text='PERSONAL_CASE_FACT_SYNTHETIC',
                                                      event_id='message-1')
    assert not store.artifacts(case['id']) and not store.turns(case['id'])
    assert 'PERSONAL_CASE_FACT_SYNTHETIC' not in json.dumps(store.get_case(case['id']), ensure_ascii=False)
    assert result['learning_status'] == 'pending_quality_gate'
    with store._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM learning_candidates').fetchone()[0] == 0


def test_feedback_event_reuse_with_different_payload_is_rejected(tmp_path, dimension_case):
    store, service, hypotheses, _, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    run = persisted(hypotheses, job)
    args = dict(run_id=run['id'], user_id=101, card_number=1,
                decision='selected', expected_revision=run['revision'], event_id='event-1')
    first = hypotheses.record_feedback(**args)
    with pytest.raises((StateError, ValueError, PermissionError)):
        hypotheses.record_feedback(**{**args, 'decision': 'not_useful',
                                      'expected_revision': first['revision']})


def test_feedback_reaction_preserves_decision_and_expired_case_rejects_input(tmp_path, dimension_case):
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    run = persisted(hypotheses, job)
    common = {'run_id': run['id'], 'user_id': 101, 'card_number': 1}
    selected = hypotheses.record_feedback(**common, decision='selected', expected_revision=run['revision'], event_id='choice')
    reaction = hypotheses.record_feedback(**common, decision='busko_reaction', expected_revision=selected['revision'], event_id='reaction')
    feedback = persisted(hypotheses, job)['cards'][0]['feedback']
    assert feedback['decision'] == 'selected' and feedback['reaction'] == 'busko_reaction'
    old = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    with store._connect() as db:
        db.execute('UPDATE cases SET last_user_activity_at=? WHERE id=?', (old, case['id']))
    with pytest.raises(StateError):
        hypotheses.begin_feedback_input(**common, expected_revision=reaction['revision'])


def test_legacy_rows_messages_and_sql_reader_survive_hypotheses_writes(tmp_path, dimension_case):
    store, service, hypotheses, _, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    store.set_progress_message(case['id'], 54321)
    # A legacy reader asks only for columns known before this migration.
    with sqlite3.connect(store.database_path) as old_reader:
        before = old_reader.execute('SELECT id, chat_id, progress_message_id FROM cases WHERE id=?', (case['id'],)).fetchone()
    job = submit(hypotheses, case)
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    reopened = VitalyStore(store.state_dir)
    with sqlite3.connect(reopened.database_path) as old_reader:
        after = old_reader.execute('SELECT id, chat_id, progress_message_id FROM cases WHERE id=?', (case['id'],)).fetchone()
        assert old_reader.execute('SELECT kind, status FROM jobs WHERE id=?', (job['id'],)).fetchone() == ('hypotheses', 'completed')
    assert after == before


def test_existing_database_migrates_without_removing_old_data(tmp_path):
    original = VitalyStore(tmp_path / 'legacy')
    case = original.create_case(101, 'old-profile')
    original.set_progress_message(case['id'], 12345)
    with original._connect() as db:
        db.execute('DROP TABLE hypotheses_feedback_input')
        db.execute('DROP TABLE hypotheses_feedback_events')
        db.execute('DROP TABLE hypotheses_runs')
    migrated = VitalyStore(original.state_dir)
    assert migrated.get_case(case['id'])['progress_message_id'] == 12345
    job = migrated.begin_hypotheses_run(case['id'], 101, 101, '')
    assert migrated.hypotheses_run(user_id=101, job_id=job['id'])['status'] == 'queued'
