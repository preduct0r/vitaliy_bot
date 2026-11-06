"""Red-first production-release contracts, using exclusively synthetic data.

The proposed small seam is ``VerifiedReleaseStore`` in hypotheses_knowledge:
install(source_dir, manifest_sha256=...), current(), load(version), bind(...),
and load_bound(binding).  It is deliberately separate from the legacy development
loader and from the case's existing general-learning knowledge package.  No real
s1 artifacts, credentials, network, model, or production state are consulted.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path

import pytest

from test_hypotheses_telegram_integration import Harness, cards, isolated_dir
from test_hypotheses_dimensions_pilot import dimension_case, topic_case
from test_hypotheses_adjudicated_pilot import adjudication_case, source_case

from tof_reliability.vitaly_bot import hypotheses_knowledge, hypotheses_ui
from tof_reliability.vitaly_bot.config import load_config


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bytes(value))


def _package(root, version='synthetic-release-1'):
    """Two meetings, including a late one; the entire inventory is fictional."""
    root.mkdir(parents=True, exist_ok=True)
    classifier = {'candidates': [{'interest_id': 'interest-early', 'source_id': 'meeting-early'},
                                 {'interest_id': 'interest-late', 'source_id': 'meeting-late'}]}
    verifier = {'verdicts': [{'interest_id': row['interest_id'], 'source_id': row['source_id'],
                              'supported': True, 'confidence': 'high'}
                             for row in classifier['candidates']]}
    files = {
        'interest-map.json': {'version': version, 'interests': [
            {'id': 'interest-early', 'source_id': 'meeting-early', 'label': 'Синтетический интерес A'},
            {'id': 'interest-late', 'source_id': 'meeting-late', 'label': 'Синтетический интерес B'}]},
        'source-index.json': {'version': version, 'sources': [
            {'id': 'meeting-early', 'date': '2030-10-01', 'sha256': 'a' * 64},
            {'id': 'meeting-late', 'date': '2030-11-01', 'sha256': 'b' * 64}]},
        'dimension-projection.json': {'version': version, 'dimensions': []},
        'classifier.json': {'answer': classifier},
        'verifier.json': {'answer': verifier},
        'classifier-trace.json': {'thread_id': 'synthetic-classifier-thread', 'answer': classifier},
        'verifier-trace.json': {'thread_id': 'synthetic-verifier-thread', 'answer': verifier},
    }
    for name, value in files.items():
        _write(root / name, value)
    manifest = {'format': 'hypotheses-knowledge-release.v1', 'version': version,
                'purpose': 'production', 'developmental_only': False,
                'source_ids': ['meeting-early', 'meeting-late'],
                'classifier': 'classifier.json', 'classifier_trace': 'classifier-trace.json',
                'verifier': 'verifier.json', 'verifier_trace': 'verifier-trace.json',
                'interest_map': 'interest-map.json', 'source_index': 'source-index.json',
                'dimension_projection': 'dimension-projection.json',
                'files': {name: _sha(root / name) for name in files}}
    _write(root / 'manifest.json', manifest)
    return root


def _rehash_manifest(root):
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['files'] = {name: _sha(root / name) for name in manifest['files']}
    _write(path, manifest)
    return _sha(path)


def _store(root):
    cls = getattr(hypotheses_knowledge, 'VerifiedReleaseStore', None)
    assert cls is not None, 'a durable verified production knowledge release store is required'
    return cls(root)


def _install(store, package):
    return store.install(package, manifest_sha256=_sha(package / 'manifest.json'))


def test_complete_release_is_portable_and_contains_late_confirmed_meeting(tmp_path):
    source = _package(tmp_path / 'synthetic-source')
    store = _store(tmp_path / 'private-releases')
    installed = _install(store, source)
    shutil.rmtree(source)
    reopened = _store(tmp_path / 'private-releases')
    knowledge = reopened.load(installed['version'])
    assert reopened.current()['version'] == 'synthetic-release-1'
    assert knowledge['knowledge_version'] == 'synthetic-release-1'
    assert {item['id'] for item in knowledge['interest_context']['interests']} == {
        'interest-early', 'interest-late'}
    assert knowledge['dimension_projection']['version'] == 'synthetic-release-1'


@pytest.mark.parametrize('damage', ['missing_version', 'missing_manifest', 'changed_content',
                                   'wrong_manifest_hash', 'omitted_late_source', 'development_only',
                                   'wrong_purpose', 'absolute_path', 'path_traversal'])
def test_invalid_release_never_replaces_previous_one(tmp_path, damage):
    store = _store(tmp_path / 'private-releases')
    first = _package(tmp_path / 'first')
    _install(store, first)
    candidate = _package(tmp_path / 'candidate', 'synthetic-release-2')
    manifest_path = candidate / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if damage == 'missing_version':
        manifest.pop('version')
    elif damage == 'missing_manifest':
        manifest_path.unlink()
    elif damage == 'changed_content':
        _write(candidate / 'interest-map.json', {'version': 'forged', 'interests': []})
    elif damage == 'omitted_late_source':
        manifest['source_ids'] = ['meeting-early']
    elif damage == 'development_only':
        manifest['developmental_only'] = True
    elif damage == 'wrong_purpose':
        manifest['purpose'] = 'development'
    elif damage == 'absolute_path':
        manifest['interest_map'] = str(candidate / 'interest-map.json')
    elif damage == 'path_traversal':
        manifest['interest_map'] = '../interest-map.json'
    if damage not in {'missing_manifest', 'changed_content', 'wrong_manifest_hash'}:
        _write(manifest_path, manifest)
    digest = ('0' * 64 if damage in {'wrong_manifest_hash', 'missing_manifest'}
              else _sha(manifest_path))
    with pytest.raises((ValueError, OSError)):
        store.install(candidate, manifest_sha256=digest)
    assert store.current()['version'] == 'synthetic-release-1'
    assert store.load('synthetic-release-1')['knowledge_version'] == 'synthetic-release-1'


@pytest.mark.parametrize('damage', ['missing_trace', 'shared_thread', 'mismatched_answer',
                                   'missing_classifier', 'missing_verifier'])
def test_independent_classifier_and_verifier_audits_are_required_even_with_new_hashes(tmp_path, damage):
    candidate = _package(tmp_path / 'candidate')
    manifest_path = candidate / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if damage == 'missing_trace':
        manifest.pop('verifier_trace')
    elif damage == 'shared_thread':
        trace = json.loads((candidate / 'verifier-trace.json').read_text(encoding='utf-8'))
        trace['thread_id'] = 'synthetic-classifier-thread'
        _write(candidate / 'verifier-trace.json', trace)
    elif damage == 'mismatched_answer':
        trace = json.loads((candidate / 'verifier-trace.json').read_text(encoding='utf-8'))
        trace['answer']['verdicts'].pop()
        _write(candidate / 'verifier-trace.json', trace)
    else:
        manifest.pop('classifier' if damage == 'missing_classifier' else 'verifier')
    _write(manifest_path, manifest)
    digest = _rehash_manifest(candidate)
    store = _store(tmp_path / 'private-releases')
    with pytest.raises((ValueError, OSError)):
        store.install(candidate, manifest_sha256=digest)
    assert store.current() is None


def test_bound_runs_are_immutable_across_install_restart_and_rollback(tmp_path):
    directory = tmp_path / 'private-releases'
    store = _store(directory)
    _install(store, _package(tmp_path / 'v1'))
    old = store.bind(profile_version='profile-old', knowledge_package_id='general-old',
                     case_id='case-old', run_id='run-old')
    assert old['knowledge_version'] == 'synthetic-release-1'
    assert re.fullmatch(r'[0-9a-f]{64}', old['knowledge_hash'])
    _install(store, _package(tmp_path / 'v2', 'synthetic-release-2'))
    new = store.bind(profile_version='profile-new', knowledge_package_id='general-new',
                     case_id='case-new', run_id='run-new')
    assert new['knowledge_version'] == 'synthetic-release-2'
    assert new['knowledge_hash'] != old['knowledge_hash']
    reopened = _store(directory)
    assert reopened.load_bound(old)['knowledge_version'] == 'synthetic-release-1'
    assert reopened.load_bound(new)['knowledge_version'] == 'synthetic-release-2'
    forged = {**old, 'knowledge_version': 'synthetic-release-2'}
    with pytest.raises(ValueError):
        reopened.load_bound(forged)
    forged = {**old, 'knowledge_hash': '0' * 64}
    with pytest.raises(ValueError):
        reopened.load_bound(forged)
    reopened.activate('synthetic-release-1')
    assert reopened.current()['version'] == 'synthetic-release-1'
    assert reopened.load_bound(new)['knowledge_version'] == 'synthetic-release-2'


def test_missing_or_modified_installed_release_fails_closed(tmp_path):
    store = _store(tmp_path / 'private-releases')
    _install(store, _package(tmp_path / 'candidate'))
    binding = store.bind(profile_version='profile', knowledge_package_id='general',
                         case_id='case', run_id='run')
    with pytest.raises((ValueError, OSError)):
        store.load('nonexistent-version')
    installed_manifest = next((tmp_path / 'private-releases').rglob('manifest.json'))
    manifest = json.loads(installed_manifest.read_text(encoding='utf-8'))
    target = installed_manifest.parent / manifest['interest_map']
    target.write_text('{"interests": []}', encoding='utf-8')
    with pytest.raises((ValueError, OSError)):
        store.load_bound(binding)


def test_production_hypotheses_is_disabled_without_explicit_private_configuration(tmp_path):
    from aiogram import Bot
    from test_hypotheses_telegram_integration import FakeSession
    from tof_reliability.vitaly_bot.telegram import TelegramRuntime

    env = {'VITALY_BOT_ENV_FILE': str(tmp_path / 'no-env'),
           'VITALY_BOT_STATE_DIR': str(tmp_path / 'state')}
    config = load_config(env, require_token=False)
    bot = Bot('123456:TEST_TOKEN', session=FakeSession())
    try:
        assert TelegramRuntime(bot, config).service.hypotheses_service is None
    finally:
        asyncio.run(bot.session.close())


@pytest.mark.parametrize('plan', [False, True])
def test_analyst_brief_is_an_actionable_human_task_not_a_stop_order(plan):
    card = cards()[0]
    card['analyst_brief'] = 'Анализ пока не выполнять'
    text = hypotheses_ui.render(card, plan=plan).casefold()
    assert 'анализ пока не выполнять' not in text
    assert 'поручение аналитику' in text
    assert 'сравнить интервалы между ремонтами' in text
    assert 'gap_missing_episode_keys' in text
    assert not re.search(r'анализ (?:выполнен|провед[её]н|заверш[её]н)', text)


def test_fake_telegram_shows_actionable_brief_without_private_meeting_text(isolated_dir):
    async def scenario():
        harness = Harness(isolated_dir)
        try:
            content = cards()
            content[0]['analyst_brief'] = 'Анализ пока не выполнять'
            await harness.completed(content=content)
            await harness.open_first_card()
            text = harness.text().casefold()
            assert 'анализ пока не выполнять' not in text
            assert 'поручение аналитику' in text
            assert 'сравнить интервалы между ремонтами' in text
            assert 'private_meeting_quote_do_not_show' not in text
        finally:
            await harness.close()
    asyncio.run(scenario())


def test_production_run_reloads_its_admitted_release_after_activation(tmp_path, dimension_case):
    from test_hypotheses_service_runtime import make_runtime, submit, drain
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    context, _, projection, _ = copy.deepcopy(dimension_case)
    class Versioned:
        def __init__(self):
            self.version = 'release-one'
        def load(self, profile_version, knowledge_package_id):
            return self.load_version(self.version, profile_version, knowledge_package_id)
        def load_version(self, version, profile_version, knowledge_package_id):
            return {'knowledge_version': version, 'interest_context': copy.deepcopy(context),
                    'dimension_projection': copy.deepcopy(projection)}
    loader = Versioned()
    hypotheses.knowledge_loader = loader
    case = service.new_case(101)
    job = submit(hypotheses, case)
    loader.version = 'release-two'
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    assert hypotheses.get_run(user_id=101, job_id=job['id'])['knowledge_version'] == 'release-one'


def test_production_and_development_settings_cannot_be_enabled_together(tmp_path):
    from tof_reliability.vitaly_bot.models import InputError
    env = {'VITALY_BOT_ENV_FILE': str(tmp_path / 'no-env'),
           'VITALY_BOT_HYPOTHESES_DEVELOPMENT': '1',
           'VITALY_BOT_HYPOTHESES_RELEASE_DIR': str(tmp_path / 'releases'),
           'VITALY_BOT_HYPOTHESES_PRODUCTION': '1',
           'VITALY_BOT_HYPOTHESES_CORPUS_DIR': 'x',
           'VITALY_BOT_HYPOTHESES_ORIGINAL_DIR': 'x',
           'VITALY_BOT_HYPOTHESES_ADJUDICATION_DIR': 'x',
           'VITALY_BOT_HYPOTHESES_MANIFEST_SHA256': 'a' * 64}
    with pytest.raises(InputError):
        load_config(env, require_token=False)


def test_new_job_on_old_case_uses_published_general_rules_and_pins_them(tmp_path, dimension_case):
    from test_hypotheses_service_runtime import make_runtime, submit, drain
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    original_active = store.active_knowledge_package
    original_get = store.get_knowledge_package
    old = original_active()
    newer = copy.deepcopy(old)
    newer['id'] = 'synthetic-published-rules'
    newer['package']['analysis_rules'] = [{'transfer_key': 'separate_fact_from_scenario'}]
    store.active_knowledge_package = lambda: newer
    store.get_knowledge_package = lambda ident: newer if ident == newer['id'] else original_get(ident)
    job = submit(hypotheses, case)
    store.active_knowledge_package = original_active
    _, failures = asyncio.run(drain(store, service, job))
    assert not failures
    prompt = next(prompt for prompt, schema in runner.calls if 'cards' in schema['properties'])
    assert 'separate_fact_from_scenario' in prompt
    assert store.get_case(case['id'])['knowledge_package_id'] == old['id']


def test_release_rejects_unverified_classifier_candidate(tmp_path):
    candidate = _package(tmp_path / 'candidate')
    verdict = json.loads((candidate / 'verifier.json').read_text(encoding='utf-8'))
    verdict['answer']['verdicts'].pop()
    _write(candidate / 'verifier.json', verdict)
    trace = json.loads((candidate / 'verifier-trace.json').read_text(encoding='utf-8'))
    trace['answer'] = verdict['answer']
    _write(candidate / 'verifier-trace.json', trace)
    digest = _rehash_manifest(candidate)
    with pytest.raises((ValueError, OSError)):
        _store(tmp_path / 'releases').install(candidate, manifest_sha256=digest)


def test_synthetic_release_runs_through_configured_telegram_service(tmp_path, dimension_case):
    from aiogram import Bot
    from test_hypotheses_telegram_integration import FakeSession
    from test_hypotheses_service_runtime import FakeCoverage, FakeRunner, drain
    from tof_reliability.vitaly_bot.telegram import TelegramRuntime
    context, coverage, projection, cards_data = copy.deepcopy(dimension_case)
    version = context['version']
    package = tmp_path / 'source'
    package.mkdir()
    sources = {}
    for item in context['interests']:
        for evidence in item['evidence']:
            sources[evidence['source_id']] = {'id': evidence['source_id'], 'date': evidence['date'], 'sha256': evidence['sha256']}
    candidates = [{'interest_id': row['interest_ids'][0], 'source_id': row['source_ids'][0],
                   'dimension_id': row['dimension_id'], 'locator': row['evidence'][0]['locator']}
                  for row in projection['dimensions']]
    classifier = {'candidates': candidates}
    verifier = {'verdicts': [{**row, 'supported': True, 'confidence': 'high'} for row in candidates]}
    data = {'interest-map.json': context, 'source-index.json': {'version': version, 'sources': list(sources.values())},
            'dimension-projection.json': projection, 'classifier.json': {'answer': classifier},
            'verifier.json': {'answer': verifier}, 'classifier-trace.json': {'thread_id': 'classifier', 'answer': classifier},
            'verifier-trace.json': {'thread_id': 'verifier', 'answer': verifier}}
    for name, value in data.items():
        _write(package / name, value)
    manifest = {'format': 'hypotheses-knowledge-release.v1', 'version': version, 'purpose': 'production',
                'developmental_only': False, 'source_ids': list(sources),
                'classifier': 'classifier.json', 'verifier': 'verifier.json',
                'classifier_trace': 'classifier-trace.json', 'verifier_trace': 'verifier-trace.json',
                'interest_map': 'interest-map.json', 'source_index': 'source-index.json',
                'dimension_projection': 'dimension-projection.json', 'files': {name: _sha(package / name) for name in data}}
    _write(package / 'manifest.json', manifest)
    releases = _store(tmp_path / 'private-releases')
    _install(releases, package)
    env = {'VITALY_BOT_ENV_FILE': str(tmp_path / 'no-env'), 'VITALY_BOT_STATE_DIR': str(tmp_path / 'state'),
           'VITALY_BOT_HYPOTHESES_PRODUCTION': '1', 'VITALY_BOT_HYPOTHESES_RELEASE_DIR': str(tmp_path / 'private-releases'),
           'VITALY_BOT_HYPOTHESES_RELEASE_SHA256': _sha(package / 'manifest.json')}
    config = load_config(env, require_token=False)
    bot = Bot('123456:TEST_TOKEN', session=FakeSession())
    try:
        runtime = TelegramRuntime(bot, config)
        runner = FakeRunner(cards_data)
        runtime.service.runner = runner
        hypotheses = runtime.service.hypotheses_service
        hypotheses.runner = runner
        hypotheses.coverage_provider = FakeCoverage(coverage)
        case = runtime.service.new_case(101)
        job = hypotheses.submit(case_id=case['id'], chat_id=101, user_id=101)
        _, failures = asyncio.run(drain(runtime.store, runtime.service, job))
        assert not failures
        assert hypotheses.get_run(user_id=101, job_id=job['id'])['knowledge_version'] == version
    finally:
        asyncio.run(bot.session.close())


def test_release_rejects_dimension_without_independent_supported_verdict(tmp_path):
    candidate = _package(tmp_path / 'candidate')
    _write(candidate / 'dimension-projection.json', {'version': 'synthetic-release-1', 'dimensions': [
        {'dimension_id': 'invented', 'interest_ids': ['interest-late'], 'source_ids': ['meeting-late'],
         'evidence': [{'interest_id': 'interest-late', 'source_id': 'meeting-late', 'locator': 'chars:1-2'}]}]})
    digest = _rehash_manifest(candidate)
    with pytest.raises((ValueError, OSError)):
        _store(tmp_path / 'releases').install(candidate, manifest_sha256=digest)


def test_production_config_rejects_release_directory_inside_git(tmp_path):
    from tof_reliability.vitaly_bot.models import InputError
    repo = tmp_path / 'repository'
    (repo / '.git').mkdir(parents=True)
    env = {'VITALY_BOT_ENV_FILE': str(tmp_path / 'no-env'),
           'VITALY_BOT_HYPOTHESES_PRODUCTION': '1',
           'VITALY_BOT_HYPOTHESES_RELEASE_DIR': str(repo / 'private'),
           'VITALY_BOT_HYPOTHESES_RELEASE_SHA256': 'a' * 64}
    with pytest.raises(InputError):
        load_config(env, require_token=False)


def test_installer_rejects_destination_inside_git(tmp_path):
    repo = tmp_path / 'repository'
    (repo / '.git').mkdir(parents=True)
    package = _package(tmp_path / 'source')
    with pytest.raises(ValueError):
        _install(_store(repo / 'private'), package)
