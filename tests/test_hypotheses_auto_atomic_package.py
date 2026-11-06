"""Red-first contract for private, evidence-complete atomic release promotion.

Proposed test seam: hypotheses_auto_atomic_package.AtomicReleaseManager wraps
VerifiedReleaseStore. promote() receives an s1 audit path, never a score; the
component must authenticate and bind the immutable s1 assessment to the exact
baseline/candidate payload and source evidence before advancing the pointer.
The interface may be adapted during implementation while keeping these
observable properties. All source texts and model doubles are synthetic.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from test_hypotheses_dimensions_pilot import dimension_case, topic_case
from test_hypotheses_scenario_comparison import lab as scenario_lab, evaluate
from test_hypotheses_scenario_comparison import BASE, CANDIDATE
from tof_reliability.vitaly_bot.hypotheses_knowledge import VerifiedReleaseStore


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')), encoding='utf-8')


def release_package(directory, context, projection, sources, version):
    """An actual v1 release with all seven mandatory, distinct role files."""
    directory.mkdir(parents=True)
    context = copy.deepcopy(context)
    context['version'] = version
    rows = []
    for dimension in projection['dimensions']:
        for evidence in dimension['evidence']:
            rows.append({key: evidence[key] for key in ('interest_id', 'source_id', 'locator')}
                        | {'dimension_id': dimension['dimension_id']})
    classifier = {'candidates': rows}
    verifier = {'verdicts': [{**r, 'supported': True, 'confidence': 'high',
                               'conflicting': False} for r in rows]}
    files = {'interest-map.json': context,
             'source-index.json': {'version': version, 'sources': sources},
             'dimension-projection.json': projection,
             'classifier.json': {'answer': classifier},
             'verifier.json': {'answer': verifier},
             'classifier-trace.json': {'thread_id': 'synthetic-baseline-classifier', 'answer': classifier},
             'verifier-trace.json': {'thread_id': 'synthetic-baseline-verifier', 'answer': verifier}}
    for name, value in files.items():
        write(directory / name, value)
    roles = {'interest_map': 'interest-map.json', 'source_index': 'source-index.json',
             'dimension_projection': 'dimension-projection.json', 'classifier': 'classifier.json',
             'verifier': 'verifier.json', 'classifier_trace': 'classifier-trace.json',
             'verifier_trace': 'verifier-trace.json'}
    manifest = {'format': 'hypotheses-knowledge-release.v1', 'purpose': 'production',
                'developmental_only': False, 'version': version,
                'source_ids': [s['id'] for s in sources], **roles,
                'files': {name: sha(directory / name) for name in files}}
    write(directory / 'manifest.json', manifest)
    return directory


def manager(store, trusted):
    from tof_reliability.vitaly_bot import hypotheses_auto_atomic_package as component
    config = store.root.parent / 'authority.key'
    return component.AtomicReleaseManager(release_store=store, trusted_manifest_sha256=trusted,
        authority_path=config, authority_fingerprint=sha(config), state_path=store.root.parent / 'trusted.sqlite',
        inventory_path=store.root.parent / 'inventory.json')


@pytest.fixture
def package_lab(scenario_lab, tmp_path):
    # The store refuses private releases inside a Git worktree. The workflow
    # supplies a per-slice isolated directory; no TOF data or network is used.
    base = Path(os.environ.get('WORKFLOW_OUTPUT_DIR', tempfile.gettempdir()))
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='atomic-package-', dir=base) as name:
        root = Path(name)
        key = root / 'authority.key'
        key.write_bytes(os.urandom(32))
        key.chmod(0o600)
        from tof_reliability.vitaly_bot.hypotheses_scenario_comparison import AssessmentAuthority
        authority = AssessmentAuthority(key, sha(key))
        context = copy.deepcopy(scenario_lab['context'])
        context['version'] = BASE
        projection = copy.deepcopy(scenario_lab['projection'])
        historical = []
        for item in context['interests']:
            evidence = item['evidence'][0]
            path = tmp_path / (evidence['source_id'] + '.txt')
            assert sha(path) == evidence['sha256']
            historical.append({'id': evidence['source_id'], 'date': evidence['date'],
                               'kind': 'transcript', 'status': 'accepted', 'path': str(path),
                               'original_path': str(path), 'sha256': sha(path),
                               'original_sha256': sha(path)})
        source = release_package(root / 'baseline-input', context, projection, historical, BASE)
        store = VerifiedReleaseStore(root / 'releases')
        anchor = sha(source / 'manifest.json')
        baseline = store.install(source, manifest_sha256=anchor)
        meetings = []
        for n in (1, 2):
            text = f'Please compare repeated pump repair intervals. Synthetic meeting {n}.'
            path = root / f'meeting-{n}.txt'
            path.write_text(text, encoding='utf-8')
            meetings.append({'id': f'new-meeting-{n}', 'meeting_id': f'meeting-{n}',
                             'date': f'2030-11-0{n}', 'kind': 'meeting', 'status': 'accepted',
                             'path': str(path), 'original_path': str(path),
                             'sha256': sha(path), 'original_sha256': sha(path)})
        evidence = [{'source_id': m['id'], 'locator': 'chars:0-46',
                     'attribution': 'direct_request'} for m in meetings]
        proposal = {'id': 'interval-pattern', 'requirement': 'Compare repeated pump repair intervals',
                    'scope': 'broad', 'evidence': evidence}
        candidate_context = copy.deepcopy(context)
        candidate_context['version'] = CANDIDATE
        candidate_interest = copy.deepcopy(proposal)
        candidate_interest['attributions'] = ['direct_request']
        for entry, meeting in zip(candidate_interest['evidence'], meetings):
            entry.update(date=meeting['date'], sha256=meeting['sha256'],
                         reference=f"{entry['source_id']}:{entry['locator']}")
        candidate_context['interests'].append(candidate_interest)
        baseline_payload = {'knowledge_version': BASE, 'interest_context': context,
                            'dimension_projection': projection}
        candidate_payload = {'knowledge_version': CANDIDATE, 'interest_context': candidate_context,
                             'dimension_projection': copy.deepcopy(projection)}
        classifier = {'candidates': [{**e, 'interest_id': proposal['id'],
                                      'dimension_id': 'repair_repeat_interval'} for e in evidence]}
        verifier = {'verdicts': [{**r, 'supported': True, 'confidence': 'high', 'conflicting': False}
                                 for r in classifier['candidates']]}
        inventory = {'complete': True, 'baseline_version': BASE,
                     'historical_source_ids': [s['id'] for s in historical],
                     'historical_sources': historical, 'meetings': meetings}
        write(root / 'inventory.json', inventory)
        (root / 'inventory.json').chmod(0o600)
        # Run s1 itself: no fabricated assessment, score, boolean or model verdict.
        assessment = evaluate(scenario_lab, baseline=baseline_payload, candidate=candidate_payload, authority=authority)
        assert assessment['activation_allowed'] is True
        yield {'root': root, 'store': store, 'anchor': anchor, 'baseline': baseline,
               'inventory': inventory, 'candidate': candidate_payload, 'proposal': proposal,
               'classifier': {'answer': classifier}, 'verifier': {'answer': verifier},
               'classifier_trace': {'thread_id': 'synthetic-classifier', 'answer': classifier},
               'verifier_trace': {'thread_id': 'synthetic-verifier', 'answer': verifier},
               'assessment': Path(assessment['audit_path']), 'scenario': scenario_lab,
               'baseline_payload': baseline_payload, 'authority': authority}


def promote(lab, **changes):
    kwargs = {name: copy.deepcopy(lab[name]) for name in
              ('inventory', 'candidate', 'proposal', 'classifier', 'verifier',
               'classifier_trace', 'verifier_trace')}
    kwargs['assessment_path'] = lab['assessment']
    kwargs.update(changes)
    return manager(lab['store'], lab['anchor']).promote(**kwargs)


def bound(store, run):
    return store.bind(profile_version='profile', knowledge_package_id='general',
                      case_id='synthetic-case', run_id=run)


def assert_baseline(lab):
    assert lab['store'].current() == lab['baseline']
    assert lab['store'].load(BASE)['knowledge_version'] == BASE


def reject(lab, **changes):
    try:
        result = promote(lab, **changes)
    except (ValueError, OSError, RuntimeError):
        pass
    else:
        assert not result or result.get('published') is not True
    assert_baseline(lab)


def test_source_gate_checks_originals_and_preserves_inventory(package_lab):
    from tof_reliability.vitaly_bot.hypotheses_auto_atomic_package import validate_source_evidence
    lab = package_lab
    result = validate_source_evidence(
        release_store=lab['store'], inventory=lab['inventory'],
        candidate=lab['candidate'], proposal=lab['proposal'],
        classifier=lab['classifier'], verifier=lab['verifier'],
        classifier_trace=lab['classifier_trace'], verifier_trace=lab['verifier_trace'])
    assert set(result) == {row['id'] for row in lab['inventory']['historical_sources']
                           + lab['inventory']['meetings']}
    assert all(result[row['id']] == Path(row['original_path']).read_bytes()
               for row in lab['inventory']['historical_sources'] + lab['inventory']['meetings'])


def test_source_gate_rejects_conflicting_trace_and_changed_original(package_lab):
    from tof_reliability.vitaly_bot.hypotheses_auto_atomic_package import validate_source_evidence
    lab = package_lab
    kwargs = {name: copy.deepcopy(lab[name]) for name in
              ('inventory', 'candidate', 'proposal', 'classifier', 'verifier',
               'classifier_trace', 'verifier_trace')}
    kwargs['release_store'] = lab['store']
    kwargs['verifier']['answer']['verdicts'][0]['conflicting'] = True
    kwargs['verifier_trace']['answer'] = copy.deepcopy(kwargs['verifier']['answer'])
    with pytest.raises(ValueError):
        validate_source_evidence(**kwargs)
    kwargs['verifier'] = lab['verifier']
    kwargs['verifier_trace'] = lab['verifier_trace']
    Path(kwargs['inventory']['historical_sources'][0]['original_path']).write_text('forged')
    with pytest.raises(ValueError):
        validate_source_evidence(**kwargs)


def test_source_gate_rejects_changed_projection_and_evidence_hash(package_lab):
    from tof_reliability.vitaly_bot.hypotheses_auto_atomic_package import validate_source_evidence
    lab = package_lab
    kwargs = {name: copy.deepcopy(lab[name]) for name in
              ('inventory', 'candidate', 'proposal', 'classifier', 'verifier',
               'classifier_trace', 'verifier_trace')}
    kwargs['release_store'] = lab['store']
    kwargs['candidate']['dimension_projection']['dimensions'][0]['description'] += ' forged'
    with pytest.raises(ValueError):
        validate_source_evidence(**kwargs)
    kwargs['candidate'] = copy.deepcopy(lab['candidate'])
    kwargs['candidate']['interest_context']['interests'][-1]['evidence'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError):
        validate_source_evidence(**kwargs)


def test_real_package_is_evidence_complete_and_qualified_assessment_advances(package_lab):
    lab = package_lab
    old = bound(lab['store'], 'queued-before')
    result = promote(lab)
    assert result['published'] is True
    active = manager(lab['store'], lab['anchor']).current()
    assert active['version'] == CANDIDATE
    new = bound(lab['store'], 'queued-after')
    assert lab['store'].load_bound(old)['knowledge_version'] == BASE
    assert lab['store'].load_bound(new)['knowledge_version'] == CANDIDATE
    directory = lab['store'].root / 'releases' / CANDIDATE
    manifest, values = VerifiedReleaseStore._validate(directory, active['manifest_sha256'])
    sources = {s['id']: s for s in values['source_index']['sources']}
    originals = lab['inventory']['historical_sources'] + lab['inventory']['meetings']
    assert set(sources) == {s['id'] for s in originals} == set(manifest['source_ids'])
    assert {k: v['sha256'] for k, v in sources.items()} == {s['id']: s['sha256'] for s in originals}
    item = next(i for i in values['interest_map']['interests'] if i['id'] == 'interval-pattern')
    assert item['evidence'] == lab['proposal']['evidence']
    assert values['classifier_trace']['thread_id'] != values['verifier_trace']['thread_id']
    assert values['classifier_trace']['answer'] == values['classifier']['answer']
    assert values['verifier_trace']['answer'] == values['verifier']['answer']
    # Every included payload must be retained byte-for-byte and covered by the
    # manifest, including the actual s1 artifact and captured independent traces.
    files = [directory / name for name in manifest['files']]
    assert lab['assessment'].read_bytes() in [p.read_bytes() for p in files]
    for source in originals:
        assert Path(source['original_path']).read_bytes() in [p.read_bytes() for p in files]


def test_no_improvement_and_arbitrary_caller_approval_cannot_advance(package_lab):
    lab = package_lab
    lab['scenario']['mode']['value'] = 'tie'
    outcome = evaluate(lab['scenario'], baseline=lab['baseline_payload'], candidate=lab['candidate'], authority=lab['authority'])
    assert outcome['activation_allowed'] is False
    reject(lab, assessment_path=Path(outcome['audit_path']))
    fake = lab['root'] / 'caller-assessment.json'
    write(fake, {'activation_allowed': True, 'score': 999, 'improved': True})
    reject(lab, assessment_path=fake)
    reject(lab, assessment_path=True)


@pytest.mark.parametrize('damage', ['assessment_modified', 'wrong_candidate', 'wrong_baseline',
                                    'same_thread', 'trace_answer', 'verifier_conflict',
                                    'verifier_low', 'source_changed', 'original_missing',
                                    'inventory_omitted', 'meeting_unconfirmed', 'one_meeting',
                                    'same_meeting', 'bad_locator', 'bad_attribution'])
def test_unverifiable_evidence_or_forgery_never_replaces_current(package_lab, damage):
    lab = package_lab
    candidate = copy.deepcopy(lab['candidate'])
    inventory = copy.deepcopy(lab['inventory'])
    proposal = copy.deepcopy(lab['proposal'])
    verifier = copy.deepcopy(lab['verifier'])
    trace = copy.deepcopy(lab['verifier_trace'])
    assessment = lab['assessment']
    if damage == 'assessment_modified':
        data = json.loads(assessment.read_text(encoding='utf-8'))
        data['scenarios'][0]['candidate']['output'] = ['forged']
        write(assessment, data)
    elif damage == 'wrong_candidate':
        candidate['interest_context']['interests'][-1]['requirement'] += ' forged'
    elif damage == 'wrong_baseline':
        inventory['baseline_version'] = 'some-other-release'
    elif damage == 'same_thread':
        trace['thread_id'] = lab['classifier_trace']['thread_id']
    elif damage == 'trace_answer':
        trace['answer'] = {'verdicts': []}
    elif damage in ('verifier_conflict', 'verifier_low'):
        verdict = verifier['answer']['verdicts'][0]
        verdict.update({'conflicting': True} if damage == 'verifier_conflict' else {'confidence': 'low'})
        trace['answer'] = copy.deepcopy(verifier['answer'])
    elif damage == 'source_changed':
        Path(inventory['meetings'][0]['path']).write_text('different source', encoding='utf-8')
    elif damage == 'original_missing':
        inventory['historical_sources'][0]['original_path'] = str(lab['root'] / 'missing-original')
    elif damage == 'inventory_omitted':
        inventory['historical_sources'].pop()
    elif damage == 'meeting_unconfirmed':
        inventory['meetings'][0]['status'] = 'unconfirmed'
    elif damage == 'one_meeting':
        inventory['meetings'].pop()
        proposal['evidence'].pop()
    elif damage == 'same_meeting':
        inventory['meetings'][1]['meeting_id'] = inventory['meetings'][0]['meeting_id']
    elif damage == 'bad_locator':
        proposal['evidence'][0]['locator'] = 'chars:0-999999'
    else:
        proposal['evidence'][0]['attribution'] = 'fabricated'
    reject(lab, candidate=candidate, inventory=inventory, proposal=proposal,
           verifier=verifier, verifier_trace=trace, assessment_path=assessment)


def test_restart_rollback_and_old_and_new_pins(package_lab):
    lab = package_lab
    old = bound(lab['store'], 'old')
    promote(lab)
    new = bound(lab['store'], 'new')
    # A fresh interpreter must use the original configured anchor, without
    # rewriting configuration to the new candidate's manifest hash.
    script = ('import sys; from tof_reliability.vitaly_bot.hypotheses_knowledge import VerifiedReleaseStore; '
              'from tof_reliability.vitaly_bot.hypotheses_auto_atomic_package import AtomicReleaseManager; '
              'm=AtomicReleaseManager(release_store=VerifiedReleaseStore(sys.argv[1]), '
              'trusted_manifest_sha256=sys.argv[2], authority_path=sys.argv[3], authority_fingerprint=sys.argv[4], '
              'state_path=sys.argv[5]); print(m.current()["version"])')
    result = subprocess.run([sys.executable, '-c', script, str(lab['store'].root), lab['anchor'], str(lab['root'] / 'authority.key'),
                             sha(lab['root'] / 'authority.key'), str(lab['root'] / 'trusted.sqlite')],
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == CANDIDATE
    reopened = manager(VerifiedReleaseStore(lab['store'].root), lab['anchor'])
    reopened.rollback(BASE)
    assert reopened.current()['version'] == BASE
    assert reopened.release_store.load_bound(old)['knowledge_version'] == BASE
    assert reopened.release_store.load_bound(new)['knowledge_version'] == CANDIDATE
    assert bound(reopened.release_store, 'after-rollback')['knowledge_version'] == BASE


def test_version_rewrite_and_rehashed_pointer_tamper_fail_closed(package_lab):
    lab = package_lab
    promote(lab)
    active = lab['store'].current()
    release = lab['store'].root / 'releases' / CANDIDATE
    interest = release / 'interest-map.json'
    data = json.loads(interest.read_text(encoding='utf-8'))
    data['interests'][-1]['requirement'] = 'forged rewritten content'
    write(interest, data)
    manifest_path = release / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['files']['interest-map.json'] = sha(interest)
    write(manifest_path, manifest)
    new_hash = sha(manifest_path)
    # An attacker rehashes the old mutable record and pointer too. The trusted
    # initial anchor and authenticated successor history must still reject it.
    write(lab['store'].root / 'records' / (CANDIDATE + '.json'), {'manifest_sha256': new_hash})
    write(lab['store'].root / 'current.json', {'version': CANDIDATE, 'manifest_sha256': new_hash})
    assert new_hash != active['manifest_sha256']
    with pytest.raises((ValueError, OSError)):
        manager(VerifiedReleaseStore(lab['store'].root), lab['anchor']).current()
    assert lab['store'].load(BASE)['knowledge_version'] == BASE


@pytest.mark.parametrize('where', ['staging', 'pointer'])
def test_interrupted_atomic_write_preserves_old_pointer_across_restart(package_lab, monkeypatch, where):
    lab = package_lab
    original_replace = os.replace
    def interrupted(src, dst, *args, **kwargs):
        path = Path(dst)
        if (where == 'pointer' and path.name == 'current.json') or (
                where == 'staging' and path.name != 'current.json'):
            raise OSError('synthetic interruption or disk full')
        return original_replace(src, dst, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'replace', interrupted)
        try:
            promote(lab)
        except (ValueError, OSError, RuntimeError):
            pass
    reopened = manager(VerifiedReleaseStore(lab['store'].root), lab['anchor'])
    assert reopened.current() == lab['baseline']
    assert reopened.release_store.load(BASE)['knowledge_version'] == BASE


def test_wrong_authority_fingerprint_and_unsigned_assessment_rejected(package_lab):
    from tof_reliability.vitaly_bot.hypotheses_auto_atomic_package import AtomicReleaseManager
    lab = package_lab
    with pytest.raises(ValueError):
        AtomicReleaseManager(release_store=lab['store'], trusted_manifest_sha256=lab['anchor'],
            authority_path=lab['root'] / 'authority.key', authority_fingerprint='0' * 64,
            state_path=lab['root'] / 'trusted.sqlite')
    unsigned = evaluate(lab['scenario'], baseline=lab['baseline_payload'], candidate=lab['candidate'])
    reject(lab, assessment_path=Path(unsigned['audit_path']))


def test_record_write_interruption_can_resume_same_release(package_lab, monkeypatch):
    lab = package_lab
    original = lab['store']._atomic_json
    failed = False
    def interrupted(path, value):
        nonlocal failed
        if Path(path).parent.name == 'records' and not failed:
            failed = True
            raise OSError('interrupted after release rename')
        return original(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(lab['store'], '_atomic_json', interrupted)
        with pytest.raises(OSError):
            promote(lab)
    assert manager(lab['store'], lab['anchor']).current() == lab['baseline']
    assert promote(lab)['published'] is True


def test_truncated_successor_chain_after_rollback_fails_closed(package_lab):
    import sqlite3
    lab = package_lab
    promote(lab)
    manager(lab['store'], lab['anchor']).rollback(BASE)
    with sqlite3.connect(lab['root'] / 'trusted.sqlite') as db:
        db.execute('DELETE FROM chain WHERE seq > 1')
    with pytest.raises(ValueError):
        manager(lab['store'], lab['anchor']).current()


def test_unverified_new_dimension_evidence_rejected(package_lab):
    lab = package_lab
    candidate = copy.deepcopy(lab['candidate'])
    dimension = candidate['dimension_projection']['dimensions'][0]
    dimension['evidence'].append({'interest_id': lab['proposal']['id'],
        'source_id': lab['inventory']['meetings'][0]['id'], 'locator': 'chars:0-1'})
    from tof_reliability.vitaly_bot.hypotheses_auto_atomic_package import validate_source_evidence
    kwargs = {name: copy.deepcopy(lab[name]) for name in ('inventory', 'proposal', 'classifier', 'verifier', 'classifier_trace', 'verifier_trace')}
    with pytest.raises(ValueError):
        validate_source_evidence(release_store=lab['store'], candidate=candidate, **kwargs)


def test_pointer_interruption_can_resume_same_release(package_lab, monkeypatch):
    lab = package_lab
    original = lab['store']._atomic_json
    failed = False
    def interrupted(path, value):
        nonlocal failed
        if Path(path).name == 'current.json' and not failed:
            failed = True
            raise OSError('interrupted before pointer switch')
        return original(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(lab['store'], '_atomic_json', interrupted)
        with pytest.raises(OSError):
            promote(lab)
    assert manager(lab['store'], lab['anchor']).current() == lab['baseline']
    assert promote(lab)['published'] is True


def test_inventory_must_match_independent_configured_inventory(package_lab):
    lab = package_lab
    extra = copy.deepcopy(lab['inventory']['meetings'][0])
    path = lab['root'] / 'unlisted-meeting.txt'
    path.write_text('Synthetic unlisted meeting', encoding='utf-8')
    extra.update(id='unlisted', meeting_id='unlisted', path=str(path), original_path=str(path),
                 sha256=sha(path), original_sha256=sha(path))
    claimed = copy.deepcopy(lab['inventory'])
    claimed['meetings'].append(extra)
    reject(lab, inventory=claimed)


def test_signed_but_inconsistent_assessment_is_not_promoted(package_lab):
    lab = package_lab
    audit = json.loads(lab['assessment'].read_text(encoding='utf-8'))
    audit['scenarios'][0]['candidate']['observations'] = {'invented': True}
    audit.pop('authority_hmac_sha256')
    audit['authority_hmac_sha256'] = lab['authority']._mac(audit)
    changed = lab['root'] / 'inconsistent-assessment.json'
    write(changed, audit)
    reject(lab, assessment_path=changed)


def test_lost_trusted_state_cannot_silently_reinitialize_after_rollback(package_lab):
    lab = package_lab
    promote(lab)
    manager(lab['store'], lab['anchor']).rollback(BASE)
    (lab['root'] / 'trusted.sqlite').unlink()
    with pytest.raises((ValueError, OSError)):
        manager(lab['store'], lab['anchor']).current()


def test_manager_bound_jobs_revalidate_the_trusted_chain(package_lab):
    lab = package_lab
    managed = manager(lab['store'], lab['anchor'])
    old = managed.bind(profile_version='profile', knowledge_package_id='general', case_id='case', run_id='old')
    promote(lab)
    new = managed.bind(profile_version='profile', knowledge_package_id='general', case_id='case', run_id='new')
    assert managed.load_bound(old)['knowledge_version'] == BASE
    assert managed.load_bound(new)['knowledge_version'] == CANDIDATE
    (lab['store'].root / 'releases' / CANDIDATE / 'interest-map.json').write_text('{}')
    with pytest.raises((ValueError, OSError)):
        managed.bind(profile_version='profile', knowledge_package_id='general', case_id='case', run_id='forged')
    with pytest.raises((ValueError, OSError)):
        managed.load_bound(new)
