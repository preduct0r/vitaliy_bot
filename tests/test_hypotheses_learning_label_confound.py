"""A release label alone must not constitute measured learning."""

import copy

import json

import pytest

from test_hypotheses_scenario_comparison import evaluate, lab, dimension_case, topic_case
from test_hypotheses_scenario_comparison import BASE, CANDIDATE
from test_hypotheses_scenario_comparison import lab as scenario_lab
from test_hypotheses_auto_atomic_package import package_lab, promote


def test_same_knowledge_with_different_version_labels_cannot_promote(lab):
    baseline = {'knowledge_version': BASE, 'interest_context': copy.deepcopy(lab['context']),
                'dimension_projection': copy.deepcopy(lab['projection'])}
    candidate = copy.deepcopy(baseline)
    candidate['knowledge_version'] = CANDIDATE
    def label_generator(request):
        result = lab['generator'](request)
        if request.get('knowledge_version') == CANDIDATE:
            row = next(c for c in result if c['meeting_interest']['dimension_id'] == 'repair_repeat_interval')
            row['metrics_method'] = 'Сопоставить интервалы между повторными ремонтными эпизодами оборудования по периодам'
        return result
    assessment = evaluate(lab, baseline=baseline, candidate=candidate, generator=label_generator)
    assert assessment['activation_allowed'] is False


def test_unbound_interest_and_label_only_improvement_cannot_publish(package_lab):
    candidate = copy.deepcopy(package_lab['candidate'])
    candidate['dimension_projection'] = copy.deepcopy(package_lab['baseline_payload']['dimension_projection'])
    try:
        result = promote(package_lab, candidate=candidate)
    except (ValueError, OSError, RuntimeError):
        result = None
    assert not result or result.get('published') is not True
    assert package_lab['store'].current() == package_lab['baseline']


def test_generator_never_receives_the_release_label(lab):
    seen = []
    def generator(request):
        seen.append(copy.deepcopy(request))
        return lab['generator'](request)
    evaluate(lab, generator=generator)
    assert seen
    assert all(BASE not in str(request) and CANDIDATE not in str(request) for request in seen)


def test_unrelated_new_binding_cannot_count_as_target_improvement(lab):
    candidate = {'knowledge_version': CANDIDATE,
                 'interest_context': copy.deepcopy(lab['context']),
                 'dimension_projection': copy.deepcopy(lab['projection'])}
    row = next(d for d in candidate['dimension_projection']['dimensions']
               if d['dimension_id'] == 'repair_downtime')
    old_interest = next(i for i in candidate['interest_context']['interests']
                        if i['id'] == row['interest_ids'][0])
    interest = copy.deepcopy(old_interest)
    interest['id'] = 'unrelated-new-interest'
    candidate['interest_context']['interests'].append(interest)
    row['interest_ids'].append(interest['id'])
    row['evidence'].append({'interest_id': interest['id'], **interest['evidence'][0]})
    def generator(request):
        result = lab['generator'](request)
        if any('unrelated-new-interest' in d['interest_ids'] for d in request['dimensions']):
            target = next(c for c in result if c['meeting_interest']['dimension_id'] == 'repair_repeat_interval')
            target['metrics_method'] = 'Сопоставить интервалы между повторными ремонтными эпизодами оборудования по периодам'
        return result
    audit = evaluate(lab, candidate=candidate, generator=generator)
    assert any(r['baseline']['output'] != r['candidate']['output'] for r in audit['scenarios'])
    assert audit['activation_allowed'] is False


def test_source_gate_rejects_verified_but_unbound_interest(package_lab):
    from tof_reliability.vitaly_bot.hypotheses_auto_atomic_package import validate_source_evidence
    fields = ('inventory', 'candidate', 'proposal', 'classifier', 'verifier',
              'classifier_trace', 'verifier_trace')
    kwargs = {name: copy.deepcopy(package_lab[name]) for name in fields}
    kwargs['candidate']['dimension_projection'] = copy.deepcopy(
        package_lab['baseline_payload']['dimension_projection'])
    with pytest.raises(ValueError):
        validate_source_evidence(release_store=package_lab['store'], **kwargs)


def test_signed_assessment_cannot_detach_cards_from_actual_output(package_lab):
    from tof_reliability.vitaly_bot.hypotheses_scenario_comparison import _digest
    lab = package_lab
    audit = json.loads(lab['assessment'].read_text(encoding='utf-8'))
    record = next(r for r in audit['scenarios'] if r['comparison']['improved'])
    row = next(c for c in record['candidate']['output']
               if c['meeting_interest']['dimension_id'] == 'repair_repeat_interval')
    row['metrics_method'] = 'A different generated method'
    record['candidate']['output_sha256'] = _digest(record['candidate']['output'])
    audit.pop('authority_hmac_sha256')
    audit['authority_hmac_sha256'] = lab['authority']._mac(audit)
    path = lab['root'] / 'detached.json'
    path.write_text(json.dumps(audit, ensure_ascii=False), encoding='utf-8')
    with pytest.raises((ValueError, OSError, RuntimeError)):
        promote(lab, assessment_path=path)
    assert lab['store'].current() == lab['baseline']


def test_source_gate_rejects_invented_meeting_count(package_lab):
    from tof_reliability.vitaly_bot.hypotheses_auto_atomic_package import validate_source_evidence
    fields = ('inventory', 'candidate', 'proposal', 'classifier', 'verifier',
              'classifier_trace', 'verifier_trace')
    kwargs = {name: copy.deepcopy(package_lab[name]) for name in fields}
    row = next(d for d in kwargs['candidate']['dimension_projection']['dimensions']
               if d['dimension_id'] == 'repair_repeat_interval')
    row['independent_meeting_count'] += 1
    with pytest.raises(ValueError):
        validate_source_evidence(release_store=package_lab['store'], **kwargs)


def test_assessment_fingerprints_include_dimension_catalog_and_source_rules():
    from tof_reliability.vitaly_bot.hypotheses_scenario_comparison import _script_fingerprint
    scripts = _script_fingerprint()
    assert {'hypotheses_pilot.py', 'hypotheses_scenario_comparison.py',
            'hypotheses_topic_projection.py', 'hypotheses_interests.py'} <= scripts.keys()


def test_audit_fingerprints_exact_prompts_outputs_and_verified_cards(lab):
    from tof_reliability.vitaly_bot.hypotheses_scenario_comparison import _digest
    audit = evaluate(lab)
    for scenario in audit['scenarios']:
        for side in ('baseline', 'candidate'):
            run = scenario[side]
            for field in ('input', 'output', 'verified_cards'):
                assert run[field + '_sha256'] == _digest(run[field])
