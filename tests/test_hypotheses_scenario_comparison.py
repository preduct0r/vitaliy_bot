"""Red-first contract for an offline, independently checked comparison.

Proposed seam (not an existing API): hypotheses_scenario_comparison exposes
``fixed_scenarios()`` and ``evaluate_scenarios``. The latter accepts baseline
and candidate mappings (knowledge_version, interest_context,
dimension_projection), scope, coverage_provider, generator,
private_verifier, scenarios, and audit_dir. It returns a mapping with
``activation_allowed``, ``scenarios`` and ``audit_path``. Each scenario
record contains its id, input, baseline/candidate actual pilot output and
property observations, mandatory/critical flags, and comparison. Audit is JSON
with the same scenario records. Observations are deterministic and rooted in
fixed facts/predicates, not model-authored labels. Failed execution may raise or
return a rejected result, but never authorize activation.

The generator uses the actual dimension-pilot callback signature and produces
valid cards based on the source-bound dimension context it is given. No network, external
models, production sources or personal data are used.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

from test_hypotheses_dimensions_pilot import dimension_case, topic_case
from tof_reliability.vitaly_bot import hypotheses_pilot as pilot


PRIVATE_CASE = 'SYNTHETIC_CASE_PRIVATE_DO_NOT_EXPORT'
PRIVATE_MEETING = 'SYNTHETIC_MEETING_PRIVATE_DO_NOT_EXPORT'
BASE = 'synthetic-old-knowledge'
CANDIDATE = 'synthetic-new-knowledge'


def comparison_module():
    from tof_reliability.vitaly_bot import hypotheses_scenario_comparison as comparison
    return comparison


@pytest.fixture
def lab(dimension_case, monkeypatch):
    context, coverage, projection, cards = copy.deepcopy(dimension_case)
    # These markers are private case/meeting material. The production generator
    # contract only passes curated dimensions, never this raw content.
    context['interests'][0]['requirement'] += ' ' + PRIVATE_MEETING
    calls, checks, pilots = [], [], []
    mode = {'value': 'improved', 'fail': False, 'drift': False, 'outage': False,
            'broken_scenario': None}
    real_pilot = pilot.generate_dimension_pilot

    def pilot_spy(**kwargs):
        pilots.append({k: copy.deepcopy(kwargs[k]) for k in
                       ('scope', 'knowledge_version', 'interest_context', 'dimension_projection')})
        return real_pilot(**kwargs)

    monkeypatch.setattr(pilot, 'generate_dimension_pilot', pilot_spy)

    def coverage_provider():
        if mode['outage']:
            raise OSError('synthetic published server outage')
        value = copy.deepcopy(coverage)
        if mode['drift'] and calls:
            value['generation']['run_id'] = 'synthetic-drift'
        return value

    def generator(request):
        calls.append(copy.deepcopy(request))
        if mode['fail']:
            raise RuntimeError('synthetic model failure')
        result = copy.deepcopy(cards)
        # Do not echo the synthetic meeting fragments embedded by the older
        # fixture; generated prose uses the public dimension descriptions.
        descriptions = {row['dimension_id']: row['description'] for row in projection['dimensions']}
        for card in result:
            dim = card['meeting_interest']['dimension_id']
            for interest in context['interests']:
                for evidence in interest['evidence']:
                    fragment = evidence.get('fragment', '')
                    if fragment:
                        for field in ('management_question', 'decision', 'metrics_method', 'analyst_brief'):
                            card[field] = card[field].replace(fragment, descriptions[dim])
        # The unchanged baseline has a general repair-interval method. The new
        # context actually makes the generator describe a comparison of repeat
        # intervals; the claimed score and verdict below are never evidence.
        interval = next(d for d in request['dimensions'] if d['dimension_id'] == 'repair_repeat_interval')
        has_new_evidence = 'interval-pattern' in interval['interest_ids']
        if has_new_evidence and mode['value'] != 'tie':
            row = next(c for c in result if c['meeting_interest']['dimension_id'] == 'repair_repeat_interval')
            row['meeting_interest']['interest_id'] = 'interval-pattern'
            row['metrics_method'] = 'Сопоставить интервалы между повторными ремонтными эпизодами оборудования по периодам'
            if mode['value'] == 'regression':
                row['scope'] = 'всё оборудование'
            if mode['value'] == 'ungrounded':
                row['analyst_brief'] = 'Ранее потрачено 987654321 рублей.'
        scenario = request['scenario']
        brief = {
            'true_contradiction': 'Противоречие подтверждено при одинаковых определениях и периодах',
            'false_contradiction': 'Противоречие не подтверждено при разных периодах',
            'units': 'Сопоставить единицы измерения перед сравнением',
            'criticality': 'Проверить критичность по подтверждённому источнику',
            'missing_join': 'Проверить отсутствующую связь оборудования и событий',
            'outage': 'При недоступности источника не делать вывод',
        }.get(scenario['kind'])
        if brief:
            result[0]['analyst_brief'] = brief
        if mode['broken_scenario'] == scenario['kind'] and has_new_evidence:
            result[0]['analyst_brief'] = 'Данных для проверки пока недостаточно'
        if mode['value'] == 'self_report':
            for card in result:
                card['declared_quality'] = 100
                card['model_verdict'] = 'pass'
        return result

    def verifier(request):
        checks.append(copy.deepcopy(request))
        return {'supported': True, 'confidence': 'high'}

    output = Path(os.environ.get('WORKFLOW_OUTPUT_DIR', tempfile.gettempdir()))
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='scenario-comparison-', dir=output) as directory:
        yield {'context': context, 'coverage': coverage, 'projection': projection,
               'calls': calls, 'checks': checks, 'pilots': pilots, 'mode': mode,
               'coverage_provider': coverage_provider, 'generator': generator,
               'verifier': verifier, 'audit_dir': Path(directory)}


def evaluate(lab, **changes):
    module = comparison_module()
    candidate_context = copy.deepcopy(lab['context'])
    interest = copy.deepcopy(candidate_context['interests'][0])
    interest['id'] = 'interval-pattern'
    candidate_context['interests'].append(interest)
    candidate_projection = copy.deepcopy(lab['projection'])
    row = next(d for d in candidate_projection['dimensions'] if d['dimension_id'] == 'repair_repeat_interval')
    row['interest_ids'].append(interest['id'])
    row['evidence'].append({'interest_id': interest['id'], **interest['evidence'][0]})
    kwargs = {'baseline': {'knowledge_version': BASE,
                           'interest_context': copy.deepcopy(lab['context']),
                           'dimension_projection': copy.deepcopy(lab['projection'])},
              'candidate': {'knowledge_version': CANDIDATE,
                            'interest_context': candidate_context,
                            'dimension_projection': candidate_projection},
              'scope': 'насосы', 'coverage_provider': lab['coverage_provider'],
              'generator': lab['generator'], 'private_verifier': lab['verifier'],
              'scenarios': module.fixed_scenarios(), 'audit_dir': lab['audit_dir']}
    kwargs.update(changes)
    return module.evaluate_scenarios(**kwargs)


def rejected_or_error(lab, **kwargs):
    try:
        result = evaluate(lab, **kwargs)
    except (ValueError, RuntimeError, OSError):
        return
    assert result['activation_allowed'] is False


def test_audit_binds_exact_baseline_and_candidate_payloads(lab):
    baseline = {'knowledge_version': BASE, 'interest_context': copy.deepcopy(lab['context']),
                'dimension_projection': copy.deepcopy(lab['projection'])}
    result = evaluate(lab)
    candidate = {'knowledge_version': CANDIDATE, 'interest_context': lab['pilots'][1]['interest_context'],
                 'dimension_projection': lab['pilots'][1]['dimension_projection']}
    audit = json.loads(Path(result['audit_path']).read_text(encoding='utf-8'))
    def digest(value):
        data = json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False).encode('utf-8')
        return hashlib.sha256(data).hexdigest()
    assert audit['knowledge_payload_sha256'] == {
        'baseline': digest(baseline), 'candidate': digest(candidate)}


def test_fixed_suite_has_independent_known_cases_and_mandatory_coverage():
    scenarios = comparison_module().fixed_scenarios()
    assert len(scenarios) >= 12
    assert len({s['id'] for s in scenarios}) == len(scenarios)
    kinds = {s['kind'] for s in scenarios}
    assert {'true_contradiction', 'false_contradiction', 'units', 'unsupported_data',
            'criticality', 'repair_episodes', 'scope', 'missing_join', 'outage'} <= kinds
    assert all(s['input'] and s['checks'] for s in scenarios)
    assert all(any(c['mandatory'] for c in s['checks']) for s in scenarios)
    # Truth and target expectations are part of the immutable test suite, not
    # output of the candidate or a natural-language model verdict.
    true_case = next(s for s in scenarios if s['kind'] == 'true_contradiction')
    false_case = next(s for s in scenarios if s['kind'] == 'false_contradiction')
    assert true_case['expected']['contradiction'] is True
    assert false_case['expected']['contradiction'] is False
    assert true_case['input'] != false_case['input']


def test_actual_pilot_generation_measures_improvement_on_frozen_inputs_and_audits(lab):
    result = evaluate(lab)
    assert result['activation_allowed'] is True
    assert len(lab['pilots']) >= 2 and lab['checks']
    assert all(BASE not in str(c) and CANDIDATE not in str(c) for c in lab['calls'])
    for old, new in zip(lab['calls'][::2], lab['calls'][1::2]):
        assert {k: v for k, v in old.items() if k != 'dimensions'} == {k: v for k, v in new.items() if k != 'dimensions'}
        assert old['dimensions'] != new['dimensions']
    assert all(p['scope'] == 'насосы' for p in lab['pilots'])
    records = result['scenarios']
    assert len(records) >= 12
    assert any(r['comparison']['improved'] for r in records)
    assert all(not r['comparison']['degraded'] and not r['comparison']['new_critical_error'] for r in records)
    assert all(r['input'] and r['baseline']['output'] and r['candidate']['output']
               and r['baseline']['observations'] and r['candidate']['observations'] for r in records)
    assert any(r['baseline']['output'] != r['candidate']['output'] for r in records)
    audit_path = Path(result['audit_path']).resolve()
    assert audit_path.is_relative_to(lab['audit_dir'].resolve())
    persisted = json.loads(audit_path.read_text(encoding='utf-8'))
    assert persisted['scenarios'] == records
    assert persisted['activation_allowed'] is True
    encoded = json.dumps({'model_requests': lab['calls'], 'audit': persisted}, ensure_ascii=False)
    assert PRIVATE_CASE not in encoded and PRIVATE_MEETING not in encoded
    # Actual card text is retained exactly, rather than a candidate's declared
    # verdict being substituted for generation results.
    assert 'Сопоставить интервалы между повторными ремонтными эпизодами' in encoded


def test_tie_and_model_self_report_cannot_authorize(lab):
    lab['mode']['value'] = 'tie'
    rejected_or_error(lab)
    assert lab['calls'], 'the comparison must actually exercise generation'
    lab['mode']['value'] = 'self_report'
    rejected_or_error(lab)


@pytest.mark.parametrize('kind', ['true_contradiction', 'false_contradiction', 'units',
                                  'criticality', 'missing_join', 'outage'])
def test_scenario_specific_regression_is_observed_from_generated_cards(lab, kind):
    lab['mode']['broken_scenario'] = kind
    rejected_or_error(lab)
    assert {call['scenario']['kind'] for call in lab['calls']} >= {kind}


@pytest.mark.parametrize('mode', ['regression', 'ungrounded'])
def test_new_critical_regression_or_unsupported_claim_rejects(lab, mode):
    lab['mode']['value'] = mode
    rejected_or_error(lab)
    assert lab['calls']


@pytest.mark.parametrize('failure', ['missing_evidence', 'generation_drift', 'model_failure',
                                     'coverage_outage', 'same_verifier'])
def test_missing_or_unexecutable_checks_fail_closed_without_stale_fallback(lab, failure):
    kwargs = {}
    if failure == 'missing_evidence':
        kwargs['candidate'] = {'knowledge_version': CANDIDATE,
                               'interest_context': copy.deepcopy(lab['context']),
                               'dimension_projection': copy.deepcopy(lab['projection'])}
        kwargs['candidate']['dimension_projection']['dimensions'][0]['evidence'] = []
    elif failure == 'generation_drift':
        lab['mode']['drift'] = True
    elif failure == 'model_failure':
        lab['mode']['fail'] = True
    elif failure == 'coverage_outage':
        lab['mode']['outage'] = True
    else:
        kwargs['private_verifier'] = lab['generator']
    rejected_or_error(lab, **kwargs)
    if failure == 'coverage_outage':
        assert not lab['calls'], 'outage must not use a cached or local generation'


def test_missing_or_unknown_mandatory_check_and_scenario_are_not_success(lab):
    module = comparison_module()
    original = module.fixed_scenarios()
    for damage in ('missing_check', 'unknown_check', 'missing_scenario'):
        scenarios = copy.deepcopy(original)
        if damage == 'missing_check':
            scenarios[0]['checks'] = []
        elif damage == 'unknown_check':
            scenarios[0]['checks'][0]['kind'] = 'unknown-unexecutable-check'
        else:
            scenarios.pop()
        rejected_or_error(lab, scenarios=scenarios)


def test_candidate_cannot_define_its_own_expected_results(lab):
    module = comparison_module()
    scenarios = copy.deepcopy(module.fixed_scenarios())
    for case in scenarios:
        case['expected'] = {'contradiction': False, 'score': 100, 'model_verdict': 'pass'}
    lab['mode']['value'] = 'tie'
    rejected_or_error(lab, scenarios=scenarios)


def test_different_safe_generator_contexts_cannot_be_compared(lab):
    candidate = {'knowledge_version': CANDIDATE,
                 'interest_context': copy.deepcopy(lab['context']),
                 'dimension_projection': copy.deepcopy(lab['projection'])}
    candidate['dimension_projection']['dimensions'][0]['independent_meeting_count'] = 2
    rejected_or_error(lab, candidate=candidate)


def test_raw_private_echo_never_reaches_audit(lab):
    def leaking_generator(request):
        value = lab['generator'](request)
        value[0]['analyst_brief'] = PRIVATE_MEETING
        return value
    rejected_or_error(lab, generator=leaking_generator)
    for path in lab['audit_dir'].glob('*.json'):
        assert PRIVATE_MEETING not in path.read_text(encoding='utf-8')


def test_failed_model_is_audited_without_inventing_an_output(lab):
    lab['mode']['fail'] = True
    rejected_or_error(lab)
    reports = list(lab['audit_dir'].glob('*.json'))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding='utf-8'))
    assert report['activation_allowed'] is False
    assert report['failure']
    assert report['attempts'][0]['input']['scenario']['id']
    assert report['attempts'][0]['output'] is None


def test_audit_directory_inside_worktree_is_forbidden(lab):
    rejected_or_error(lab, audit_dir=Path(__file__).parent)
    assert not lab['calls']
