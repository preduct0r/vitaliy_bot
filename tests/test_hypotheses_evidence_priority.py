"""Offline evidence-priority contracts; fixtures contain synthetic data only.

The expanded s1 projection supplies source-verified dimension evidence and
independent_meeting_count. These tests exercise its downstream contract without
reading the private corpus or calling a model, Telegram, or the TOF server.
"""
import copy
import json

import pytest

from test_hypotheses_dimensions_pilot import dimension_case, evidence_case, reconcile
from test_hypotheses_seven_compatible import case as topic_case
from tof_reliability.vitaly_bot import hypotheses_pilot as pilot
from tof_reliability.vitaly_bot import hypotheses_topic_projection as topics


def _generate(case, cards=None, *, verifier=None, provider=None, generator=None, scope='насосы'):
    interests, coverage, projection, defaults = copy.deepcopy(case)
    return pilot.generate_dimension_pilot(
        scope=scope, knowledge_version='k1', interest_context=interests,
        dimension_projection=projection,
        coverage_provider=provider or (lambda: copy.deepcopy(coverage)),
        generator=generator or (lambda _: copy.deepcopy(defaults if cards is None else cards)),
        private_verifier=verifier or (lambda _: {'supported': True, 'confidence': 'high'}),
    )


def _pool(case):
    """Nine genuinely distinct catalog dimensions, each with a source and route."""
    context, coverage, projection, cards = copy.deepcopy(case)
    catalog = topics._CATALOGS[topics.DIMENSION_VERSION]
    for dim, source in [('repair_episode_causes', 0),
                        ('repair_cost_by_period_asset_work', 3),
                        ('repair_cost_per_action', 3)]:
        topic, description = catalog[dim]
        row = copy.deepcopy(projection['dimensions'][source])
        row.update(dimension_id=dim, topic_id=topic, description=description)
        projection['dimensions'].append(row)
        card = copy.deepcopy(cards[source])
        card['meeting_interest'].update(dimension_id=dim, topic_id=topic)
        card['decision'] = 'Выбрать проверку: ' + description
        card['metrics_method'] = 'Проверить методом: ' + description
        card['number'] = len(cards) + 1
        cards.append(card)
    return context, coverage, projection, cards


def test_model_receives_verified_recurrence_and_distinguishes_attribution_without_private_text(dimension_case):
    case = copy.deepcopy(dimension_case)
    # Separate counts are evidence strength, not model-authored business facts.
    first, second = case[2]['dimensions'][:2]
    first['independent_meeting_count'], second['independent_meeting_count'] = 4, 1
    ident = first['interest_ids'][0]
    interest = next(x for x in case[0]['interests'] if x['id'] == ident)
    interest['attributions'] = ['direct_request']
    interest['evidence'][0]['attribution'] = 'direct_request'
    first['evidence'][0]['attribution'] = 'direct_request'
    # The fixture's first two dimensions share an interest; make attribution
    # distinct using the third independently source-bound interest instead.
    second.update(interest_ids=copy.deepcopy(case[2]['dimensions'][2]['interest_ids']),
                  source_ids=copy.deepcopy(case[2]['dimensions'][2]['source_ids']),
                  evidence=copy.deepcopy(case[2]['dimensions'][2]['evidence']))
    case[3][1]['meeting_interest']['interest_id'] = second['interest_ids'][0]
    case[0]['interests'][0]['requirement'] += ' PRIVATE_BUSINESS_987654321'
    first['evidence'][0]['fragment'] = 'PRIVATE_QUOTATION_MARKER'
    seen = []
    _generate(case, generator=lambda request: seen.append(request) or copy.deepcopy(case[3]))
    rows = {r['dimension_id']: r for r in seen[0]['dimensions']}
    encoded_first = json.dumps(rows[first['dimension_id']], ensure_ascii=False)
    encoded_second = json.dumps(rows[second['dimension_id']], ensure_ascii=False)
    assert rows[first['dimension_id']]['independent_meeting_count'] == 4
    assert rows[second['dimension_id']]['independent_meeting_count'] == 1
    assert 'direct_request' in encoded_first
    assert 'interpretation' in encoded_second
    assert 'PRIVATE_BUSINESS_987654321' not in json.dumps(seen, ensure_ascii=False)
    assert 'PRIVATE_QUOTATION_MARKER' not in json.dumps(seen, ensure_ascii=False)
    assert 'chars:' not in json.dumps(seen, ensure_ascii=False)


def test_reconciliation_deduplicates_same_meeting_and_keeps_reported_attribution(evidence_case):
    data = copy.deepcopy(evidence_case)
    # The same source and locator appear in both independent projection audits.
    row = copy.deepcopy(data['v2']['dimensions'][0])
    data['v1']['dimensions'].append(row)
    data['v1_audit'].extend(copy.deepcopy(data['v2_audit']))
    result = reconcile(data)
    merged = next(x for x in result['dimensions'] if x['dimension_id'] == row['dimension_id'])
    assert merged['independent_meeting_count'] == 1
    assert len(merged['evidence']) == 1
    assert merged['evidence'][0]['attribution'] == 'third_party'


def test_bounded_candidate_pool_survives_two_independent_rejections_and_preserves_reasons(dimension_case):
    case = _pool(dimension_case)
    seen = []
    def review(request):
        seen.append(copy.deepcopy(request))
        if request['dimension_id'] in {case[3][0]['meeting_interest']['dimension_id'],
                                       case[3][1]['meeting_interest']['dimension_id']}:
            return {'supported': False, 'confidence': 'high', 'reason': 'decision: недостаточно обосновано'}
        return {'supported': True, 'confidence': 'high'}
    run = _generate(case, verifier=review)
    assert 5 <= len(run['cards']) <= 7
    assert len(seen) == len(case[3])
    assert len(run['rejected_cards']) == 2
    assert all('недостаточно обосновано' in r['verdict']['reason'] for r in run['rejected_cards'])
    assert {c['meeting_interest']['dimension_id'] for c in run['cards']}.isdisjoint(
        {case[3][0]['meeting_interest']['dimension_id'], case[3][1]['meeting_interest']['dimension_id']})
    assert [c['number'] for c in run['cards']] == list(range(1, len(run['cards']) + 1))


def test_recurring_direct_need_can_outrank_high_model_score_for_weak_inference(dimension_case):
    case = copy.deepcopy(dimension_case)
    strong, weak = case[2]['dimensions'][:2]
    strong['independent_meeting_count'], weak['independent_meeting_count'] = 5, 1
    # Use distinct interests and attribution for the two dimensions.
    weak.update(interest_ids=copy.deepcopy(case[2]['dimensions'][2]['interest_ids']),
                source_ids=copy.deepcopy(case[2]['dimensions'][2]['source_ids']),
                evidence=copy.deepcopy(case[2]['dimensions'][2]['evidence']))
    case[3][1]['meeting_interest']['interest_id'] = weak['interest_ids'][0]
    interest = next(i for i in case[0]['interests'] if i['id'] == strong['interest_ids'][0])
    interest['attributions'] = ['direct_request']
    interest['evidence'][0]['attribution'] = 'direct_request'
    strong['evidence'][0]['attribution'] = 'direct_request'
    case[3][0]['ranking'].update(utility=3, feasibility=3, interest=3)
    case[3][1]['ranking'].update(utility=5, feasibility=5, interest=5)
    run = _generate(case)
    order = [c['meeting_interest']['dimension_id'] for c in run['cards']]
    assert order.index(strong['dimension_id']) < order.index(weak['dimension_id'])


def test_verified_unavailable_route_can_be_exploratory_and_discloses_missing_data(dimension_case):
    case = copy.deepcopy(dimension_case)
    card = case[3][3]
    route = next(r for r in case[1]['routes'] if r['capability_id'] == card['capability_id'])
    route.update(availability='unavailable', missing_evidence=['независимый набор данных отсутствует'],
                 source_references=[])
    card['source_references'] = []
    card['metrics_method'] = 'Сначала установить доступность источника и метод проверки'
    card['decision'] = 'Определить, какие данные нужны для проверки поставок'
    card['missing_evidence'] = ['независимый набор данных отсутствует']
    seen = []
    run = _generate(case, verifier=lambda request: seen.append(request) or {'supported': True, 'confidence': 'high'})
    selected = next(c for c in run['cards'] if c['meeting_interest']['dimension_id'] == card['meeting_interest']['dimension_id'])
    assert selected['source_readiness'] == 'unavailable'
    assert 'независимый набор данных отсутствует' in selected['missing_evidence']
    assert selected['source_references'] == []
    assert any(r['route']['availability'] == 'unavailable' for r in seen)


@pytest.mark.parametrize('damage', ['scope', 'invented_source', 'false_availability', 'private_fact'])
def test_exploratory_or_partial_candidate_does_not_relax_safety(dimension_case, damage):
    case = copy.deepcopy(dimension_case)
    card = case[3][0]
    if damage == 'scope':
        card['scope'] = 'все оборудование предприятия'
    elif damage == 'invented_source':
        card['source_references'] = ['ledger:invented']
    elif damage == 'false_availability':
        card['analyst_brief'] = 'Все необходимые данные доступны и результат рассчитан.'
    else:
        card['analyst_brief'] = 'Компания СЕКРЕТ ранее потратила 987654321 рублей.'
    with pytest.raises((ValueError, RuntimeError)):
        _generate(case, verifier=lambda r: {'supported': False, 'confidence': 'high', 'reason': 'analyst_brief: неподтверждённое утверждение'} if damage == 'false_availability' else {'supported': True, 'confidence': 'high'})


def test_fewer_than_five_after_independent_review_fails(dimension_case):
    case = _pool(dimension_case)
    calls = []
    def review(request):
        calls.append(request['dimension_id'])
        return {'supported': False, 'confidence': 'high', 'reason': 'decision: нет подтверждения'} if len(calls) <= 5 else {'supported': True, 'confidence': 'high'}
    with pytest.raises((ValueError, RuntimeError), match='(?i)five|пяти|5|verified'):
        _generate(case, verifier=review)
    assert len(calls) >= 5


def test_server_unavailable_and_missing_attribution_fail_before_generation(dimension_case):
    calls = []
    def failed_coverage():
        raise OSError('synthetic server down')
    with pytest.raises((ValueError, RuntimeError), match='(?i)server|coverage|down'):
        _generate(dimension_case, provider=failed_coverage, generator=lambda r: calls.append(r))
    case = copy.deepcopy(dimension_case)
    case[2]['dimensions'][0]['evidence'][0].pop('attribution')
    with pytest.raises((ValueError, RuntimeError)):
        _generate(case, generator=lambda r: calls.append(r))
    assert calls == []


def test_independent_reviewer_exception_keeps_diagnostic(dimension_case):
    def broken(_):
        raise RuntimeError('reviewer synthetic timeout')
    with pytest.raises((ValueError, RuntimeError)) as exc:
        _generate(dimension_case, verifier=broken)
    assert 'synthetic timeout' in str(exc.value) or (exc.value.__cause__ and 'synthetic timeout' in str(exc.value.__cause__))


def test_candidate_pool_rejects_duplicates_and_unbounded_input(dimension_case):
    case = _pool(dimension_case)
    duplicate = copy.deepcopy(case[3])
    duplicate[1]['decision'] = duplicate[0]['decision']
    duplicate[1]['metrics_method'] = duplicate[0]['metrics_method']
    with pytest.raises((ValueError, RuntimeError)):
        _generate(case, cards=duplicate)
    with pytest.raises((ValueError, RuntimeError)):
        _generate(case, cards=case[3] * 100)


def test_expanded_projection_is_bound_to_rehashed_profile_index_and_meeting_count(evidence_case):
    data = copy.deepcopy(evidence_case)
    result = pilot.bind_expanded_projection(profile=data['profile'], source_index=data['index'],
                                            projection=data['v2'])
    assert result['version'] == pilot.DIMENSION_PILOT_VERSION
    assert result['dimensions'][0]['independent_meeting_count'] == 1
    data['v2']['dimensions'][0]['independent_meeting_count'] = 2
    with pytest.raises(ValueError):
        pilot.bind_expanded_projection(profile=data['profile'], source_index=data['index'], projection=data['v2'])


def test_available_route_requires_a_real_selected_reference(dimension_case):
    case = copy.deepcopy(dimension_case)
    case[3][0]['source_references'] = []
    with pytest.raises(ValueError):
        _generate(case)


def test_unmapped_direction_states_verified_route_gap_without_inventing_dataset(dimension_case):
    case = _pool(dimension_case)
    catalog = topics._CATALOGS[topics.DIMENSION_VERSION]
    dimension = next(d for d, (topic, _) in catalog.items() if topic not in pilot.TOPIC_ROUTES)
    topic, description = catalog[dimension]
    row = copy.deepcopy(case[2]['dimensions'][0])
    row.update(dimension_id=dimension, topic_id=topic, description=description, independent_meeting_count=4)
    case[2]['dimensions'].append(row)
    card = copy.deepcopy(case[3][0])
    card['meeting_interest'].update(dimension_id=dimension, topic_id=topic)
    card.update(capability_id='NO-VERIFIED-ROUTE', source_references=[],
                decision='Определить данные для направления: ' + description,
                metrics_method='Исследовать доступные источники: ' + description)
    card['management_question'] = 'Какие данные о ремонте нужно проверить?'
    case[3].append(card)
    seen = []
    result = _generate(case, generator=lambda request: seen.append(request) or case[3])
    selected = next(c for c in result['cards'] if c['meeting_interest']['dimension_id'] == dimension)
    assert selected['source_readiness'] == 'unavailable'
    assert not selected['source_references']
    assert any('маршрут' in entry for entry in selected['missing_evidence'])
    assert 'NO-VERIFIED-ROUTE' in seen[0]['eligible_dimension_routes'][dimension]


def test_generator_is_told_to_avoid_written_numeric_claims(dimension_case):
    seen = []
    _generate(dimension_case, generator=lambda request: seen.append(request) or copy.deepcopy(dimension_case[3]))
    assert 'числительн' in seen[0]['instructions'].lower()


def test_candidate_validation_error_identifies_card_and_field(dimension_case):
    case = copy.deepcopy(dimension_case)
    case[3][0]['decision'] = 'Выбрать два направления проверки'
    with pytest.raises(ValueError, match='(?i)card 1.*decision'):
        _generate(case)


def test_expanded_projection_counts_attribution_per_independent_meeting(evidence_case):
    data = copy.deepcopy(evidence_case)
    result = pilot.bind_expanded_projection(profile=data['profile'], source_index=data['index'], projection=data['v2'])
    row = result['dimensions'][0]
    assert row['attribution_meeting_counts']['third_party'] == 1
    assert row['attribution_meeting_counts'].get('direct_request', 0) == 0


def test_impossible_attribution_counts_are_rejected(dimension_case):
    case = copy.deepcopy(dimension_case)
    case[2]['dimensions'][0]['attribution_meeting_counts'] = {'direct_request': 999}
    with pytest.raises(ValueError):
        _generate(case)
