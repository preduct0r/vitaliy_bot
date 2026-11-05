"""Synthetic, offline contracts for evidence-balanced candidate selection.

No private corpus, historical comparisons, model, or live server is used.
"""
import copy

import pytest

from test_hypotheses_dimensions_pilot import dimension_case, topic_case
from test_hypotheses_evidence_priority import _pool
from test_hypotheses_empty_scope import setup_case, generate as generate_empty
from tof_reliability.vitaly_bot import hypotheses_pilot as pilot
from tof_reliability.vitaly_bot import hypotheses_topic_projection as topics


def _generate(case, *, verifier=None, provider=None, scope='насосы'):
    interests, coverage, projection, cards = copy.deepcopy(case)
    return pilot.generate_dimension_pilot(
        scope=scope, knowledge_version='synthetic-k', interest_context=interests,
        dimension_projection=projection,
        coverage_provider=provider or (lambda: copy.deepcopy(coverage)),
        generator=lambda _: copy.deepcopy(cards),
        private_verifier=verifier or (lambda _: {'supported': True, 'confidence': 'high'}))


def _id(card):
    return card['meeting_interest']['dimension_id']


def _add_distinct_route(case, topic_case):
    """Add a source-backed candidate on an otherwise absent capability."""
    context, _, _, _, original_cards = topic_case
    dim = 'repair_labor_norm_actual'
    topic, description = topics._CATALOGS[topics.DIMENSION_VERSION][dim]
    interest = context['interests'][4]
    evidence = interest['evidence'][0]
    case[2]['dimensions'].append({
        'dimension_id': dim, 'topic_id': topic, 'description': description,
        'interest_ids': [interest['id']], 'source_ids': [evidence['source_id']],
        'evidence': [{'interest_id': interest['id'], **copy.deepcopy(evidence)}],
        'independent_meeting_count': 1,
    })
    card = copy.deepcopy(original_cards[4])
    card['meeting_interest'] = {'interest_id': interest['id'], 'topic_id': topic,
                                'dimension_id': dim}
    card['decision'] = 'Выбрать проверку: ' + description
    card['metrics_method'] = 'Проверить методом: ' + description
    case[3].append(card)
    return dim


def _order(result):
    return [_id(card) for card in result['cards']]


def test_recurrence_saturates_so_strong_utility_and_feasibility_can_win(dimension_case):
    case = copy.deepcopy(dimension_case)
    frequent, useful = case[2]['dimensions'][:2]
    frequent['independent_meeting_count'] = 100
    useful['independent_meeting_count'] = 4
    case[3][0]['ranking'].update(utility=0, feasibility=0, interest=0)
    case[3][1]['ranking'].update(utility=5, feasibility=5, interest=5)
    order = _order(_generate(case))
    assert order.index(useful['dimension_id']) < order.index(frequent['dimension_id'])


def test_direct_attribution_matters_without_becoming_absolute_priority(dimension_case):
    case = copy.deepcopy(dimension_case)
    # The first two cards share an interest; use another independently bound
    # interest for the competing inference, preserving authentic attribution.
    direct, inference = case[2]['dimensions'][0], case[2]['dimensions'][2]
    item = next(i for i in case[0]['interests'] if i['id'] == direct['interest_ids'][0])
    item['attributions'] = ['direct_request']
    item['evidence'][0]['attribution'] = 'direct_request'
    for row in case[2]['dimensions'][:2]:
        row['evidence'][0]['attribution'] = 'direct_request'
    direct['independent_meeting_count'] = inference['independent_meeting_count'] = 1
    for index in (0, 2):
        case[3][index]['ranking'].update(utility=3, feasibility=3, interest=3)
    order = _order(_generate(case))
    assert order.index(direct['dimension_id']) < order.index(inference['dimension_id'])
    case[3][0]['ranking'].update(utility=0, feasibility=0, interest=0)
    case[3][2]['ranking'].update(utility=5, feasibility=5, interest=5)
    order = _order(_generate(case))
    assert order.index(inference['dimension_id']) < order.index(direct['dimension_id'])


def test_selection_preserves_useful_distinct_capability_when_many_similar_candidates_recur(dimension_case, topic_case):
    case = _pool(dimension_case)
    novel = _add_distinct_route(case, topic_case)
    for row in case[2]['dimensions']:
        row['independent_meeting_count'] = 1 if row['dimension_id'] == novel else 20
    for card in case[3]:
        card['ranking'].update(utility=5, feasibility=5, interest=5)
    result = _generate(case)
    assert len(result['cards']) == 7
    assert novel in _order(result)
    assert len({c['capability_id'] for c in result['cards']}) >= 5
    assert len({_id(c) for c in result['cards']}) == 7


def test_ties_do_not_depend_on_generator_order(dimension_case):
    case = _pool(dimension_case)
    for card in case[3]:
        card['ranking'].update(utility=3, feasibility=3, interest=3)
    first = _order(_generate(case))
    case[3].reverse()
    second = _order(_generate(case))
    assert first == second
    assert first == _order(_generate(case))


def test_all_candidates_reviewed_before_selection_and_rejected_not_resurrected(dimension_case):
    case = _pool(dimension_case)
    rejected = _id(case[3][0])
    calls = []

    def review(request):
        calls.append(request['dimension_id'])
        if request['dimension_id'] == rejected:
            return {'supported': False, 'confidence': 'high', 'reason': 'нет основания'}
        return {'supported': True, 'confidence': 'high'}

    result = _generate(case, verifier=review)
    assert len(calls) == len(case[3]) and set(calls) == {_id(c) for c in case[3]}
    assert rejected not in _order(result)
    assert result['rejected_cards'][0]['verdict']['reason'] == 'нет основания'
    assert len(result['cards']) == 7
    assert [c['number'] for c in result['cards']] == list(range(1, 8))


def test_distinct_decisions_with_same_topic_survive_when_pool_is_small(dimension_case):
    result = _generate(dimension_case)
    assert {_id(dimension_case[3][0]), _id(dimension_case[3][1])} <= set(_order(result))
    assert len(result['cards']) == 6


def test_empty_scope_stays_unset_through_balanced_selection(setup_case):
    result, requests, reviews = generate_empty(setup_case)
    assert result['scope'] == requests[0]['scope'] == ''
    assert all(review['scope'] == '' for review in reviews)
    assert all('насос' not in card['scope'].casefold() for card in result['cards'])


def test_explicit_scope_and_server_failure_remain_enforced(dimension_case):
    case = copy.deepcopy(dimension_case)
    case[3][0]['scope'] = 'все оборудование'
    with pytest.raises(ValueError):
        _generate(case, scope='насосы')

    def unavailable():
        raise OSError('synthetic server unavailable')

    with pytest.raises(RuntimeError, match='(?i)server|coverage|unavailable'):
        _generate(dimension_case, provider=unavailable)


def test_third_party_paraphrase_is_ranked_as_indirect_evidence(dimension_case):
    case = copy.deepcopy(dimension_case)
    row = case[2]['dimensions'][2]
    interest = next(i for i in case[0]['interests'] if i['id'] == row['interest_ids'][0])
    interest['attributions'] = ['third_party_paraphrase']
    interest['evidence'][0]['attribution'] = 'third_party_paraphrase'
    row['evidence'][0]['attribution'] = 'third_party_paraphrase'
    row['attribution_meeting_counts'] = {'third_party_paraphrase': 1}
    assert row['dimension_id'] in _order(_generate(case))


def test_diversity_recognizes_distinct_decision_and_method_within_a_family():
    cards, rows = [], {}
    for index in range(8):
        dim = f'dimension-{index}'
        rows[dim] = {'independent_meeting_count': 2,
                     'evidence': [{'attribution': 'interpretation'}]}
        cards.append({'meeting_interest': {'dimension_id': dim, 'topic_id': 'same'},
                      'capability_id': 'same', 'source_readiness': 'partial',
                      'ranking': {'utility': 3, 'feasibility': 3, 'interest': 3},
                      'decision': 'choose common action' if index < 7 else 'prioritize alternative',
                      'metrics_method': 'compare common records' if index < 7 else 'observe independent measurements'})
    result = pilot._balanced_dimension_selection(cards, rows)
    assert 'dimension-7' in [_id(card) for card in result]
