"""Synthetic regression contracts for selection-level attribution.

No private pilot artifacts or model calls are needed to run these tests.  The
independent verifier below is a deliberately small test double, not an LLM or
evidence that an actual verifier will accept a particular business claim.
"""
import copy
import json

import pytest

from test_hypotheses_dimensions_pilot import dimension_case, topic_case
from tof_reliability.vitaly_bot import hypotheses_pilot as pilot


@pytest.fixture
def mixed_case(dimension_case):
    case = copy.deepcopy(dimension_case)
    context, _, projection, _ = case
    row = projection['dimensions'][0]
    # A different interest in the same dimension is supported by a third-party
    # paraphrase. It must not become the selected interpretation's attribution.
    other = copy.deepcopy(context['interests'][1])
    other['id'] = 'interest-attribution-other'
    other['attributions'] = ['third_party_paraphrase']
    other['evidence'][0]['attribution'] = 'third_party_paraphrase'
    context['interests'].append(other)
    row['interest_ids'].append(other['id'])
    row['source_ids'].append(other['evidence'][0]['source_id'])
    row['evidence'].append({'interest_id': other['id'], **other['evidence'][0]})
    row['independent_meeting_count'] = 2
    row['attribution_meeting_counts'] = {'interpretation': 1, 'third_party_paraphrase': 1}
    return case


def run_case(case, *, cards=None, verifier=None, scope='насосы'):
    context, coverage, projection, default_cards = copy.deepcopy(case)
    seen, checked = [], []

    def generator(request):
        seen.append(copy.deepcopy(request))
        return copy.deepcopy(default_cards if cards is None else cards)

    def independent_verifier(request):
        checked.append(copy.deepcopy(request))
        if verifier:
            return verifier(request)
        return {'supported': True, 'confidence': 'high'}

    result = pilot.generate_dimension_pilot(
        scope=scope, knowledge_version='synthetic', interest_context=context,
        dimension_projection=projection,
        coverage_provider=lambda: copy.deepcopy(coverage), generator=generator,
        private_verifier=independent_verifier)
    return result, seen[0], checked


def selected_row(request, card):
    dim = card['meeting_interest']['dimension_id']
    return next(row for row in request['dimensions'] if row['dimension_id'] == dim)


def test_generator_gets_exact_selection_attribution_without_private_evidence(mixed_case):
    original = copy.deepcopy(mixed_case)
    run, request, checks = run_case(mixed_case)
    card = mixed_case[3][0]
    row = selected_row(request, card)
    ident = card['meeting_interest']['interest_id']
    other = mixed_case[0]['interests'][-1]['id']
    # Public contract: the mapping is keyed by interest within the dimension;
    # values contain only the attribution of the deterministic selected binding.
    assert row['interest_attributions'][ident] == 'interpretation'
    assert row['interest_attributions'][other] == 'third_party_paraphrase'
    assert set(row['interest_attributions']) == set(row['interest_ids'])
    assert row['independent_meeting_count'] == 2
    assert row['attribution_meeting_counts'] == {'interpretation': 1, 'third_party_paraphrase': 1}
    instructions = request['instructions'].casefold()
    assert ('выбранн' in instructions or 'selected' in instructions)
    assert ('привяз' in instructions or 'binding' in instructions)
    assert ('общ' in instructions or 'general' in instructions)
    serialized = json.dumps(request, ensure_ascii=False)
    for interest in mixed_case[0]['interests']:
        assert interest['requirement'] not in serialized
        for evidence in interest['evidence']:
            assert evidence['fragment'] not in serialized
            assert evidence['sha256'] not in serialized
            assert evidence['locator'] not in serialized
    assert 'reference' not in row and 'source_id' not in row
    review = next(c for c in checks if c['dimension_id'] == row['dimension_id'])
    accepted = next(c for c in run['cards'] if c['meeting_interest']['dimension_id'] == row['dimension_id'])
    assert review['source_evidence']['attribution'] == row['interest_attributions'][ident]
    for field in ('source_id', 'locator', 'reference', 'sha256', 'attribution'):
        assert review['source_evidence'][field] == accepted['meeting_interest'][field]
    assert mixed_case == original


@pytest.mark.parametrize('brief,accepted', [
    ('Основание интереса: интерпретация встречи.', True),
    ('Основание интереса: прямой запрос.', False),
    ('Основание интереса: пересказ третьего лица.', False),
    ('Основание интереса: интерпретация и пересказ третьего лица.', False),
])
def test_independent_review_accepts_only_selected_attribution(mixed_case, brief, accepted):
    cards = copy.deepcopy(mixed_case[3])
    cards[0]['analyst_brief'] = brief
    dim = cards[0]['meeting_interest']['dimension_id']

    def review(request):
        if request['dimension_id'] != dim:
            return {'supported': True, 'confidence': 'high'}
        assert request['source_evidence']['attribution'] == 'interpretation'
        supported = request['card']['analyst_brief'] == 'Основание интереса: интерпретация встречи.'
        return ({'supported': True, 'confidence': 'high'} if supported else
                {'supported': False, 'confidence': 'high', 'reason': 'analyst_brief: неподтверждённая атрибуция'})

    run, _, checks = run_case(mixed_case, cards=cards, verifier=review)
    assert len(checks) == len(cards)
    found = [c for c in run['cards'] if c['meeting_interest']['dimension_id'] == dim]
    assert bool(found) is accepted
    if accepted:
        assert found[0]['analyst_brief'] == brief
    else:
        rejection = next(r for r in run['rejected_cards'] if r['card']['meeting_interest']['dimension_id'] == dim)
        assert rejection['card']['analyst_brief'] == brief
        assert 'analyst_brief' in rejection['verdict']['reason']


def test_overlapping_attributions_do_not_inflate_independent_meeting_count(mixed_case):
    row = mixed_case[2]['dimensions'][0]
    # Both attribution types can occur at one meeting; per-type counts overlap.
    row['independent_meeting_count'] = 1
    row['attribution_meeting_counts'] = {'interpretation': 1, 'third_party_paraphrase': 1}
    _, request, _ = run_case(mixed_case)
    visible = selected_row(request, mixed_case[3][0])
    assert visible['independent_meeting_count'] == 1
    assert visible['attribution_meeting_counts'] == row['attribution_meeting_counts']
    assert visible['interest_attributions'][mixed_case[3][0]['meeting_interest']['interest_id']] == 'interpretation'


@pytest.mark.parametrize('reverse', [False, True])
def test_multiple_bindings_for_same_selection_are_not_silently_order_dependent(mixed_case, reverse):
    context, _, projection, _ = mixed_case
    row = projection['dimensions'][0]
    ident = mixed_case[3][0]['meeting_interest']['interest_id']
    interest = next(i for i in context['interests'] if i['id'] == ident)
    second = copy.deepcopy(interest['evidence'][0])
    second.update(source_id='other-synthetic-source', locator='chars:20-30',
                  reference='other-synthetic-source:chars:20-30', attribution='third_party_paraphrase')
    interest['attributions'].append('third_party_paraphrase')
    interest['evidence'].append(second)
    row['source_ids'].append(second['source_id'])
    row['evidence'].append({'interest_id': ident, **second})
    if reverse:
        row['evidence'].reverse()
    # The old setdefault silently chose whichever citation happened to be first.
    # Require either a single explicitly unambiguous selection or fail closed.
    try:
        run, request, checks = run_case(mixed_case)
    except ValueError:
        return
    visible = selected_row(request, mixed_case[3][0])
    binding = visible['interest_attributions'][ident]
    review = next(c for c in checks if c['dimension_id'] == row['dimension_id'])
    assert binding == review['source_evidence']['attribution']
    chosen = next(c for c in run['cards'] if c['meeting_interest']['dimension_id'] == row['dimension_id'])
    assert chosen['meeting_interest']['reference'] == review['source_evidence']['reference']
    # Selection must have the same defined result regardless of input ordering.
    assert binding == 'interpretation'
