"""Synthetic regressions for case scope; no live corpus or verifier is used."""

import copy

import pytest

from test_hypotheses_seven_compatible import case
from tof_reliability.vitaly_bot import hypotheses_pilot as pilot


CASE_SCOPE = 'насосы участка сгущения'


def run_case(case, cards, *, scope=CASE_SCOPE, coverage=None, verifier=None, generator=None):
    context, labels, audit, original_coverage, _ = case
    calls = []
    checks = []

    def generate(model_context):
        calls.append(copy.deepcopy(model_context))
        return copy.deepcopy(cards) if generator is None else generator(model_context)

    def verify(request):
        checks.append(copy.deepcopy(request))
        return (verifier(request) if verifier else
                {'supported': True, 'confidence': 'high'})

    result = pilot.generate_pilot(
        scope=scope, knowledge_version='synthetic-k', interest_context=context,
        verified_labels=labels, verifier_audit=audit,
        coverage_provider=lambda: copy.deepcopy(original_coverage if coverage is None else coverage),
        generator=generate, private_verifier=verify)
    return result, calls, checks


def scoped_cards(case, scope=CASE_SCOPE):
    cards = copy.deepcopy(case[4])
    for card in cards:
        card['scope'] = scope
    return cards


@pytest.mark.parametrize('unsupported', [
    'Согласованный перечень насосов участка сгущения',
    'Согласованная совокупность насосов участка сгущения',
    'Согласованные объекты и период для насосов участка сгущения',
    'Насосы участка сгущения; согласованный период',
])
def test_unsupported_prior_agreement_is_not_accepted_even_if_verifier_approves(case, unsupported):
    cards = scoped_cards(case)
    cards[0]['scope'] = unsupported
    try:
        result, calls, _ = run_case(case, cards)
    except ValueError:
        return
    assert len(calls) == 1
    assert all(card['scope'] != unsupported for card in result['cards'])


@pytest.mark.parametrize('missing', [
    'Согласованные объекты и период',
    'Оборудование и период определить перед анализом',
    'Насосы',  # omits the current user's specific equipment location
])
def test_missing_current_case_scope_cannot_be_accepted_or_silently_appended(case, missing):
    cards = scoped_cards(case)
    cards[0]['scope'] = missing
    topic = cards[0]['meeting_interest']['topic_id']
    try:
        result, calls, _ = run_case(case, cards)
    except ValueError:
        return
    assert len(calls) == 1
    assert topic not in {card['meeting_interest']['topic_id'] for card in result['cards']}


def test_future_agreement_is_conditional_and_preserves_exact_user_scope(case):
    cards = scoped_cards(case)
    cards[0]['scope'] = 'Насосы участка сгущения; определить и согласовать перечень и период до анализа'
    result, calls, checks = run_case(case, cards)
    assert len(calls) == 1 and len(checks) == 7
    assert len(result['cards']) == 7
    assert cards[0]['scope'] in [card['scope'] for card in result['cards']]
    pilot.validate_pilot(result, coverage=case[3], interest_context=case[0])


def test_explicitly_agreed_scope_from_current_user_is_allowed(case):
    scope = 'Согласованный перечень насосов участка сгущения'
    cards = scoped_cards(case, scope)
    result, calls, checks = run_case(case, cards, scope=scope)
    assert len(calls) == 1 and len(checks) == 7
    assert all(card['scope'] == scope for card in result['cards'])
    pilot.validate_pilot(result, coverage=case[3], interest_context=case[0])


def test_generator_instructions_require_scope_and_future_agreement(case):
    cards = scoped_cards(case)
    _, calls, _ = run_case(case, cards)
    context = calls[0]
    instruction = context['instructions'].casefold()
    assert context['scope'] == CASE_SCOPE
    assert 'согласова' in instruction
    assert any(word in instruction for word in ('определить', 'определя', 'будущ', 'предсто'))
    assert any(word in instruction for word in ('кажд', 'карточ'))
    assert any(word in instruction for word in ('область', 'периметр', 'охват', 'scope'))


def test_scope_changes_do_not_allow_invented_values_or_provenance(case):
    cards = scoped_cards(case)
    cards[0]['decision'] = 'Простой насосов составил 987654 часов.'
    with pytest.raises(ValueError):
        run_case(case, cards)
    cards = scoped_cards(case)
    cards[0]['source_references'] = ['synthetic:invented-source']
    with pytest.raises(ValueError):
        run_case(case, cards)
    cards = scoped_cards(case)
    cards[0]['meeting_interest'].update(reference='forged:chars:0-1', attribution='direct_request')
    with pytest.raises(ValueError):
        run_case(case, cards)


def test_unavailable_data_cannot_be_promoted_to_assumed_fact_or_generate(case):
    cards = scoped_cards(case)
    coverage = copy.deepcopy(case[3])
    for route in coverage['routes']:
        route['availability'] = 'unavailable'
    generated = []
    with pytest.raises((ValueError, RuntimeError)):
        run_case(case, cards, coverage=coverage,
                 generator=lambda context: generated.append(context) or cards)
    assert not generated


def test_private_verifier_remains_independent_and_never_retries(case):
    cards = scoped_cards(case)
    rejected = cards[0]['meeting_interest']['topic_id']

    def verify(request):
        if request['topic_id'] == rejected:
            return {'supported': False, 'confidence': 'high', 'reason': 'Синтетический отказ проверки'}
        return {'supported': True, 'confidence': 'high'}

    result, calls, checks = run_case(case, cards, verifier=verify)
    assert len(calls) == 1 and len(checks) == 7
    assert len(result['cards']) == 6
    assert result['rejected_cards'][0]['verdict']['reason'] == 'Синтетический отказ проверки'
    assert rejected not in {card['meeting_interest']['topic_id'] for card in result['cards']}
