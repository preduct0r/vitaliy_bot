"""Synthetic numeric-claim regression tests; no real ledger or meetings.

Coverage metadata and references are not verified, case-scoped numeric facts.
Until a trustworthy numeric source is actually bound to the pinned generation
and case, these tests require conservative rejection of its alleged values.
"""

import copy

import pytest

from pilot_projection_support import TOPIC, project_cards, projection
from test_hypotheses_pilot import inputs, cards
from tof_reliability.vitaly_bot import hypotheses_pilot as pilot


def approve(_):
    return {"supported": True, "confidence": "high"}


def check(claim, *, scope="Проверить повторные ремонты насосов", interests=None,
          field="decision"):
    card = {"management_question": "Почему повторяются ремонты насосов?",
            "decision": "Определить приоритет проверки", "hard_questions": []}
    if field == "hard_questions":
        card[field] = [claim]
    else:
        card[field] = claim
    pilot._validate_card_content(card, scope, TOPIC,
                                 interests or {"interests": []}, approve)


@pytest.mark.parametrize("claim", [
    "Экономия составила пятнадцать рублей.",
    "Экономия превысила пятнадцати рублей.",
    "Затраты составили одиннадцать рублей.",
    "Потери составили двенадцать рублей.",
    "Расход составил двадцать пять рублей.",
    "Расход составил сто двадцать три рубля.",
    "Расход составил две тысячи триста рублей.",
    "Расход составил 2 тыс. рублей.",
    "Расход составил 3 млн руб.",
    "Рост составил пятнадцать процентов.",
    "Рост составил 15%.",
    "Простой составил семнадцать часов.",
    "Заменено три насоса.",
])
@pytest.mark.parametrize("field", ["decision", "hard_questions"])
def test_invented_values_are_rejected_even_with_approving_verifier(claim, field):
    with pytest.raises(ValueError, match="(?i)numerical|numeric|числ"):
        check(claim, field=field)


@pytest.mark.parametrize("scope,claim", [
    ("Зафиксирован расход пятнадцать рублей.", "Зафиксирован расход пятнадцать рублей."),
    ("Зафиксирован простой 15 часов.", "Зафиксирован простой 15 часов."),
    ("Зафиксирован рост 15%.", "Зафиксирован рост 15%."),
    ("Зафиксирован расход две тысячи триста рублей.",
     "Зафиксирован расход две тысячи триста рублей."),
])
def test_identical_explicit_case_fact_can_be_used(scope, claim):
    check(claim, scope=scope)


@pytest.mark.parametrize("scope,claim", [
    ("Расход составил 15 рублей.", "Простой составил 15 часов."),
    ("Расход составил пятнадцать рублей.", "Простой составил пятнадцать часов."),
    ("Расход составил 15 рублей.", "Экономия составила 15 рублей."),
    ("Расход составил 15 тысяч рублей.", "Расход составил 15 миллионов рублей."),
    ("Проверены 2 насоса и 3 двигателя.", "Проверены 3 насоса и 2 двигателя."),
    ("Рост составил 15 процентов.", "Рост составил 15 рублей."),
    ("Расход составил двадцать рублей и пять копеек.", "Расход составил двадцать пять рублей."),
])
def test_numbers_do_not_authorize_different_units_amounts_or_context(scope, claim):
    with pytest.raises(ValueError, match="(?i)numerical|numeric|числ"):
        check(claim, scope=scope)


def test_historical_meeting_value_does_not_authorize_current_case():
    interests = {"interests": [{"id": "other-case", "requirement":
                               "Историческая экономия пятнадцать рублей"}]}
    with pytest.raises(ValueError, match="(?i)numerical|numeric|числ"):
        check("Экономия составила пятнадцать рублей.", interests=interests)


def test_ordinary_text_is_not_mistaken_for_numeric_claim():
    check("Выбрать ремонт по состоянию оборудования и повторным отказам.")


def test_coverage_and_other_case_cannot_whitelist_written_amount(inputs, cards):
    context = copy.deepcopy(inputs['interest_context'])
    other = copy.deepcopy(context['interests'][0])
    other['id'] = 'other-case'
    other['requirement'] = 'Другая встреча: пятнадцать рублей'
    other['evidence'][0].pop('fragment', None)
    context['interests'].append(other)
    coverage = copy.deepcopy(inputs['coverage'])
    coverage['routes'][0]['tables'] = {'repairs': {'count': 15}}
    cards[0]['decision'] = 'Экономия составила пятнадцать рублей.'
    with pytest.raises(ValueError, match='(?i)numerical|numeric|числ'):
        pilot.generate_pilot(scope='Проверить ремонты насосов', knowledge_version='k1',
                             interest_context=context, coverage_provider=lambda: coverage,
                             generator=lambda _: project_cards(copy.deepcopy(cards)),
                             **projection(context))


def test_structural_card_number_is_not_a_numeric_fact():
    card = {"number": 15, "management_question": "Почему повторяются ремонты насосов?",
            "decision": "Проверить данные", "hard_questions": []}
    pilot._validate_card_content(card, "Проверить ремонты насосов", TOPIC,
                                 {"interests": []}, approve)


@pytest.mark.parametrize('scope,claim', [
    ('Расход составил 15 рублей.', 'Расход составил 15 рублей и 20 копеек.'),
    ('Расход составил 15 рублей.', 'Расход не составил 15 рублей.'),
    ('Расход составил 15 рублей.', 'Расход составил 15 рублей ежегодно.'),
    ('Проверены 2 насоса и 3 двигателя.', 'Проверены 2 двигателя и 3 насоса.'),
])
def test_numeric_sentence_must_match_complete_case_statement(scope, claim):
    with pytest.raises(ValueError, match='(?i)numerical|numeric|числ'):
        check(claim, scope=scope)


@pytest.mark.parametrize('claim', [
    'Потери составили восемнадцати рублей.',
    'Простой составил сорока пяти часов.',
    'Проверено шестнадцать насосов.',
])
def test_additional_inflected_numerals_are_rejected(claim):
    with pytest.raises(ValueError, match='(?i)numerical|numeric|числ'):
        check(claim)


@pytest.mark.parametrize('claim', [
    'Обсудить ремонт в пятницу.',
    'Изучить состояние шестигранных болтов.',
    'Проверить семена повреждения покрытия.',
])
def test_non_numeric_words_with_numeral_prefixes_are_allowed(claim):
    check(claim)


@pytest.mark.parametrize('claim', [
    'В вымышленном случае простой насоса составил полтора часа.',
    'Проверены двое насосов.',
    'Простой превысил пятидесяти часов.',
    'Затраты составили двумястами рублями.',
    'Простой увеличился вдвое.',
])
def test_independent_review_identified_number_forms(claim):
    with pytest.raises(ValueError, match='(?i)numerical|numeric|числ'):
        check(claim)


def test_question_does_not_authorize_assertion():
    with pytest.raises(ValueError, match='(?i)numerical|numeric|числ'):
        check('Расход составил 15 рублей.', scope='Расход составил 15 рублей?')
