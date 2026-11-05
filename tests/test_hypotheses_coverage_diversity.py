"""Synthetic development regressions; no historical meeting contents are used."""
import copy
import json

import pytest

from tof_reliability.vitaly_bot import hypotheses_pilot as pilot
from test_hypotheses_pilot import inputs as legacy_inputs, cards as legacy_cards
from pilot_projection_support import projection


@pytest.fixture
def synthetic_case():
    data = legacy_inputs.__wrapped__()
    interests = data['interest_context']
    interests['interests'][0]['requirement'] += ' PRIVATE_SOURCE_SECRET 987654321'
    interests['interests'][0]['fragment'] = 'PRIVATE_TRANSCRIPT_SENTINEL'
    projected = projection(interests)
    coverage = data['coverage']
    coverage['routes'].append({'capability_id': 'AN-TECHCARD-LABOR', 'availability': 'unavailable',
                               'missing_evidence': ['фактическая трудоёмкость'], 'source_references': []})
    cards = legacy_cards.__wrapped__()
    for card in cards:
        card['decision'] = 'Выбрать проверку причин повторных ремонтов'
        card['metrics_method'] = 'Сопоставить ключи ремонтных эпизодов'
    return interests, projected['verified_labels'], projected['verifier_audit'], coverage, cards


def generate(case, generator, verifier=None):
    interests, labels, audit, coverage, _ = case
    return pilot.generate_pilot(
        scope='насосы', knowledge_version='k1', interest_context=interests,
        verified_labels=labels, verifier_audit=audit,
        coverage_provider=lambda: copy.deepcopy(coverage), generator=generator,
        private_verifier=verifier or (lambda _: {'supported': True, 'confidence': 'high'}))


def test_repeated_decision_and_method_do_not_count_as_five_directions(synthetic_case):
    cards = synthetic_case[-1]
    with pytest.raises(ValueError, match='(?i)divers|distinct|duplicate|различ|повтор'):
        generate(synthetic_case, lambda _: copy.deepcopy(cards))


def test_rewording_questions_alone_does_not_create_diverse_decisions(synthetic_case):
    cards = copy.deepcopy(synthetic_case[-1])
    for card, question in zip(cards, [
        'Как проверить причины повторного ремонта насосов?',
        'Какие причины повторных ремонтов насосов следует проверить?',
        'Почему повторяются ремонты насосов?',
        'Как исследовать повторные ремонтные эпизоды насосов?',
        'Что вызывает повторный ремонт насосов?',
    ]):
        card['management_question'] = question
    with pytest.raises(ValueError, match='(?i)divers|distinct|duplicate|различ|повтор'):
        generate(synthetic_case, lambda _: cards)


def test_generator_is_asked_to_explore_distinct_decisions_and_data_gaps(synthetic_case):
    seen = []
    def capture(context):
        seen.append(context)
        raise RuntimeError('stop after capturing synthetic prompt')
    with pytest.raises(RuntimeError, match='stop after capturing'):
        generate(synthetic_case, capture)
    payload = json.dumps(seen[0], ensure_ascii=False).casefold()
    instructions = seen[0]['instructions'].casefold()
    for stem in ('техкарт', 'трудо', 'ресурс', 'материал', 'реестр', 'провер', 'решени', 'метод'):
        assert stem in instructions
    assert 'фактическая трудоёмкость' in payload and 'unavailable' in payload
    for secret in ('PRIVATE_SOURCE_SECRET', '987654321', 'PRIVATE_TRANSCRIPT_SENTINEL',
                   'synthetic-source', 'chars:0-10'):
        assert secret.casefold() not in payload


def test_unsupported_historical_amount_is_rejected_even_if_verifier_approves(synthetic_case):
    cards = copy.deepcopy(synthetic_case[-1])
    cards[0]['analyst_brief'] = 'В прошлом на ремонт потрачено 987654321 рублей.'
    with pytest.raises(ValueError, match='(?i)numeric|private|числ|факт'):
        generate(synthetic_case, lambda _: cards)


def test_cross_route_source_cannot_prove_missing_labor_data(synthetic_case):
    cards = copy.deepcopy(synthetic_case[-1])
    cards[0].update(capability_id='AN-TECHCARD-LABOR', source_references=['ledger:repairs'])
    with pytest.raises(ValueError, match='(?i)reference|source|route|источник'):
        generate(synthetic_case, lambda _: cards)

@pytest.mark.parametrize('field', ['decision', 'metrics_method'])
def test_each_decision_and_method_must_be_distinct(synthetic_case, field):
    from pilot_projection_support import distinct_directions
    cards = distinct_directions(copy.deepcopy(synthetic_case[-1]))
    cards[1][field] = cards[0][field]
    with pytest.raises(ValueError, match='(?i)divers|distinct|duplicate|различ|повтор'):
        generate(synthetic_case, lambda _: cards)
