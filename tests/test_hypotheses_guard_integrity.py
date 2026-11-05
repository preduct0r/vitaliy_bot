"""Synthetic adversarial contracts for the hypothesis-card integrity guard.

The verifier deliberately approves everything; it is not proof of numeric facts.
No real user, meeting, ledger data or private verifier is used here.
"""

import copy

import pytest

from pilot_projection_support import TOPIC, project_cards, projection
from test_hypotheses_pilot import cards, inputs, pilot


def attacker_verifier(_):
    return {"supported": True, "confidence": "high"}


def generate(pilot, inputs, cards, *, scope=None):
    options = projection(inputs["interest_context"])
    options["private_verifier"] = attacker_verifier
    return pilot.generate_pilot(
        scope=inputs["scope"] if scope is None else scope,
        knowledge_version=inputs["knowledge_version"],
        interest_context=inputs["interest_context"],
        coverage_provider=lambda: copy.deepcopy(inputs["coverage"]),
        generator=lambda _: project_cards(copy.deepcopy(cards)),
        **options,
    )


def report(inputs):
    evidence = inputs["interest_context"]["interests"][0]["evidence"][0]
    return {"verified": True, "corpus_version": "corpus-1",
            "interest_version": "interests-1",
            "checks": [{**e, "interest_id": item["id"], "verified": True}
                       for item in inputs["interest_context"]["interests"] for e in item["evidence"]]}


@pytest.mark.parametrize("claim", [
    "Простой составил IV часа.",
    "Расход составил XV рублей.",
    "Это первый ремонт насоса.",
    "Обнаружен второй отказ насоса.",
    "Расход превысил сорока пяти рублей.",
    "Потери составили тремя тысячами рублей.",
    "Расход составил 2,5 млн рублей.",
])
@pytest.mark.parametrize("field", ["decision", "hard_questions"])
def test_invented_numeric_variants_are_rejected_with_approving_attacker(pilot, claim, field):
    card = {"management_question": "Почему повторяются ремонты насосов?",
            "decision": "Выбрать проверку", "hard_questions": ["Как проверить ремонт?"],
            "ranking": {key: 0 for key in pilot.SCORES}}
    card[field] = [claim] if field == "hard_questions" else claim
    with pytest.raises(ValueError):
        pilot._validate_card_content(card, "Проверить ремонты насосов", TOPIC,
                                     {"interests": []}, attacker_verifier)


@pytest.mark.parametrize("text", [
    "Проверить состояние шестигранных болтов.",
    "Обсудить ремонт в пятницу.",
    "Проверить римский профиль уплотнения.",
    "Проверить станцию насосов.",
])
def test_ordinary_words_are_not_numerical_claims(pilot, text):
    card = {"management_question": "Почему повторяются ремонты насосов?",
            "decision": text, "hard_questions": [], "ranking": {key: 0 for key in pilot.SCORES}}
    pilot._validate_card_content(card, "Проверить ремонты насосов", TOPIC,
                                 {"interests": []}, attacker_verifier)


@pytest.mark.parametrize("fact", [
    "Расход составил XV рублей.",
    "Простой составил 15 часов.",
    "Проверен первый насос.",
])
def test_exact_request_fact_is_preserved_by_legacy_content_check(pilot, fact):
    card = {"management_question": "Почему повторяются ремонты насосов?",
            "decision": fact, "hard_questions": [], "ranking": {key: 0 for key in pilot.SCORES}}
    pilot._validate_card_content(card, fact, TOPIC, {"interests": []}, attacker_verifier)


@pytest.mark.parametrize("location", ["card", "meeting_interest", "ranking"])
def test_unknown_fields_rejected_recursively_at_validation(pilot, inputs, cards, location):
    run = generate(pilot, inputs, cards)
    target = run["cards"][0] if location == "card" else run["cards"][0][location]
    target["attacker_extra"] = {"claim": "Экономия составила 900 рублей."}
    with pytest.raises(ValueError):
        pilot.validate_pilot(run, coverage=inputs["coverage"],
                             interest_context=inputs["interest_context"])


@pytest.mark.parametrize("location", ["card", "meeting_interest", "ranking"])
def test_unknown_fields_rejected_from_generator(pilot, inputs, cards, location):
    target = cards[0] if location == "card" else cards[0][location]
    target["attacker_extra"] = "Экономия составила 900 рублей."
    with pytest.raises(ValueError):
        generate(pilot, inputs, cards)


def test_generated_declared_meeting_metadata_is_accepted(pilot, inputs, cards):
    run = generate(pilot, inputs, cards)
    meeting = run["cards"][0]["meeting_interest"]
    assert {"interest_id", "topic_id", "reference", "attribution", "source_id",
            "date", "locator", "sha256"} <= set(meeting)
    pilot.validate_pilot(run, coverage=inputs["coverage"],
                         interest_context=inputs["interest_context"])


def test_matching_untrusted_scope_does_not_authorize_generation_without_binding(pilot, inputs, cards):
    fact = "Расход составил 900 рублей."
    cards[0]["decision"] = fact
    # The caller provided no independently trusted, persisted request context.
    with pytest.raises(ValueError):
        generate(pilot, inputs, cards, scope=fact)


def test_same_case_scope_and_claim_cannot_be_rewritten_before_validation(pilot, inputs, cards):
    run = generate(pilot, inputs, cards)
    fact = "Расход составил 900 рублей."
    run["scope"] = fact
    run["cards"][0]["decision"] = fact
    with pytest.raises(ValueError):
        pilot.validate_pilot(run, coverage=inputs["coverage"],
                             interest_context=inputs["interest_context"])


def test_same_case_scope_and_claim_cannot_be_rewritten_before_export(pilot, inputs, cards, tmp_path):
    run = generate(pilot, inputs, cards)
    fact = "Расход составил 900 рублей."
    run["scope"] = fact
    run["cards"][0]["decision"] = fact
    output = tmp_path / "blocked"
    with pytest.raises(ValueError):
        pilot.export_pilot(run, output_dir=output, source_verification=report(inputs))
    assert not output.exists()


def test_export_without_trusted_request_context_fails_closed(pilot, inputs, cards, tmp_path):
    run = generate(pilot, inputs, cards)
    output = tmp_path / "blocked"
    with pytest.raises(ValueError):
        pilot.export_pilot(run, output_dir=output, source_verification=report(inputs))
    assert not output.exists()


def test_ledger_reference_and_count_do_not_prove_numeric_fact(pilot, inputs, cards):
    inputs["coverage"]["routes"][0]["tables"] = {"repairs": {"count": 900}}
    cards[0]["decision"] = "Расход составил 900 рублей."
    with pytest.raises(ValueError):
        generate(pilot, inputs, cards)


@pytest.mark.parametrize("count", [4, 8])
def test_card_count_boundary_remains_enforced(pilot, inputs, cards, count):
    invalid = [dict(copy.deepcopy(cards[i % len(cards)]), number=i + 1)
               for i in range(count)]
    with pytest.raises(ValueError):
        generate(pilot, inputs, invalid)


@pytest.mark.parametrize("mutation", ["meeting", "capability"])
def test_existing_meeting_and_capability_bindings_remain_enforced(pilot, inputs, cards, mutation):
    run = generate(pilot, inputs, cards)
    if mutation == "meeting":
        run["cards"][0]["meeting_interest"]["reference"] = "other:chars:0-4"
    else:
        run["cards"][0]["capability_id"] = "invented-capability"
    with pytest.raises(ValueError):
        pilot.validate_pilot(run, coverage=inputs["coverage"],
                             interest_context=inputs["interest_context"])
