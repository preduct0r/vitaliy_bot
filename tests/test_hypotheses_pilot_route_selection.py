"""Synthetic contracts for selecting an exact published coverage route."""

import copy

import pytest

from tof_reliability.vitaly_bot import hypotheses_pilot as pilot


@pytest.fixture
def inputs():
    evidence = {
        "source_id": "meeting-a", "date": "2026-10-01",
        "locator": "chars:0-4", "attribution": "interpretation",
        "sha256": "a" * 64, "reference": "meeting-a:chars:0-4",
    }
    interests = {"version": "interests-1", "corpus_version": "corpus-1", "interests": [
        {"id": "repeat", "requirement": "Проверить повторы",
         "attributions": ["interpretation"], "evidence": [evidence]},
    ]}
    coverage = {"coverage_version": 1, "generation": {"run_id": "gen-1"}, "routes": [
        {"capability_id": "repair-episodes", "availability": "partial",
         "missing_evidence": ["связь заказов с эпизодами"],
         "source_references": ["ledger:repairs"]},
        {"capability_id": "repair-list", "availability": "available",
         "missing_evidence": [], "source_references": ["ledger:repairs", "ledger:orders"]},
        {"capability_id": "scenario-effect", "availability": "unavailable",
         "missing_evidence": ["методика сценария"],
         "source_references": ["capability:scenario-effect"]},
    ]}
    cards = [{
        "number": n, "capability_id": "repair-episodes",
        "management_question": f"Как проверить повторы {n}?",
        "decision": "Выбрать проверку", "scope": "насосы",
        "metrics_method": "Проверить ключи эпизодов",
        "meeting_interest": {"interest_id": "repeat", "attribution": "interpretation",
                             "reference": evidence["reference"]},
        "source_references": ["ledger:repairs"], "source_readiness": "partial",
        "missing_evidence": ["связь заказов с эпизодами"],
        "falsification": "Повторы не отличаются от базового уровня",
        "hard_questions": ["Как определён эпизод?"],
        "analyst_brief": "Проверить ключи", "novelty": "unknown",
        "ranking": {"utility": 5, "feasibility": 3, "interest": 4, "novelty": 0},
    } for n in range(1, 6)]
    return interests, coverage, cards


def generate(interests, coverage, cards):
    return pilot.generate_pilot(
        scope="насосы", knowledge_version="knowledge-1", interest_context=interests,
        coverage_provider=lambda: copy.deepcopy(coverage),
        generator=lambda _: copy.deepcopy(cards),
    )


def test_shared_table_uses_selected_route_and_preserves_meeting_reference(inputs):
    interests, coverage, cards = inputs
    run = generate(interests, coverage, cards)
    assert all(card["capability_id"] == "repair-episodes" for card in run["cards"])
    assert all(card["source_readiness"] == "partial" for card in run["cards"])
    assert all(card["missing_evidence"] == ["связь заказов с эпизодами"] for card in run["cards"])
    assert all(card["meeting_interest"] == cards[0]["meeting_interest"] for card in run["cards"])
    pilot.validate_pilot(run, coverage=coverage, interest_context=interests)


def test_readiness_and_missing_evidence_are_derived_from_selected_route(inputs):
    interests, coverage, cards = inputs
    for card in cards:
        card["source_readiness"] = "available"
        card["missing_evidence"] = []
    run = generate(interests, coverage, cards)
    assert all(card["source_readiness"] == "partial" for card in run["cards"])
    assert all(card["missing_evidence"] == ["связь заказов с эпизодами"] for card in run["cards"])


@pytest.mark.parametrize("mutation", ["unknown", "missing", "wrong_route_reference", "duplicate_route"])
def test_unknown_ambiguous_or_mismatched_route_is_rejected(inputs, mutation):
    interests, coverage, cards = inputs
    if mutation == "unknown":
        cards[0]["capability_id"] = "invented"
    elif mutation == "missing":
        del cards[0]["capability_id"]
    elif mutation == "wrong_route_reference":
        cards[0]["source_references"] = ["ledger:orders"]
    else:
        coverage["routes"].append(copy.deepcopy(coverage["routes"][0]))
    with pytest.raises((ValueError, RuntimeError)):
        generate(interests, coverage, cards)


def test_schema_requires_exact_capability_id(inputs):
    _, _, cards = inputs
    captured = {}

    class Runner:
        async def run(self, prompt, schema):
            captured.update(prompt=prompt, schema=schema)
            return type("Result", (), {"value": {"cards": cards}})()

    pilot.codex_generator(Runner())({"scope": "насосы"})
    item = captured["schema"]["properties"]["cards"]["items"]
    assert "capability_id" in item["required"]
    assert item["properties"]["capability_id"]["type"] == "string"


def test_model_failure_includes_classified_detail():
    from tof_reliability.vitaly_bot.models import ModelError

    class Runner:
        async def run(self, prompt, schema):
            raise ModelError("model_timeout", "limit_seconds=42")

    with pytest.raises(RuntimeError, match="model_timeout.*limit_seconds=42"):
        pilot.codex_generator(Runner())({"scope": "насосы"})
