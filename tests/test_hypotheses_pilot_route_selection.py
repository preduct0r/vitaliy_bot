"""Synthetic contracts for selecting an exact published coverage route."""

import copy

import pytest

from pilot_projection_support import LABEL, TOPIC, distinct_directions, project_cards, projection

from tof_reliability.vitaly_bot import hypotheses_pilot as pilot
from test_hypotheses_pilot import inputs as legacy_inputs, cards as legacy_cards


@pytest.fixture
def inputs():
    data = legacy_inputs.__wrapped__()
    coverage = data['coverage']
    for route in coverage['routes']:
        route['missing_evidence'] = ['связь заказов с эпизодами']
    coverage['routes'][1]['source_references'].append('ledger:orders')
    coverage['routes'].append({'capability_id': 'AN-TECHCARD-LABOR', 'availability': 'unavailable',
                               'missing_evidence': ['методика'], 'source_references': []})
    return data['interest_context'], coverage, legacy_cards.__wrapped__()


def generate(interests, coverage, cards):
    return pilot.generate_pilot(
        scope="насосы", knowledge_version="knowledge-1", interest_context=interests,
        coverage_provider=lambda: copy.deepcopy(coverage),
        generator=lambda _: copy.deepcopy(project_cards(cards)),
        **projection(interests),
    )


def test_shared_table_uses_selected_route_and_preserves_meeting_reference(inputs):
    interests, coverage, cards = inputs
    run = generate(interests, coverage, cards)
    assert [card["capability_id"] for card in run["cards"]] == [card["capability_id"] for card in cards]
    assert all(card["source_readiness"] == "partial" for card in run["cards"])
    assert all(card["missing_evidence"] == ["связь заказов с эпизодами"] for card in run["cards"])
    for card in run["cards"]:
        item = next(i for i in interests["interests"] if i["id"] == card["meeting_interest"]["interest_id"])
        assert all(card["meeting_interest"][k] == v for k, v in item["evidence"][0].items() if k != "fragment")
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
