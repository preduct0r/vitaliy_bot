"""S3 contract. All inputs are synthetic; no network, bot or real corpus is used.

The offline entry points are generate_pilot(...), validate_pilot(...), and
export_pilot(...). The model and coverage providers are injectable callables.
The real-data pilot is a separate operator acceptance, not these unit tests.
"""

import copy
import importlib
import json

import pytest

from pilot_projection_support import LABEL, TOPIC, project_cards, projection


@pytest.fixture
def pilot():
    try:
        return importlib.import_module("tof_reliability.vitaly_bot.hypotheses_pilot")
    except ModuleNotFoundError as exc:
        if exc.name != "tof_reliability.vitaly_bot.hypotheses_pilot":
            raise

        class MissingPilot:
            def __getattr__(self, name):
                pytest.fail("S3 hypotheses pilot is not implemented: " + name)

        return MissingPilot()


@pytest.fixture
def inputs():
    return {
        "scope": "насосы цеха A",
        "knowledge_version": "knowledge-1",
        "interest_context": {"version": "interests-1", "corpus_version": "corpus-1", "interests": [
            {"id": "repeat", "requirement": "Изучить повторные ремонты",
             "attributions": ["interpretation"], "evidence": [
                 {"source_id": "synthetic", "date": "2026-10-01", "locator": "chars:0-4",
                  "attribution": "interpretation", "sha256": "a" * 64,
                  "reference": "synthetic:chars:0-4"}]}]},
        "coverage": {"coverage_version": 1, "generation": {"run_id": "gen-1"},
                     "routes": [{"capability_id": "AN-REPAIR-EPISODES",
                                 "availability": "partial",
                                 "missing_evidence": ["связь заказов с эпизодами"],
                                 "source_references": ["ledger:repairs"]}]},
    }


@pytest.fixture
def cards():
    return [{
        "number": i,
        "capability_id": "AN-REPAIR-EPISODES",
        "management_question": f"Какие причины повторного ремонта {chr(1040 + i)}?",
        "decision": "Определить приоритет проверки",
        "scope": "насосы цеха A",
        "metrics_method": "Проверить эпизоды по подтверждённым ключам",
        "meeting_interest": {"interest_id": "repeat", "attribution": "interpretation",
                             "reference": "synthetic:chars:0-4"},
        "source_references": ["ledger:repairs"],
        "source_readiness": "partial",
        "missing_evidence": ["связь заказов с эпизодами"],
        "falsification": "Повторы не отличаются от базового уровня",
        "hard_questions": ["Как определён эпизод?"],
        "analyst_brief": "Проверить ключи и период до расчёта",
        "novelty": "unknown",
        "ranking": {"utility": 5, "feasibility": 3, "interest": 4, "novelty": 0},
    } for i in range(1, 6)]


def generate(pilot, inputs, cards, model=None, coverage=None):
    return pilot.generate_pilot(
        scope=inputs["scope"], knowledge_version=inputs["knowledge_version"],
        interest_context=inputs["interest_context"],
        coverage_provider=coverage or (lambda: copy.deepcopy(inputs["coverage"])),
        generator=model or (lambda context: copy.deepcopy(project_cards(cards))),
        **projection(inputs["interest_context"]),
    )


def test_generation_without_files_preserves_versions_and_evidence(pilot, inputs, cards):
    run = generate(pilot, inputs, cards)
    assert len(run["cards"]) == 5
    assert run["generation_id"] == "gen-1"
    assert run["knowledge_version"] == "knowledge-1"
    assert run["interest_version"] == "interests-1"
    assert all(c["scope"] == inputs["scope"] for c in run["cards"])
    assert all(c["source_readiness"] == "partial" and c["missing_evidence"] for c in run["cards"])
    assert all(c["novelty"] == "unknown" for c in run["cards"])
    pilot.validate_pilot(run, coverage=inputs["coverage"], interest_context=inputs["interest_context"])


@pytest.mark.parametrize("field", [
    "management_question", "decision", "scope", "metrics_method", "meeting_interest",
    "capability_id", "source_references", "source_readiness", "missing_evidence", "falsification",
    "hard_questions", "analyst_brief", "novelty", "ranking",
])
def test_missing_required_card_field_is_rejected(pilot, inputs, cards, field):
    run = generate(pilot, inputs, cards)
    del run["cards"][0][field]
    with pytest.raises(ValueError):
        pilot.validate_pilot(run, coverage=inputs["coverage"], interest_context=inputs["interest_context"])


@pytest.mark.parametrize("count", [0, 4, 8])
def test_card_count_is_bounded(pilot, inputs, cards, count):
    invalid = [dict(cards[i % len(cards)], number=i + 1) for i in range(count)]
    with pytest.raises(ValueError):
        generate(pilot, inputs, invalid)


def test_fabricated_readiness_reference_and_interest_are_rejected(pilot, inputs, cards):
    for changes in (
        {"source_references": ["ledger:invented"]},
        {"meeting_interest": {"interest_id": "invented", "attribution": "direct_request", "reference": "x"}},
        {"meeting_interest": {"interest_id": "repeat", "attribution": "direct_request", "reference": "x"}},
    ):
        invalid = copy.deepcopy(cards)
        invalid[0].update(changes)
        with pytest.raises(ValueError):
            generate(pilot, inputs, invalid)


def test_server_failure_stops_before_model_without_local_fallback(pilot, inputs, cards):
    calls = []
    def outage():
        raise RuntimeError("synthetic server unavailable")
    with pytest.raises(Exception, match="(?i)unavailable|недоступ"):
        generate(pilot, inputs, cards, model=lambda ctx: calls.append(ctx) or cards, coverage=outage)
    assert calls == []


def test_generator_receives_only_generalized_interests_and_no_holdout(pilot, inputs, cards):
    inputs["interest_context"]["private_fragment"] = "PRIVATE_SENTINEL"
    inputs["interest_context"]["expected_topic_ids"] = ["HOLDOUT_SENTINEL"]
    seen = []
    generate(pilot, inputs, cards, model=lambda ctx: seen.append(ctx) or copy.deepcopy(project_cards(cards)))
    payload = json.dumps(seen, ensure_ascii=False)
    assert "PRIVATE_SENTINEL" not in payload
    assert "HOLDOUT_SENTINEL" not in payload
    assert "Изучить повторные ремонты" not in payload
    assert LABEL in payload and TOPIC in payload


def test_export_is_reproducible_and_preserves_verification(pilot, inputs, cards, tmp_path):
    run = generate(pilot, inputs, cards)
    evidence = inputs['interest_context']['interests'][0]['evidence'][0]
    verification = {"verified": True, "corpus_version": "corpus-1", "interest_version": "interests-1",
                    "checks": [{**evidence, "interest_id": "repeat", "verified": True}]}
    first, second = tmp_path / "one", tmp_path / "two"
    pilot.export_pilot(run, output_dir=first, source_verification=verification)
    pilot.export_pilot(run, output_dir=second, source_verification=verification)
    files = lambda root: {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert files(first) and files(first) == files(second)
    assert b'"verified"' in b"\n".join(files(first).values())


def test_missing_or_failed_source_verification_is_not_accepted_as_pilot(pilot, inputs, cards, tmp_path):
    run = generate(pilot, inputs, cards)
    for report in (None, {"verified": False, "checks": []}):
        with pytest.raises((ValueError, RuntimeError)):
            pilot.export_pilot(run, output_dir=tmp_path / "blocked", source_verification=report)


def test_s2_table_references_are_derived_and_coverage_is_snapshotted(pilot, inputs, cards):
    route = inputs['coverage']['routes'][0]
    route.pop('source_references')
    route['tables'] = {'repairs': {'count': 2, 'source_document_ids': ['doc']}}
    run = generate(pilot, inputs, cards)
    assert run['coverage'] == inputs['coverage']
    inputs['coverage']['routes'][0]['missing_evidence'].append('later')
    assert 'later' not in run['coverage']['routes'][0]['missing_evidence']


def test_verification_checks_must_be_nonempty_and_all_verified(pilot, inputs, cards, tmp_path):
    run = generate(pilot, inputs, cards)
    for report in ({'verified': True, 'checks': []},
                   {'verified': True, 'checks': [{'verified': False}]}):
        with pytest.raises((ValueError, RuntimeError)):
            pilot.export_pilot(run, output_dir=tmp_path / 'bad', source_verification=report)


def test_cards_are_ranked_and_renumbered_by_scores(pilot, inputs, cards):
    cards[4]['ranking']['utility'] = 1
    cards[1]['ranking']['utility'] = 2
    run = generate(pilot, inputs, cards)
    assert [c['management_question'] for c in run['cards']] == [cards[i]['management_question'] for i in (0, 2, 3, 1, 4)]
    assert [c['number'] for c in run['cards']] == list(range(1, 6))


def test_runner_adapter_uses_structured_schema(pilot, inputs, cards):
    class Result:
        value = {'cards': cards}
    class Runner:
        async def run(self, prompt, schema):
            assert 'Полнота' not in prompt
            assert schema['properties']['cards']['minItems'] == 5
            return Result()
    generated = pilot.codex_generator(Runner())({'scope': 'насосы'})
    assert generated == cards
