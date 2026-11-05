"""Synthetic regression contracts for source-bound, offline hypothesis cards."""

import copy
import hashlib
import json

import pytest

from pilot_projection_support import LABEL, TOPIC, distinct_directions, project_cards, projection
from test_hypotheses_pilot_evidence_resolve import inputs as resolved_inputs

from tof_reliability.vitaly_bot import hypotheses_interests as interests
from tof_reliability.vitaly_bot import hypotheses_pilot as pilot


@pytest.fixture
def evidence(tmp_path):
    text = "PRIVATE_FRAGMENT: synthetic meeting evidence"
    source = tmp_path / "meeting.txt"
    source.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    index = {"version": "corpus-1", "sources": [
        {"id": "meeting-a", "date": "2026-10-01", "kind": "transcript",
         "sha256": digest, "path": str(source)},
        {"id": "meeting-b", "date": "2026-10-02", "kind": "transcript",
         "sha256": digest, "path": str(source)},
    ]}
    profile = {"version": "interests-1", "corpus_version": "corpus-1", "interests": [
        {"id": "repeat", "requirement": "Проверить повторные ремонты", "evidence": [
            {"source_id": "meeting-a", "locator": "chars:0-16", "fragment": text[:16],
             "attribution": "interpretation"}]},
        {"id": "labor", "requirement": "Сравнить норму и измеренную трудоёмкость", "evidence": [
            {"source_id": "meeting-b", "locator": "chars:0-16", "fragment": text[:16],
             "attribution": "direct_request"}]},
    ]}
    split = {"version": index["version"], "training": index["sources"],
             "holdout": [{"id": "HOLDOUT_SENTINEL", "text": "HOLDOUT_PRIVATE"}]}
    return index, profile, split


@pytest.fixture
def coverage():
    return {"coverage_version": 1, "generation": {"run_id": "gen-1"}, "routes": [
        {"capability_id": "AN-REPAIR-EPISODES", "availability": "partial",
         "missing_evidence": ["связь заказов с эпизодами"],
         "source_references": ["ledger:repairs"]},
        {"capability_id": "scenario_effect", "availability": "unavailable",
         "missing_evidence": ["методика сценария", "ресурсные ограничения"],
         "source_references": ["capability:scenario_effect"]},
    ]}


@pytest.fixture
def source_context(evidence):
    index, profile, _ = evidence
    # This safe context represents two independently verified evidence entries.
    return {"version": profile["version"], "corpus_version": index["version"],
            "interests": [
                {"id": "repeat", "requirement": "Проверить повторные ремонты",
                 "attributions": ["interpretation"], "evidence": [
                     {"source_id": "meeting-a", "date": "2026-10-01",
                      "locator": "chars:0-16", "attribution": "interpretation",
                      "sha256": index["sources"][0]["sha256"],
                      "reference": "meeting-a:chars:0-16"}]},
                {"id": "labor", "requirement": "Сравнить норму и измеренную трудоёмкость",
                 "attributions": ["direct_request"], "evidence": [
                     {"source_id": "meeting-b", "date": "2026-10-02",
                      "locator": "chars:0-16", "attribution": "direct_request",
                      "sha256": index["sources"][1]["sha256"],
                      "reference": "meeting-b:chars:0-16"}]},
            ]}


def make_run(coverage, context):
    cards = [{"number": n, "capability_id": "AN-REPAIR-EPISODES", "management_question": f"Что проверить в ремонтах {chr(1040 + n)}?",
              "decision": "Выбрать проверку", "scope": "насосы", "metrics_method": "Проверить ключи",
              "meeting_interest": {"interest_id": "repeat", "attribution": "interpretation",
                                   "reference": "meeting-a:chars:0-16"},
              "source_references": ["ledger:repairs"], "source_readiness": "partial",
              "missing_evidence": ["связь заказов с эпизодами"],
              "falsification": "Повторов не больше базового уровня",
              "hard_questions": ["Как определён эпизод?"], "analyst_brief": "Проверить данные",
              "novelty": "unknown", "ranking": {"utility": 3, "feasibility": 3,
                                                     "interest": 3, "novelty": 0}}
             for n in range(1, 6)]
    distinct_directions(cards)
    return {"generation_id": "gen-1", "knowledge_version": "knowledge-1",
            "interest_version": context["version"], "coverage": coverage,
            "interest_context": context, "cards": cards}


def test_projection_contains_only_training_source_identifiers_and_safe_locators(evidence):
    index, profile, split = evidence
    assert interests.assess_provenance(profile, index)["verified"]
    context = interests.generator_context(split, profile)
    payload = json.dumps(context, ensure_ascii=False)
    for expected in ("meeting-a", "meeting-b", "2026-10-01", "chars:0-16"):
        assert expected in payload
    for secret in ("PRIVATE_FRAGMENT", str(index["sources"][0]["path"]), "HOLDOUT_SENTINEL", "HOLDOUT_PRIVATE"):
        assert secret not in payload


@pytest.mark.parametrize("reference,attribution", [
    ("invented meeting reference", "interpretation"),
    ("meeting-b:chars:0-16", "interpretation"),
    ("meeting-a:chars:99-100", "interpretation"),
    ("meeting-a:chars:0-16", "direct_request"),
])
def test_rejects_forged_or_misattributed_meeting_reference(coverage, source_context, reference, attribution):
    run = make_run(coverage, source_context)
    run["cards"][0]["meeting_interest"].update(reference=reference, attribution=attribution)
    with pytest.raises(ValueError):
        pilot.validate_pilot(run, coverage=coverage, interest_context=source_context)


def test_matching_verified_reference_is_accepted(coverage, source_context):
    pilot.validate_pilot(make_run(coverage, source_context), coverage=coverage, interest_context=source_context)


@pytest.mark.parametrize("mutation", ["missing_check", "wrong_source", "wrong_locator", "wrong_version", "failed_check"])
def test_export_rejects_unrelated_or_incomplete_provenance(coverage, source_context, tmp_path, mutation):
    run = make_run(coverage, source_context)
    checks = [{"interest_id": "repeat", "source_id": "meeting-a", "locator": "chars:0-16", "verified": True},
              {"interest_id": "labor", "source_id": "meeting-b", "locator": "chars:0-16", "verified": True}]
    report = {"verified": True, "corpus_version": "corpus-1", "interest_version": "interests-1", "checks": checks}
    if mutation == "missing_check":
        checks.pop(0)
    elif mutation == "wrong_source":
        checks[0]["source_id"] = "meeting-b"
    elif mutation == "wrong_locator":
        checks[0]["locator"] = "chars:1-16"
    elif mutation == "wrong_version":
        report["corpus_version"] = "old-corpus"
    else:
        checks[0]["verified"] = False
    with pytest.raises((ValueError, RuntimeError)):
        pilot.export_pilot(run, output_dir=tmp_path / "out", source_verification=report)


def test_changed_source_bytes_invalidate_provenance(evidence):
    index, profile, _ = evidence
    from pathlib import Path
    Path(index["sources"][0]["path"]).write_text("tampered", encoding="utf-8")
    assert not interests.assess_provenance(profile, index)["verified"]


def test_projection_rejects_unverified_evidence_and_unsafe_ids(evidence):
    index, profile, split = evidence
    profile["interests"][0]["evidence"][0]["fragment"] = "forged"
    with pytest.raises(ValueError):
        interests.generator_context(split, profile)
    profile["interests"][0]["evidence"][0]["fragment"] = "PRIVATE_FRAGMENT"
    split["training"][0]["id"] = "private text /path"
    profile["interests"][0]["evidence"][0]["source_id"] = "private text /path"
    with pytest.raises(ValueError):
        interests.generator_context(split, profile)


def test_export_binds_verified_hash_attribution_and_context(resolved_inputs, tmp_path):
    context, report, coverage, cards, _, _ = resolved_inputs
    store = pilot.TrustedRequestStore(tmp_path / 'trusted.sqlite')
    trusted = store.create({'run_id': 'run', 'case_id': 'case', 'user_id': 'user', 'scope': ''})
    run = pilot.generate_pilot(scope='', knowledge_version='knowledge-1', interest_context=context,
                               coverage_provider=lambda: copy.deepcopy(coverage),
                               generator=lambda _: project_cards(copy.deepcopy(cards)),
                               trusted_context=trusted, trusted_store=store, **projection(context))
    pilot.export_pilot(run, output_dir=tmp_path / "good", source_verification=report, trusted_context=trusted, trusted_store=store)
    for field, value in (("sha256", "a" * 64), ("attribution", "direct_request"),
                         ("date", "2026-10-03")):
        altered = copy.deepcopy(run)
        altered["interest_context"]["interests"][0]["evidence"][0][field] = value
        with pytest.raises(ValueError):
            pilot.export_pilot(altered, output_dir=tmp_path / field, source_verification=report, trusted_context=trusted, trusted_store=store)


def test_model_reference_must_match_selected_evidence(coverage, source_context):
    cards = make_run(coverage, source_context)["cards"]
    cards[0]["meeting_interest"]["reference"] = "meeting-b:chars:0-16"
    with pytest.raises(ValueError):
        pilot.generate_pilot(scope="", knowledge_version="k", interest_context=source_context,
                             coverage_provider=lambda: coverage, generator=lambda _: project_cards(cards), **projection(source_context))


def test_pilot_training_index_excludes_holdout_even_if_index_contains_it(evidence):
    index, profile, _ = evidence
    held = {**index['sources'][0], 'id': 'held', 'date': '2026-10-26'}
    index['sources'].append(held)
    training = interests.pilot_training_index(index)
    assert {s['id'] for s in training['sources']} == {'meeting-a', 'meeting-b'}
    profile['interests'][0]['evidence'][0]['source_id'] = 'held'
    with pytest.raises(ValueError):
        interests.generator_context({'version': index['version'], 'training': training['sources']}, profile)


@pytest.mark.parametrize("changes", [
    {"source_references": ["ledger:invented"], "source_readiness": "available", "missing_evidence": []},
    {"source_references": ["capability:scenario_effect"], "source_readiness": "available", "missing_evidence": []},
    {"source_references": ["capability:scenario_effect"], "source_readiness": "unavailable", "missing_evidence": ["методика сценария"]},
    {"source_references": ["ledger:repairs"], "source_readiness": "partial", "missing_evidence": []},
])
def test_coverage_cannot_claim_unavailable_sources_or_omit_gaps(coverage, source_context, changes):
    run = make_run(coverage, source_context)
    run["cards"][0].update(changes)
    with pytest.raises(ValueError):
        pilot.validate_pilot(run, coverage=coverage, interest_context=source_context)


def test_generation_requests_distinct_analytical_questions_and_preserves_schema(resolved_inputs):
    source_context, _, coverage, cards, _, _ = resolved_inputs
    seen = {}

    class Runner:
        async def run(self, prompt, schema):
            seen.update(prompt=prompt, schema=schema)
            return type("Result", (), {"value": {"cards": copy.deepcopy(project_cards(cards))}})()

    pilot.generate_pilot(scope="насосы", knowledge_version="knowledge-1", interest_context=source_context,
                         coverage_provider=lambda: coverage, generator=pilot.codex_generator(Runner()), **projection(source_context))
    prompt = seen["prompt"].casefold()
    for term in ("качеств", "техкарт", "трудо", "ресурс", "различ"):
        assert term in prompt
    schema = seen["schema"]["properties"]["cards"]
    assert schema["minItems"] == 5 and schema["maxItems"] == 7
    assert "meeting_interest" in schema["items"]["required"]
    assert "HOLDOUT_PRIVATE" not in seen["prompt"]
