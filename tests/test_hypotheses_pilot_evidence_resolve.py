"""Regression contracts for resolving model-selected interests to verified evidence."""

import copy
import hashlib
import json

import pytest

from tof_reliability.vitaly_bot import hypotheses_interests as interests
from tof_reliability.vitaly_bot import hypotheses_pilot as pilot


@pytest.fixture
def inputs(tmp_path):
    text = "SYNTHETIC_PRIVATE_MEETING_CONTENT"
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
        {"id": "repeat", "requirement": "Проверить повторы", "evidence": [
            {"source_id": "meeting-a", "locator": "chars:0-9", "fragment": text[:9],
             "attribution": "interpretation"},
            {"source_id": "meeting-b", "locator": "chars:10-17", "fragment": text[10:17],
             "attribution": "direct_request"}]},
        {"id": "labor", "requirement": "Проверить трудоёмкость", "evidence": [
            {"source_id": "meeting-b", "locator": "chars:0-9", "fragment": text[:9],
             "attribution": "direct_request"}]},
    ]}
    split = {"version": "corpus-1", "training": index["sources"], "holdout": []}
    context = interests.generator_context(split, profile)
    report = interests.assess_provenance(profile, index)
    assert report["verified"]
    coverage = {"coverage_version": 1, "generation": {"run_id": "gen-1"}, "routes": [
        {"capability_id": "repairs", "availability": "partial",
         "missing_evidence": ["ключи эпизодов"], "source_references": ["ledger:repairs"]}]}
    cards = [{"number": n, "capability_id": "repairs",
              "management_question": f"Как проверить повторы {n}?", "decision": "Выбрать проверку",
              "scope": "насосы", "metrics_method": "Проверить ключи",
              "meeting_interest": {"interest_id": "repeat"},
              "source_references": ["ledger:repairs"], "source_readiness": "partial",
              "missing_evidence": ["ключи эпизодов"], "falsification": "Нет различий",
              "hard_questions": ["Как определён эпизод?"], "analyst_brief": "Проверить данные",
              "novelty": "unknown", "ranking": {"utility": 3, "feasibility": 3,
                                                   "interest": 3, "novelty": 0}}
             for n in range(1, 6)]
    return context, report, coverage, cards, text, str(source)


def generate(context, coverage, cards):
    return pilot.generate_pilot(scope="насосы", knowledge_version="knowledge-1",
                                interest_context=context, coverage_provider=lambda: copy.deepcopy(coverage),
                                generator=lambda _: copy.deepcopy(cards))


def test_id_only_resolves_exact_evidence_deterministically_and_exports(inputs, tmp_path):
    context, report, coverage, cards, secret, path = inputs
    cards[1]["meeting_interest"]["interest_id"] = "labor"
    first = generate(context, coverage, cards)
    second = generate(context, coverage, cards)
    assert first == second
    by_id = {item["id"]: item for item in context["interests"]}
    for card in first["cards"]:
        meeting = card["meeting_interest"]
        entries = by_id[meeting["interest_id"]]["evidence"]
        assert any(all(meeting.get(key) == entry[key] for key in
                       ("reference", "date", "attribution", "source_id", "locator", "sha256"))
                   for entry in entries)
    pilot.validate_pilot(first, coverage=coverage, interest_context=context)
    output = tmp_path / "export"
    pilot.export_pilot(first, output_dir=output, source_verification=report)
    exported = (output / "pilot.json").read_text(encoding="utf-8")
    assert "meeting-a:chars:0-9" in exported or "meeting-b:chars:10-17" in exported
    assert "meeting-b:chars:0-9" in exported
    assert secret not in exported and path not in exported


def test_schema_requests_only_interest_id(inputs):
    context, _, coverage, cards, secret, path = inputs
    seen = {}

    class Runner:
        async def run(self, prompt, schema):
            seen.update(prompt=prompt, schema=schema)
            return type("Result", (), {"value": {"cards": copy.deepcopy(cards)}})()

    pilot.generate_pilot(scope="насосы", knowledge_version="knowledge-1", interest_context=context,
                         coverage_provider=lambda: coverage, generator=pilot.codex_generator(Runner()))
    meeting = seen["schema"]["properties"]["cards"]["items"]["properties"]["meeting_interest"]
    assert set(meeting["properties"]) == {"interest_id"}
    assert meeting["required"] == ["interest_id"]
    assert meeting["additionalProperties"] is False
    assert secret not in seen["prompt"] and path not in seen["prompt"]


@pytest.mark.parametrize("invalid", ["unknown", "", None])
def test_unknown_or_empty_interest_is_never_replaced(inputs, invalid):
    context, _, coverage, cards, _, _ = inputs
    cards[0]["meeting_interest"]["interest_id"] = invalid
    with pytest.raises((ValueError, RuntimeError)):
        generate(context, coverage, cards)


def test_selected_interest_without_evidence_fails(inputs):
    context, _, coverage, cards, _, _ = inputs
    context["interests"][0]["evidence"] = []
    with pytest.raises((ValueError, RuntimeError)):
        generate(context, coverage, cards)


@pytest.mark.parametrize("field,value", [
    ("reference", "meeting-b:chars:0-9"), ("attribution", "direct_request"),
    ("source_id", "meeting-b"), ("date", "2026-10-03"),
    ("locator", "chars:99-100"), ("sha256", "0" * 64),
])
def test_resolved_binding_rejects_tampered_metadata(inputs, field, value):
    context, _, coverage, cards, _, _ = inputs
    run = generate(context, coverage, cards)
    meeting = run["cards"][0]["meeting_interest"]
    # Ensure the mutation changes the actual selected entry.
    if meeting.get(field) == value:
        value = "1" * 64 if field == "sha256" else "forged"
    meeting[field] = value
    with pytest.raises(ValueError):
        pilot.validate_pilot(run, coverage=coverage, interest_context=context)


def test_existing_valid_reference_format_remains_supported_and_forgery_rejected(inputs):
    context, _, coverage, cards, _, _ = inputs
    entry = context["interests"][0]["evidence"][0]
    for card in cards:
        card["meeting_interest"].update(reference=entry["reference"], attribution=entry["attribution"])
    run = generate(context, coverage, cards)
    pilot.validate_pilot(run, coverage=coverage, interest_context=context)
    cards[0]["meeting_interest"]["reference"] = context["interests"][1]["evidence"][0]["reference"]
    with pytest.raises(ValueError):
        generate(context, coverage, cards)


def test_export_still_rejects_failed_or_mismatched_provenance(inputs, tmp_path):
    context, report, coverage, cards, _, _ = inputs
    run = generate(context, coverage, cards)
    for mutation in ("failed", "hash", "attribution"):
        bad = copy.deepcopy(report)
        if mutation == "failed":
            bad["checks"][0]["verified"] = False
        elif mutation == "hash":
            bad["checks"][0]["sha256"] = "0" * 64
        else:
            bad["checks"][0]["attribution"] = "direct_request"
        with pytest.raises(ValueError):
            pilot.export_pilot(run, output_dir=tmp_path / mutation, source_verification=bad)
