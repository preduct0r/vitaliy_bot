"""S1 contract: synthetic evidence only; no production archive is accessed."""

import copy
import importlib
import json

import pytest


@pytest.fixture
def interests():
    try:
        return importlib.import_module("tof_reliability.vitaly_bot.hypotheses_interests")
    except ModuleNotFoundError as exc:
        if exc.name != "tof_reliability.vitaly_bot.hypotheses_interests":
            raise
        class MissingInterests:
            def __getattr__(self, _name):
                pytest.fail("S1 interest corpus and holdout module is not implemented")
        return MissingInterests()


@pytest.fixture
def sources():
    # The sentinel is deliberately synthetic and must never reach model context.
    return [
        {"id": "base", "date": "2026-10-01", "status": "accepted", "kind": "baseline",
         "sha256": "a" * 64, "text": "PRIVATE_SENTINEL_BASE"},
        {"id": "early", "date": "2026-10-25", "status": "accepted", "kind": "transcript",
         "sha256": "b" * 64, "text": "PRIVATE_SENTINEL_EARLY"},
        *[{"id": day, "date": day, "status": "accepted", "kind": "transcript",
           "sha256": char * 64, "text": "PRIVATE_SENTINEL_" + day}
          for day, char in [("2026-10-26", "c"), ("2026-10-28", "d"), ("2026-10-30", "e")]],
        {"id": "rejected", "date": "2026-10-20", "status": "rejected", "kind": "transcript",
         "sha256": "f" * 64, "text": "PRIVATE_SENTINEL_REJECTED"},
    ]


def test_index_is_versioned_deduplicated_and_tracks_accepted_sources(interests, sources):
    duplicate = {**sources[1], "id": "early-copy"}
    index = interests.index_corpus(sources + [duplicate], version="corpus-test-1")
    assert index["version"] == "corpus-test-1"
    assert {item["id"] for item in index["sources"]} == {
        "base", "early", "2026-10-26", "2026-10-28", "2026-10-30"
    }
    assert all(len(item["sha256"]) == 64 for item in index["sources"])
    assert index == interests.index_corpus(sources + [duplicate], version="corpus-test-1")


def test_holdout_is_chronological_and_excluded_from_generator_context(interests, sources):
    index = interests.index_corpus(sources, version="corpus-test-1")
    split = interests.chronological_holdout(index, dates=["2026-10-26", "2026-10-28", "2026-10-30"])
    assert {item["id"] for item in split["training"]} == {"base", "early"}
    assert {item["id"] for item in split["holdout"]} == {"2026-10-26", "2026-10-28", "2026-10-30"}
    # Expectations are verifier-only even if supplied with the split.
    split["expected_topic_ids"] = ["SECRET_EXPECTED_TOPIC"]
    context = json.dumps(interests.generator_context(split), ensure_ascii=False)
    assert "SECRET_EXPECTED_TOPIC" not in context
    assert "PRIVATE_SENTINEL" not in context
    assert "2026-10-26" not in context and "2026-10-28" not in context and "2026-10-30" not in context


def test_split_rejects_later_training_source_instead_of_leaking_future(interests, sources):
    sources.append({"id": "future", "date": "2026-10-29", "status": "accepted", "kind": "transcript",
                    "sha256": "0" * 64, "text": "synthetic"})
    index = interests.index_corpus(sources, version="corpus-test-1")
    with pytest.raises(ValueError, match="(?i)chronolog|holdout|date|дат"):
        interests.chronological_holdout(index, dates=["2026-10-26", "2026-10-28", "2026-10-30"])


def test_holdout_uses_first_seven_and_requires_seventy_percent(interests):
    expected = [f"group-{i}" for i in range(10)]
    passing = interests.score_interest_holdout(expected, expected[:7] + ["other"])
    assert passing["coverage"] == pytest.approx(0.7)
    assert passing["passing"] is True
    failing = interests.score_interest_holdout(expected, expected[:6] + ["other", expected[6]])
    assert failing["coverage"] == pytest.approx(0.6)
    assert failing["passing"] is False
    assert interests.score_interest_holdout([], ["other"])["passing"] is False


def test_interest_validator_enforces_traceability_and_attribution(interests, sources):
    index = interests.index_corpus(sources, version="corpus-test-1")
    profile = {"version": "interests-test-1", "corpus_version": "corpus-test-1", "interests": [
        {"id": "repair-repeat", "requirement": "Проверять повторные ремонты",
         "evidence": [{"source_id": "early", "locator": "00:01-00:05", "attribution": "interpretation"}]}
    ]}
    interests.validate_interest_map(profile, index)
    for edit in (lambda p: p["interests"][0]["evidence"][0].update(source_id="missing"),
                 lambda p: p["interests"][0]["evidence"][0].update(locator=""),
                 lambda p: p["interests"][0]["evidence"][0].update(attribution="certain"),
                 lambda p: p.update(corpus_version="wrong")):
        invalid = copy.deepcopy(profile)
        edit(invalid)
        with pytest.raises(ValueError):
            interests.validate_interest_map(invalid, index)
    for attribution in ("direct_request", "interpretation", "third_party_paraphrase"):
        variant = copy.deepcopy(profile)
        variant["interests"][0]["evidence"][0]["attribution"] = attribution
        interests.validate_interest_map(variant, index)


def test_fixed_scenarios_cover_required_independent_cases(interests):
    scenarios = interests.fixed_evaluation_scenarios()
    assert len(scenarios) >= 12
    assert len({scenario["id"] for scenario in scenarios}) == len(scenarios)
    tags = {tag for scenario in scenarios for tag in scenario["tags"]}
    assert {"no_files", "scope", "unknown_prior_analyses", "partial_periods", "missing_joins",
            "units", "unconfirmed_numbers", "repeated_repairs", "labor", "criticality",
            "server_outage", "planned_effect"} <= tags
    assert all(scenario.get("input") and scenario.get("expected_checks") for scenario in scenarios)


def test_index_rejects_conflicting_identity_and_invalid_hash(interests, sources):
    with pytest.raises(ValueError):
        interests.index_corpus(sources + [{**sources[1], "id": "base", "sha256": "1" * 64}], version="v")
    with pytest.raises(ValueError):
        interests.index_corpus([{**sources[0], "sha256": "not-a-hash"}], version="v")


def test_holdout_requires_every_date_and_validates_training_map(interests, sources):
    index = interests.index_corpus(sources[:-2], version="v")
    with pytest.raises(ValueError, match="(?i)holdout|date"):
        interests.chronological_holdout(index, dates=["2026-10-26", "2026-10-28", "2026-10-30"])
    full = interests.index_corpus(sources, version="v")
    split = interests.chronological_holdout(full, dates=["2026-10-26", "2026-10-28", "2026-10-30"])
    profile = {"version": "p", "corpus_version": "v", "interests": [{"id": "x", "requirement": "r", "evidence": [{"source_id": "2026-10-26", "locator": "1", "attribution": "direct_request"}]}]}
    with pytest.raises(ValueError):
        interests.generator_context(split, profile)


def test_preparation_verifies_bytes_and_evidence_fragments(interests, tmp_path):
    import hashlib
    source = tmp_path / "source.txt"
    source.write_text("synthetic private fragment", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    records = [{"id": "s", "date": "2026-10-01", "status": "accepted", "kind": "baseline", "sha256": digest, "path": str(source)}]
    index = interests.prepare_private_corpus(records, version="v")
    profile = {"version": "p", "corpus_version": "v", "interests": [{"id": "x", "requirement": "обобщение", "evidence": [{"source_id": "s", "locator": "chars:0-9", "attribution": "interpretation", "fragment": "synthetic"}]}]}
    assert interests.assess_provenance(profile, index)["verified"] is True
    profile["interests"][0]["evidence"][0]["fragment"] = "invented"
    assert interests.assess_provenance(profile, index)["verified"] is False
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)hash|sha"):
        interests.prepare_private_corpus(records, version="v")
