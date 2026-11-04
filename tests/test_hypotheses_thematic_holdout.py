"""Synthetic contracts for a separate, source-bound thematic holdout score."""
from __future__ import annotations

import copy
import importlib

import pytest


@pytest.fixture
def evaluation():
    name = "tof_reliability.vitaly_bot.hypotheses_thematic_holdout"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        class MissingEvaluation:
            def __getattr__(self, attribute):
                pytest.fail("thematic holdout evaluation contract is not implemented")

        return MissingEvaluation()


@pytest.fixture
def sample():
    sources = {"s": "alpha beta gamma delta"}
    topics = [
        {"id": name, "description": f"analytical question {name}", "evidence": [
            {"source_id": "s", "start": start, "end": start + len(name), "fragment": name}
        ]}
        for name, start in [("alpha", 0), ("beta", 6), ("gamma", 11), ("delta", 17)]
    ]
    groups = [
        {"id": "g1", "description": "first theme", "child_ids": ["alpha", "beta"],
         "evidence": copy.deepcopy(topics[0]["evidence"])},
        {"id": "g2", "description": "second theme", "child_ids": ["gamma"],
         "evidence": copy.deepcopy(topics[2]["evidence"])},
        {"id": "g3", "description": "third theme", "child_ids": ["delta"],
         "evidence": copy.deepcopy(topics[3]["evidence"])},
    ]
    predictions = [f"idea {i} about alpha beta gamma delta" for i in range(8)]
    mappings = [
        {"group_id": "g1", "prediction_index": 0, "support": "strong",
         "source_quote": "alpha", "prediction_quote": "alpha", "rationale": "same question"},
        {"group_id": "g2", "prediction_index": 1, "support": "partial",
         "source_quote": "gamma", "prediction_quote": "gamma", "rationale": "related theme"},
    ]
    return dict(topics=topics, groups=groups, exclusions=[], source_texts=sources,
                predictions=predictions, mappings=mappings)


def test_partition_accepts_complete_evidence_bound_groups(evaluation, sample):
    evaluation.validate_partition(**{k: sample[k] for k in
                                    ("topics", "groups", "exclusions", "source_texts")})


@pytest.mark.parametrize("mutation", ["dropped", "duplicate", "unknown", "duplicate_group"])
def test_partition_rejects_lost_or_repeated_topics(evaluation, sample, mutation):
    groups = sample["groups"]
    if mutation == "dropped":
        groups[0]["child_ids"].remove("beta")
    elif mutation == "duplicate":
        groups[1]["child_ids"].append("beta")
    elif mutation == "unknown":
        groups[1]["child_ids"].append("unknown")
    else:
        groups[1]["id"] = groups[0]["id"]
    with pytest.raises(ValueError):
        evaluation.validate_partition(**{k: sample[k] for k in
                                        ("topics", "groups", "exclusions", "source_texts")})


@pytest.mark.parametrize("mutation", ["missing_topic_evidence", "invalid_topic_span",
                                      "missing_group_evidence", "invalid_group_quote"])
def test_every_topic_and_group_requires_verified_source_evidence(evaluation, sample, mutation):
    if mutation == "missing_topic_evidence":
        sample["topics"][1]["evidence"] = []
    elif mutation == "invalid_topic_span":
        sample["topics"][1]["evidence"][0]["end"] += 1
    elif mutation == "missing_group_evidence":
        sample["groups"][0]["evidence"] = []
    else:
        sample["groups"][0]["evidence"][0]["fragment"] = "invented"
    with pytest.raises(ValueError):
        evaluation.validate_partition(**{k: sample[k] for k in
                                        ("topics", "groups", "exclusions", "source_texts")})


def test_exclusion_must_account_for_topic_and_explain_non_analytical_reason(evaluation, sample):
    sample["groups"][2]["child_ids"].remove("delta")
    sample["groups"].pop()
    sample["exclusions"] = [{"topic_id": "delta", "rationale": "non-analytical greeting"}]
    args = {k: sample[k] for k in ("topics", "groups", "exclusions", "source_texts")}
    evaluation.validate_partition(**args)
    sample["exclusions"][0]["rationale"] = ""
    with pytest.raises(ValueError):
        evaluation.validate_partition(**args)


def test_strict_and_broad_scores_are_distinct_and_partial_does_not_pass(evaluation, sample):
    result = evaluation.score_groups(**sample)
    assert result["denominator"] == 3
    assert result["strict_matched_count"] == 1
    assert result["broad_matched_count"] == 2
    assert result["strict_coverage"] == pytest.approx(1 / 3)
    assert result["broad_coverage"] == pytest.approx(2 / 3)
    assert result["passing"] is False


def test_threshold_is_inclusive_and_never_rounded_up(evaluation):
    assert evaluation.passes_threshold(7, 10) is True
    assert evaluation.passes_threshold(6, 10) is False
    assert evaluation.passes_threshold(69, 100) is False
    with pytest.raises(ValueError):
        evaluation.passes_threshold(0, 0)


@pytest.mark.parametrize("mutation", ["late_prediction", "invented_source_quote",
                                      "invented_prediction_quote", "unknown_group", "weak_support"])
def test_mappings_require_first_seven_and_grounded_supported_quotes(evaluation, sample, mutation):
    mapping = sample["mappings"][0]
    if mutation == "late_prediction":
        mapping["prediction_index"] = 7
    elif mutation == "invented_source_quote":
        mapping["source_quote"] = "invented"
    elif mutation == "invented_prediction_quote":
        mapping["prediction_quote"] = "invented"
    elif mutation == "unknown_group":
        mapping["group_id"] = "unknown"
    else:
        mapping["support"] = "weak"
        result = evaluation.score_groups(**sample)
        assert result["strict_matched_count"] == 0
        assert result["broad_matched_count"] == 1
        assert result["passing"] is False
        return
    with pytest.raises(ValueError):
        evaluation.score_groups(**sample)


def test_grouping_prompt_uses_only_holdout_topics_and_evidence(evaluation, sample):
    original = copy.deepcopy(sample["topics"])
    prompt = evaluation.build_grouping_prompt(topics=sample["topics"])
    assert "analytical question alpha" in prompt
    assert "alpha" in prompt and "s" in prompt
    assert "idea 0 about" not in prompt
    assert "70%" not in prompt and "6/30" not in prompt
    assert sample["topics"] == original
