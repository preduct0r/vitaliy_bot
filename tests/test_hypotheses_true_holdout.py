"""Synthetic contracts for an independently assessed, source-bound holdout.

The proposed pure API lives in hypotheses_true_holdout. None of these fixtures
contain meeting data or call a model. A mapping is a comparator's assertion,
not itself proof: the scorer must validate references and quoted source spans.
"""
from __future__ import annotations

import copy
import importlib
import json

import pytest


@pytest.fixture
def evaluation():
    name = "tof_reliability.vitaly_bot.hypotheses_true_holdout"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        class MissingEvaluation:
            def __getattr__(self, attribute):
                pytest.fail("true holdout evaluation contract is not implemented")

        return MissingEvaluation()


@pytest.fixture
def sample():
    sources = {"s1": "🐟 alpha source evidence", "s2": "beta source evidence",
               "s3": "gamma source evidence"}
    topics = [
        {"id": "a", "description": "topic alpha", "evidence": [
            {"source_id": "s1", "start": 2, "end": 7, "fragment": "alpha"}]},
        {"id": "b", "description": "topic beta", "evidence": [
            {"source_id": "s2", "start": 0, "end": 4, "fragment": "beta"}]},
        {"id": "c", "description": "topic gamma", "evidence": [
            {"source_id": "s3", "start": 0, "end": 5, "fragment": "gamma"}]},
    ]
    predictions = ["hypothesis one", "hypothesis two", "three", "four", "five",
                   "six", "seven", "late hypothesis"]
    mappings = [
        {"topic_id": "a", "prediction_index": 0, "rationale": "same analytical question"},
        {"topic_id": "b", "prediction_index": 1, "rationale": "same outcome and population"},
    ]
    return sources, topics, predictions, mappings


def score(evaluation, sample):
    sources, topics, predictions, mappings = sample
    return evaluation.score_holdout(topics=topics, predictions=predictions,
                                    mappings=mappings, source_texts=sources)


def test_semantic_mapping_counts_all_topics_and_reports_misses(evaluation, sample):
    result = score(evaluation, sample)
    assert result["denominator"] == 3
    assert result["matched_count"] == 2
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["passing"] is False
    assert set(result["matched_topic_ids"]) == {"a", "b"}
    assert result["missed_topic_ids"] == ["c"]


def test_threshold_is_inclusive_and_not_rounded_up(evaluation):
    sources = {"s": "abcdefghij"}
    topics = [{"id": str(i), "description": f"topic {i}", "evidence": [
        {"source_id": "s", "start": i, "end": i + 1, "fragment": sources["s"][i]}]}
        for i in range(10)]
    predictions = [f"hypothesis {i}" for i in range(7)]
    mappings = [{"topic_id": str(i), "prediction_index": i, "rationale": "same scope"}
                for i in range(7)]
    args = dict(topics=topics, predictions=predictions, source_texts=sources)
    passed = evaluation.score_holdout(**args, mappings=mappings)
    assert passed["coverage"] == pytest.approx(0.7)
    assert passed["passing"] is True
    failed = evaluation.score_holdout(**args, mappings=mappings[:-1])
    assert failed["denominator"] == 10
    assert failed["passing"] is False


def test_eighth_prediction_cannot_rescue_score(evaluation, sample):
    sample[3].append({"topic_id": "c", "prediction_index": 7, "rationale": "late match"})
    with pytest.raises(ValueError, match="(?i)(seven|first|index|prediction)"):
        score(evaluation, sample)


@pytest.mark.parametrize("mutation", ["invented_quote", "wrong_offset", "unknown_source", "no_evidence"])
def test_source_evidence_must_match_exact_text_and_position(evaluation, sample, mutation):
    topics = sample[1]
    evidence = topics[0]["evidence"][0]
    if mutation == "invented_quote":
        evidence["fragment"] = "other"
    elif mutation == "wrong_offset":
        evidence["start"] = 1
    elif mutation == "unknown_source":
        evidence["source_id"] = "missing"
    else:
        topics[0]["evidence"] = []
    with pytest.raises(ValueError, match="(?i)(evidence|source|fragment|position|span)"):
        score(evaluation, sample)


@pytest.mark.parametrize("mapping", [
    {"topic_id": "missing", "prediction_index": 0, "rationale": "yes"},
    {"topic_id": "c", "prediction_index": -1, "rationale": "yes"},
    {"topic_id": "c", "prediction_index": True, "rationale": "yes"},
    {"topic_id": "c", "prediction_index": 0, "rationale": ""},
])
def test_unknown_or_unexplained_mapping_is_rejected(evaluation, sample, mapping):
    sample[3].append(mapping)
    with pytest.raises(ValueError):
        score(evaluation, sample)


def test_duplicate_topic_ids_cannot_inflate_coverage(evaluation, sample):
    sample[1].append(copy.deepcopy(sample[1][0]))
    with pytest.raises(ValueError, match="(?i)(duplicate|unique)"):
        score(evaluation, sample)


def test_generator_prompt_receives_only_validated_training_content(evaluation):
    training = {"interests": [{"requirement": "training analytical interest",
                               "evidence": [{"source_id": "training-source", "fragment": "train"}]}]}
    original = copy.deepcopy(training)
    prompt = evaluation.build_generator_prompt(training_map=training, provenance_verified=True)
    assert "training analytical interest" in prompt
    assert training == original
    assert "training-source" not in prompt
    assert "2026-10-26" not in prompt
    assert "HOLDOUT_SECRET" not in prompt
    with pytest.raises(ValueError, match="(?i)(provenance|valid|verif)"):
        evaluation.build_generator_prompt(training_map=training, provenance_verified=False)


@pytest.mark.parametrize("item_type", ["command_execution", "mcp_tool_call", "web_search"])
def test_generator_trace_rejects_any_tool_event(evaluation, item_type):
    events = [{"type": "thread.started"}, {"type": "turn.started"},
              {"type": "item.completed", "item": {"type": item_type}},
              {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
              {"type": "turn.completed"}]
    with pytest.raises(ValueError, match="(?i)(tool|trace|isolation|event)"):
        evaluation.validate_generator_trace("\n".join(json.dumps(e) for e in events))
