"""Red-first synthetic contracts for the additional September holdout.

These fixtures contain no private meeting data and perform no model calls.
The new trial is separate from, and cannot change, the original release gate.
"""
from __future__ import annotations

import copy
import hashlib
import importlib

import pytest

from tof_reliability.vitaly_bot.hypotheses_interests import chronological_holdout


DATES = ("2026-09-17", "2026-09-24", "2026-09-25")


@pytest.fixture
def index():
    return {"version": "synthetic", "sources": [
        {"id": sid, "date": day} for sid, day in [
            ("early", "2026-09-16"), ("early2", "2026-09-01"),
            ("h1", DATES[0]), ("h1b", DATES[0]),
            ("between", "2026-09-20"), ("h2", DATES[1]),
            ("h3", DATES[2]), ("late", "2026-10-26"),
        ]
    ]}


@pytest.fixture
def third():
    name = "tof_reliability.vitaly_bot.hypotheses_third_holdout"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        class MissingThirdHoldout:
            def __getattr__(self, attribute):
                pytest.fail("third holdout provenance and scoring contract is not implemented")

        return MissingThirdHoldout()


def test_preselected_dates_include_every_meeting_and_cut_training_strictly(index):
    split = chronological_holdout(index, dates=DATES)
    assert {s["id"] for s in split["training"]} == {"early", "early2"}
    assert {s["id"] for s in split["holdout"]} == {"h1", "h1b", "h2", "h3"}
    assert {s["id"] for s in split["excluded"]} == {"between", "late"}


@pytest.mark.parametrize("missing", DATES)
def test_missing_preselected_date_fails_closed(index, missing):
    index["sources"] = [s for s in index["sources"] if s["date"] != missing]
    with pytest.raises(ValueError, match="(?i)missing"):
        chronological_holdout(index, dates=DATES)


def test_context_contamination_excludes_early_chunks_and_checkpoints_reasons(third, index):
    chunks = [
        {"id": "clean", "source_id": "early", "context_id": "c1"},
        {"id": "shared_holdout", "source_id": "early", "context_id": "c2"},
        {"id": "heldout", "source_id": "h1", "context_id": "c2"},
        {"id": "shared_later", "source_id": "early2", "context_id": "c3"},
        {"id": "later", "source_id": "late", "context_id": "c3"},
        {"id": "intervening", "source_id": "between", "context_id": "c4"},
    ]
    original = copy.deepcopy(chunks)
    result = third.filter_training_chunks(index=index, chunks=chunks, holdout_dates=DATES)
    assert [c["id"] for c in result["training"]] == ["clean"]
    excluded = {e["chunk_id"]: e for e in result["exclusions"]}
    assert set(excluded) == {"shared_holdout", "heldout", "shared_later", "later", "intervening"}
    assert excluded["shared_holdout"]["context_id"] == "c2"
    assert excluded["shared_later"]["context_id"] == "c3"
    assert all(e.get("reason") for e in excluded.values())
    assert chunks == original


@pytest.mark.parametrize("bad_chunk", [
    {"id": "unknown", "source_id": "missing", "context_id": "c"},
    {"id": "no-context", "source_id": "early"},
])
def test_unverifiable_context_provenance_fails_closed(third, index, bad_chunk):
    with pytest.raises(ValueError, match="(?i)(source|context|provenance)"):
        third.filter_training_chunks(index=index, chunks=[bad_chunk], holdout_dates=DATES)


@pytest.fixture
def trial():
    text = "alpha beta gamma calendar"
    digest = hashlib.sha256(text.encode()).hexdigest()
    spans = [("a", 0, 5, "analytical"), ("b", 6, 10, "analytical"),
             ("c", 11, 16, "analytical"), ("d", 17, 25, "organizational")]
    groups = [{"id": ident, "description": ident, "disposition": disposition,
               "reason": "fixed before matching", "evidence": [{
                   "source_id": "h", "start": start, "end": end,
                   "fragment": text[start:end], "sha256": digest}]}
              for ident, start, end, disposition in spans]
    predictions = ["alpha question", "beta question", "third", "fourth", "fifth"]
    mappings = [
        {"group_id": "a", "prediction_index": 0, "support": "strong",
         "rationale": "same question", "source_quote": "alpha", "prediction_quote": "alpha"},
        {"group_id": "b", "prediction_index": 1, "support": "partial",
         "rationale": "overlapping question", "source_quote": "beta", "prediction_quote": "beta"},
    ]
    return {"observed_group_ids": ["a", "b", "c", "d"], "groups": groups,
            "source_texts": {"h": text}, "predictions": predictions, "mappings": mappings}


def test_five_or_six_predictions_score_all_fixed_analytical_groups(third, trial):
    for count in (5, 6):
        args = copy.deepcopy(trial)
        args["predictions"] = (args["predictions"] + ["sixth"])[:count]
        result = third.score_third_holdout(**args)
        assert result["denominator"] == 3
        assert result["strict_matched_count"] == 1
        assert result["broad_matched_count"] == 2
        assert result["passing"] is False
        assert result["original_gate_met"] is False


@pytest.mark.parametrize("mutation", ["missing_group", "missing_disposition", "missing_reason",
                                     "bad_hash", "bad_offset", "duplicate_group"])
def test_every_observed_group_needs_a_grounded_disposition(third, trial, mutation):
    groups = trial["groups"]
    if mutation == "missing_group":
        groups.pop()
    elif mutation == "missing_disposition":
        del groups[-1]["disposition"]
    elif mutation == "missing_reason":
        groups[-1]["reason"] = ""
    elif mutation == "bad_hash":
        groups[-1]["evidence"][0]["sha256"] = "0" * 64
    elif mutation == "bad_offset":
        groups[-1]["evidence"][0]["start"] += 1
    else:
        groups.append(copy.deepcopy(groups[-1]))
    with pytest.raises(ValueError):
        third.score_third_holdout(**trial)


def test_insufficient_predictions_cannot_be_padded_or_scored(third, trial):
    trial["predictions"] = trial["predictions"][:4]
    with pytest.raises(ValueError, match="(?i)(prediction|card|insufficient)"):
        third.score_third_holdout(**trial)


def test_grouping_prompt_never_receives_prediction_fields(third, trial):
    group = copy.deepcopy(trial["groups"][0])
    group["prediction"] = "PREDICTION_SECRET"
    prompt = third.build_grouping_prompt(observations=[group])
    assert "alpha" in prompt
    assert "PREDICTION_SECRET" not in prompt
