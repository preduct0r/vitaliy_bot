"""Synthetic contracts for the additional chronological evaluation.

No private corpus, model, production service, or original evaluation is used.
"""
from __future__ import annotations

import hashlib
import importlib

import pytest

from tof_reliability.vitaly_bot.hypotheses_interest_extraction import select_training_chunks
from tof_reliability.vitaly_bot.hypotheses_interests import chronological_holdout
from tof_reliability.vitaly_bot.hypotheses_thematic_holdout import passes_threshold


DATES = ("2026-10-02", "2026-10-05", "2026-10-23")


@pytest.fixture
def index():
    return {"version": "synthetic", "sources": [
        {"id": ident, "date": day} for ident, day in (
            ("early", "2026-10-01"), ("h1", "2026-10-02"),
            ("h1b", "2026-10-02"), ("between", "2026-10-03"),
            ("h2", "2026-10-05"), ("h3", "2026-10-23"),
            ("original", "2026-10-26"), ("late", "2026-10-31"),
        )]}


def test_split_keeps_all_selected_meetings_and_excludes_every_later_meeting(index):
    split = chronological_holdout(index, dates=DATES)
    assert {s["id"] for s in split["training"]} == {"early"}
    assert {s["id"] for s in split["holdout"]} == {"h1", "h1b", "h2", "h3"}
    assert {s["id"] for s in split["excluded"]} == {"between", "original", "late"}


def test_missing_selected_day_fails_closed(index):
    index["sources"] = [s for s in index["sources"] if s["date"] != DATES[1]]
    with pytest.raises(ValueError, match="(?i)missing"):
        chronological_holdout(index, dates=DATES)


def test_chunk_selection_cannot_admit_intervening_or_original_holdout(index):
    chunks = [{"source_id": s["id"], "marker": "chunk-" + s["id"]}
              for s in index["sources"]]
    selected = select_training_chunks(index, chunks, holdout_dates=DATES)
    assert [c["source_id"] for c in selected] == ["early"]


@pytest.fixture
def second_holdout():
    name = "tof_reliability.vitaly_bot.hypotheses_second_holdout"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        pytest.fail("second holdout reconstruction contract is not implemented")


def test_reconstruction_revalidates_source_and_quote_before_grouping(second_holdout):
    text = "synthetic training evidence"
    source = {"id": "early", "date": "2026-10-01", "sha256": hashlib.sha256(text.encode()).hexdigest()}
    observation = {"source_id": "early", "requirement": "synthetic question",
                   "start": 0, "end": 9, "fragment": "synthetic"}
    result = second_holdout.reconstruct_observations(
        sources=[source], source_texts={"early": text}, observations=[observation])
    assert len(result) == 1
    for broken_source, broken_observation in (
        (dict(source, sha256="0" * 64), observation),
        (source, dict(observation, fragment="invented")),
        (source, dict(observation, source_id="h1")),
        (dict(source, date="2026-10-03"), observation),
    ):
        with pytest.raises(ValueError):
            second_holdout.reconstruct_observations(
                sources=[broken_source], source_texts={"early": text},
                observations=[broken_observation])


def test_training_grouping_receives_only_revalidated_observations(second_holdout):
    text = "training excerpt"
    source = {"id": "early", "date": "2026-10-01", "sha256": hashlib.sha256(text.encode()).hexdigest()}
    obs = {"source_id": "early", "requirement": "training question", "start": 0,
           "end": len(text), "fragment": text}
    seen = []
    result = second_holdout.build_training_map(
        sources=[source], source_texts={"early": text}, observations=[obs],
        grouper=lambda value: seen.append(value) or {"interests": [{"requirement": "training question"}]})
    assert result["interests"]
    assert len(seen) == 1
    assert "HOLDOUT_SECRET" not in str(seen)
    assert "2026-10-02" not in str(seen)


def test_strict_threshold_uses_exact_integer_counts():
    assert passes_threshold(7, 10) is True
    assert passes_threshold(69, 100) is False
    assert passes_threshold(699, 1000) is False
    with pytest.raises(ValueError):
        passes_threshold(0, 0)
