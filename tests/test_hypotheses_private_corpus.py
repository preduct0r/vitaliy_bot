"""Operator workflow contract. All meeting content and model outputs are synthetic.

The operator API accepts source records and independently supplied evidence; it must
persist private audit artifacts and must never infer approval from a model response.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile

import pytest

from tof_reliability.vitaly_bot.hypotheses_interests import assess_provenance


@pytest.fixture
def operator():
    try:
        return importlib.import_module("tof_reliability.vitaly_bot.hypotheses_private_corpus")
    except ModuleNotFoundError as exc:
        if exc.name != "tof_reliability.vitaly_bot.hypotheses_private_corpus":
            raise
        pytest.fail("private corpus operator workflow is not implemented")


@pytest.fixture
def case():
    root = Path(os.environ["WORKFLOW_OUTPUT_DIR"]) if os.environ.get("WORKFLOW_OUTPUT_DIR") else None
    if root:
        root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="private-corpus-test-", dir=root) as directory:
        root = Path(directory)
        sources = []
        for ident, day, content in [
            ("train", "2026-10-01", "Привет 🐟 повторные ремонты — synthetic training"),
            ("h26", "2026-10-26", "HOLDOUT_26_SYNTHETIC качество"),
            ("h28", "2026-10-28", "HOLDOUT_28_SYNTHETIC сроки"),
            ("h30", "2026-10-30", "HOLDOUT_30_SYNTHETIC риски"),
            ("rejected", "2026-10-20", "REJECTED_SECRET"),
        ]:
            path = root / (ident + ".txt")
            path.write_text(content, encoding="utf-8")
            sources.append({"id": ident, "date": day, "status": "rejected" if ident == "rejected" else "accepted",
                            "kind": "transcript", "path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        train_text = Path(sources[0]["path"]).read_text(encoding="utf-8")
        fragment = "повторные ремонты"
        start = train_text.index(fragment)
        profile = {"version": "train-v1", "corpus_version": "corpus-v1", "interests": [
            {"id": "repeat", "requirement": "Исследовать повторы", "evidence": [
                {"source_id": "train", "locator": f"chars:{start}-{start + len(fragment)}",
                 "fragment": fragment, "attribution": "third_party_paraphrase"}]}]}
        topics = []
        for source, label in zip(sources[1:4], ["quality", "duration", "risk"]):
            content = Path(source["path"]).read_text(encoding="utf-8")
            topics.append({"id": label, "evidence": [{"source_id": source["id"],
                           "locator": f"chars:0-{len(content)}", "fragment": content}]})
        yield {"root": root, "sources": sources, "profile": profile, "topics": topics}


class FakeModel:
    def __init__(self, predictions=None, error=None):
        self.calls = []
        self.predictions = predictions or ["quality", "duration", "risk", "x", "y", "z", "q"]
        self.error = error

    def __call__(self, request):
        self.calls.append(copy.deepcopy(request))
        if self.error:
            raise RuntimeError(self.error)
        return {"predictions": self.predictions}


def run(operator, case, model, **overrides):
    arguments = {"sources": case["sources"], "training_map": case["profile"],
                 "holdout_topics": case["topics"], "model": model,
                 "output_dir": case["root"] / "artifacts", "corpus_version": "corpus-v1"}
    arguments.update(overrides)
    return operator.build_and_assess_private_corpus(**arguments)


def load_artifacts(result):
    return {name: json.loads(Path(path).read_text(encoding="utf-8"))
            for name, path in result["artifacts"].items()}


def test_roundtrip_provenance_chronology_and_no_generator_leak(operator, case):
    model = FakeModel()
    result = run(operator, case, model)
    assert result["status"] == "assessed"
    artifacts = load_artifacts(result)
    index = artifacts["corpus"]
    training = artifacts["training_map"]
    assert assess_provenance(training, index)["verified"] is True
    assert training["interests"][0]["evidence"][0]["attribution"] == "third_party_paraphrase"
    assert len(model.calls) == 1
    serialized = json.dumps(model.calls[0], ensure_ascii=False)
    for secret in ("HOLDOUT_26_SYNTHETIC", "HOLDOUT_28_SYNTHETIC", "HOLDOUT_30_SYNTHETIC",
                   "REJECTED_SECRET", "quality", "duration", "risk", "h26", "h28", "h30"):
        assert secret not in serialized
    assert artifacts["score"]["coverage"] == 1.0
    assert artifacts["score"]["passing"] is True
    assert artifacts["model_input"] == model.calls[0]
    assert artifacts["model_output"]["predictions"] == model.predictions
    assert artifacts["holdout_topics"] == case["topics"]
    assert artifacts["report"]["limitations"]
    assert artifacts["report"]["independent_verifier_required"] is True
    assert all(case["root"] in Path(path).parents for path in result["artifacts"].values())


def test_inventory_deduplicates_and_reports_rejected_and_missing(operator, case):
    duplicate = {**case["sources"][0], "id": "train-copy"}
    missing = {**case["sources"][0], "id": "missing", "path": str(case["root"] / "absent.txt")}
    model = FakeModel()
    result = run(operator, case, model, sources=case["sources"] + [duplicate, missing])
    assert result["status"] == "blocked"
    assert model.calls == []
    artifacts = load_artifacts(result)
    inventory = artifacts["inventory"]
    rows = {row["id"]: row for row in inventory["sources"]}
    assert rows["rejected"]["status"] == "rejected"
    assert rows["missing"]["status"] == "missing"
    assert rows["train-copy"]["duplicate_of"] == "train"
    assert "missing" in json.dumps(artifacts["report"])


def test_bad_evidence_or_holdout_as_training_blocks_before_model(operator, case):
    model = FakeModel()
    profile = copy.deepcopy(case["profile"])
    profile["interests"][0]["evidence"][0]["locator"] = "chars:0-2"
    result = run(operator, case, model, training_map=profile)
    assert result["status"] == "blocked"
    assert model.calls == []
    assert "provenance" in json.dumps(load_artifacts(result)["report"]).lower()

    profile["interests"][0]["evidence"][0].update(source_id="h26", locator="chars:0-10", fragment="HOLDOUT_26")
    result = run(operator, case, model, training_map=profile, output_dir=case["root"] / "leak")
    assert result["status"] == "blocked"
    assert model.calls == []


def test_invalid_holdout_evidence_and_missing_date_block(operator, case):
    model = FakeModel()
    topics = copy.deepcopy(case["topics"])
    topics[0]["evidence"][0]["fragment"] = "fabricated"
    result = run(operator, case, model, holdout_topics=topics)
    assert result["status"] == "blocked"
    assert model.calls == []
    result = run(operator, case, model, sources=[s for s in case["sources"] if s["id"] != "h30"],
                 output_dir=case["root"] / "missing-date")
    assert result["status"] == "blocked"
    assert model.calls == []
    assert "2026-10-30" in json.dumps(load_artifacts(result)["report"])


def test_failed_score_and_model_unavailable_are_honestly_blocked(operator, case):
    model = FakeModel(["quality", "unmatched", "x", "y", "z", "a", "b", "duration", "risk"])
    result = run(operator, case, model)
    artifacts = load_artifacts(result)
    assert result["status"] == "blocked"
    assert artifacts["score"]["coverage"] == pytest.approx(1 / 3)
    assert artifacts["score"]["passing"] is False
    assert artifacts["model_output"]["predictions"] == model.predictions
    unavailable = FakeModel(error="synthetic model unavailable")
    result = run(operator, case, unavailable, output_dir=case["root"] / "unavailable")
    assert result["status"] == "blocked"
    assert "synthetic model unavailable" in json.dumps(load_artifacts(result)["report"])
    assert "score" not in result["artifacts"]


def test_repeated_runs_produce_deterministic_audit_content(operator, case):
    first = load_artifacts(run(operator, case, FakeModel(), output_dir=case["root"] / "one"))
    second = load_artifacts(run(operator, case, FakeModel(), output_dir=case["root"] / "two"))
    for key in ("inventory", "corpus", "training_map", "holdout_topics", "model_input", "model_output", "score", "report"):
        assert first[key] == second[key]
