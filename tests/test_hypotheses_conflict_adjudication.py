"""Synthetic contracts for source-bound, independent conflict adjudication.

No private corpus, model, production service, or holdout is used.  The new
entry point consumes the original projection and verdicts without rewriting
those records. Hashes use canonical JSON for structured artifacts and raw
bytes for source text, fragments, prompts, and model traces.
"""
import copy
import hashlib
import json

import pytest

from tof_reliability.vitaly_bot import hypotheses_topic_projection as projection


VERSION = "conflict-adjudication.v1"
DIMENSION = "repair_techcard_equipment_operations_coverage"
TOPIC = "techcard_coverage_verification"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def byte_hash(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def case(tmp_path):
    texts = ["Сопоставить ремонты и технологические карты.",
             "Участник предположил проверить ремонты и карты.",
             "Данные о фактических затратах\nотсутствуют.",
             "Другая тема: хранение документов.",
             "Не сравнивать ремонты и технологические карты."]
    sources, evidence = [], []
    for i, fragment in enumerate(texts):
        text = "Ввод. " + fragment + " Конец."
        path = tmp_path / f"source-{i}.txt"
        path.write_text(text, encoding="utf-8")
        start = text.index(fragment)
        sources.append({"id": f"source-{i}", "path": str(path), "kind": "transcript",
                        "date": f"2026-09-0{i + 1}", "meeting_id": f"meeting-{i}",
                        "sha256": byte_hash(path.read_bytes())})
        evidence.append({"source_id": f"source-{i}", "locator": f"chars:{start}-{start + len(fragment)}",
                         "fragment": fragment,
                         "attribution": "interpretation" if i == 1 else "third_party"})
    index = {"version": "synthetic-generation-1", "sources": sources}
    profile = {"version": "synthetic-profile-1", "corpus_version": index["version"],
               "interests": [{"id": "interest-a", "requirement": "Проверить сопоставление",
                              "evidence": evidence}]}
    candidates = [{"interest_id": "interest-a", "dimension_id": DIMENSION, "topic_id": TOPIC,
                   "source_id": e["source_id"], "locator": e["locator"], "confidence": "high"}
                  for e in evidence]
    statuses = [
        {"supported": True, "confidence": "high", "conflicting": False},
        {"supported": True, "confidence": "high", "conflicting": False},
        {"supported": True, "confidence": "low", "conflicting": True},
        {"supported": False, "confidence": "low", "conflicting": True},
        {"supported": False, "confidence": "low", "conflicting": False},
    ]
    verdicts = [{**{k: c[k] for k in ("interest_id", "dimension_id", "source_id", "locator")},
                 "verdict": copy.deepcopy(v)} for c, v in zip(candidates, statuses)]
    by_source = {v["source_id"]: v["verdict"] for v in verdicts}
    original = projection.project_semantic_dimensions(profile, index, classifier=lambda _: candidates,
                 verifier=lambda request: copy.deepcopy(by_source[request["evidence"]["source_id"]]))
    reviews = []
    for n in (2, 3):
        entry = verdicts[n]
        review_pair = []
        for run in (1, 2):
            prompt = json.dumps({"run": run, "source_id": entry["source_id"],
                                 "locator": entry["locator"], "fragment": evidence[n]["fragment"]},
                                ensure_ascii=False).encode()
            trace = (json.dumps({"type": "thread.started", "thread_id": f"model-run-{run}"}) + "\n" +
                     json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text":
                         json.dumps({"verdicts": [{"source_id": entry["source_id"], "locator": entry["locator"],
                                                   "conflict_status": "no"}]})}})).encode()
            review_pair.append({"run_id": f"model-run-{run}", "prompt_id": f"prompt-{run}",
                                "prompt": prompt.decode(), "prompt_sha256": byte_hash(prompt),
                                "trace": trace.decode(), "trace_sha256": byte_hash(trace),
                                "decision": "no"})
        reviews.append({"interest_id": "interest-a", "dimension_id": DIMENSION,
                        "source_id": entry["source_id"], "locator": entry["locator"],
                        "original_verdict_sha256": digest(entry),
                        "fragment_sha256": byte_hash(evidence[n]["fragment"].encode()),
                        "source_sha256": sources[n]["sha256"], "reviews": review_pair})
    adjudications = {"version": VERSION, "profile_sha256": digest(profile),
                     "index_sha256": digest(index), "projection_sha256": digest(original),
                     "verdicts_sha256": digest(verdicts), "candidates_sha256": digest(candidates), "decisions": reviews}
    return {"profile": profile, "source_index": index, "original_projection": original,
            "original_verdicts": verdicts, "original_candidates": candidates, "adjudications": adjudications}


def adjudicate(data):
    assert hasattr(projection, "adjudicate_conflicts"), "versioned conflict adjudication is not implemented"
    return projection.adjudicate_conflicts(**data)


def refs(rows):
    return {(e.get("interest_id", row.get("interest_id")), row["dimension_id"],
             e["source_id"], e["locator"]) for row in rows for e in row["evidence"]}


def test_both_independent_reviews_recover_only_original_high_confidence_links(case):
    before = copy.deepcopy(case)
    assert not case["original_projection"]["dimensions"]
    result = adjudicate(case)
    assert case == before, "original projection, verdict history, and inputs must not be mutated"
    assert result["version"] != case["original_projection"]["version"]
    assert refs(result["dimensions"]) == {
        (v["interest_id"], v["dimension_id"], v["source_id"], v["locator"])
        for v in case["original_verdicts"][:2]}
    assert {e["attribution"] for row in result["dimensions"] for e in row["evidence"]} == {"third_party", "interpretation"}
    assert result["dimensions"][0]["independent_meeting_count"] == 2
    assert {v["source_id"] for v in case["original_verdicts"][2:]} <= {r[2] for r in refs(result["unconfirmed"])}
    assert result["dimensions"][0]["description"] == projection._CATALOGS[projection.DIMENSION_VERSION][DIMENSION][1]


@pytest.mark.parametrize("decision", ["yes", "unknown", "uncertain"])
def test_real_or_uncertain_conflict_never_releases_group(case, decision):
    case["adjudications"]["decisions"][0]["reviews"][1]["decision"] = decision
    try:
        result = adjudicate(case)
    except ValueError:
        return
    assert not result["dimensions"]
    assert len(refs(result["unconfirmed"])) == len(case["original_verdicts"])


def test_unreviewed_original_conflict_blocks_the_whole_group(case):
    case["adjudications"]["decisions"].pop()
    try:
        result = adjudicate(case)
    except ValueError:
        return
    assert not result["dimensions"]


@pytest.mark.parametrize("damage", [
    "missing_review", "duplicate_review", "same_run", "same_trace", "same_prompt_id",
    "missing_run", "missing_prompt", "missing_trace", "forged_trace", "forged_prompt",
    "duplicate_decision", "wrong_interest", "wrong_dimension", "wrong_source", "wrong_locator",
    "wrong_original_verdict", "wrong_fragment", "wrong_source_hash", "wrong_profile_generation",
    "wrong_index_generation", "wrong_projection_generation", "wrong_verdict_generation", "unknown_version",
])
def test_adjudication_rejects_invalid_identity_provenance_and_independence(case, damage):
    bundle = case["adjudications"]
    d = bundle["decisions"][0]
    a, b = d["reviews"]
    if damage == "missing_review": d["reviews"].pop()
    elif damage == "duplicate_review": d["reviews"].append(copy.deepcopy(a))
    elif damage == "same_run": b["run_id"] = a["run_id"]
    elif damage == "same_trace": b["trace"], b["trace_sha256"] = a["trace"], a["trace_sha256"]
    elif damage == "same_prompt_id": b["prompt_id"] = a["prompt_id"]
    elif damage == "missing_run": b.pop("run_id")
    elif damage == "missing_prompt": b.pop("prompt")
    elif damage == "missing_trace": b.pop("trace")
    elif damage == "forged_trace": b["trace"] += " tampered"
    elif damage == "forged_prompt": b["prompt"] += " tampered"
    elif damage == "duplicate_decision": bundle["decisions"].append(copy.deepcopy(d))
    elif damage == "wrong_interest": d["interest_id"] = "interest-other"
    elif damage == "wrong_dimension": d["dimension_id"] = "repair_labor_norm_actual"
    elif damage == "wrong_source": d["source_id"] = "source-0"
    elif damage == "wrong_locator": d["locator"] = "chars:0-1"
    elif damage == "wrong_original_verdict": d["original_verdict_sha256"] = "0" * 64
    elif damage == "wrong_fragment": d["fragment_sha256"] = "0" * 64
    elif damage == "wrong_source_hash": d["source_sha256"] = "0" * 64
    elif damage == "wrong_profile_generation": bundle["profile_sha256"] = "0" * 64
    elif damage == "wrong_index_generation": bundle["index_sha256"] = "0" * 64
    elif damage == "wrong_projection_generation": bundle["projection_sha256"] = "0" * 64
    elif damage == "wrong_verdict_generation": bundle["verdicts_sha256"] = "0" * 64
    elif damage == "unknown_version": bundle["version"] = "conflict-adjudication.v999"
    with pytest.raises(ValueError):
        adjudicate(case)


def test_changed_source_bytes_are_rejected_even_if_index_is_unchanged(case):
    from pathlib import Path
    Path(case["source_index"]["sources"][2]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError):
        adjudicate(case)


def test_a_review_cannot_upgrade_a_low_confidence_conflict_into_a_positive(case):
    result = adjudicate(case)
    confirmed = {r[2] for r in refs(result["dimensions"])}
    assert confirmed == {"source-0", "source-1"}
    assert "source-2" not in confirmed and "source-3" not in confirmed
    assert "source-4" not in confirmed


def test_missing_classifier_candidates_cannot_promote_links(case):
    case["original_candidates"] = []
    with pytest.raises(ValueError):
        adjudicate(case)


def test_decision_must_match_trace_contents_even_when_trace_hash_is_valid(case):
    review = case["adjudications"]["decisions"][0]["reviews"][0]
    review["trace"] = review["trace"].replace('\\"no\\"', '\\"yes\\"')
    assert 'yes' in review["trace"]
    review["trace_sha256"] = byte_hash(review["trace"].encode())
    with pytest.raises(ValueError):
        adjudicate(case)
