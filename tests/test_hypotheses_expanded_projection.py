"""Red-first contracts for the historical dimension projection; synthetic data only."""

import copy
import hashlib
import json

import pytest

from tof_reliability.vitaly_bot import hypotheses_topic_projection as projection


DIMENSION = "repair_techcard_equipment_operations_coverage"
TOPIC = "techcard_coverage_verification"


@pytest.fixture
def corpus(tmp_path):
    texts = [
        "Контекст первой встречи. Сравнить ремонты оборудования с технологическими картами. Конец встречи.",
        "Второй фрагмент той же встречи. Проверить ремонты по технологическим картам. Конец.",
        "На другой встрече участник сообщил: руководитель обсуждал сравнение ремонтов и карт. Конец.",
        "На третьей встрече: не сравнивать ремонты оборудования с технологическими картами. Конец.",
    ]
    dates = ["2026-09-01", "2026-09-01", "2026-09-02", "2026-09-03"]
    meetings = ["meeting-a", "meeting-a", "meeting-b", "meeting-c"]
    fragments = [
        "Сравнить ремонты оборудования с технологическими картами.",
        "Проверить ремонты по технологическим картам.",
        "участник сообщил: руководитель обсуждал сравнение ремонтов и карт.",
        "не сравнивать ремонты оборудования с технологическими картами.",
    ]
    sources, evidence = [], []
    for n, (text, fragment) in enumerate(zip(texts, fragments)):
        path = tmp_path / f"source-{n}.txt"
        path.write_text(text, encoding="utf-8")
        start = text.index(fragment)
        sources.append({"id": f"source-{n}", "path": str(path), "date": dates[n],
                        "kind": "transcript", "meeting_id": meetings[n],
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        evidence.append({"source_id": f"source-{n}", "locator": f"chars:{start}-{start + len(fragment)}",
                         "fragment": fragment,
                         "attribution": "third_party" if n == 2 else "direct_request"})
    index = {"version": "synthetic-pre-oct26-v1", "sources": sources}
    profile = {"version": "synthetic-interests-v1", "corpus_version": index["version"],
               "interests": [{"id": "new-interest-1", "requirement": "Сопоставить ремонты и карты",
                              "evidence": evidence}]}
    return profile, index, texts


def run(profile, index, positions, verdict=None):
    evidence = profile["interests"][0]["evidence"]
    candidates = [{"interest_id": "new-interest-1", "dimension_id": DIMENSION, "topic_id": TOPIC,
                   "source_id": evidence[n]["source_id"], "locator": evidence[n]["locator"],
                   "confidence": "high"} for n in positions]
    classified, verified = [], []

    def classifier(request):
        classified.append(copy.deepcopy(request))
        return copy.deepcopy(candidates)

    def verifier(request):
        verified.append(copy.deepcopy(request))
        return (verdict(request) if verdict else
                {"supported": True, "confidence": "high", "conflicting": False})

    result = projection.project_semantic_dimensions(profile, index, classifier=classifier, verifier=verifier)
    return result, classified, verified


def test_input_corpus_version_mismatch_rejected_before_any_component(corpus):
    profile, index, _ = corpus
    profile["corpus_version"] = "other-corpus"
    calls = []
    with pytest.raises(ValueError):
        projection.project_semantic_dimensions(profile, index,
            classifier=lambda x: calls.append(x), verifier=lambda x: calls.append(x))
    assert calls == []


@pytest.mark.parametrize("heldout_day", ["2026-10-26", "2026-10-28", "2026-10-30"])
def test_excluded_meeting_rejected_before_classifier_or_verifier(corpus, heldout_day):
    profile, index, _ = corpus
    index["sources"][0]["date"] = heldout_day
    calls = []
    with pytest.raises(ValueError):
        projection.project_semantic_dimensions(profile, index,
            classifier=lambda x: calls.append(x) or [], verifier=lambda x: calls.append(x))
    assert calls == []


def test_trace_contains_input_versions_and_hashes_and_exact_source_reference(corpus):
    profile, index, _ = corpus
    result, _, verified = run(profile, index, [0])
    assert result["version"] == "repair-analytics-dimensions.v2"
    assert result["corpus_version"] == index["version"]
    assert result["profile_version"] == profile["version"]
    trace = json.dumps(result, ensure_ascii=False)
    assert index["sources"][0]["sha256"] in trace
    assert profile["interests"][0]["evidence"][0]["locator"] in trace
    assert verified[0]["evidence"]["fragment"] == profile["interests"][0]["evidence"][0]["fragment"]


def test_verifier_receives_source_context_and_true_attribution(corpus):
    profile, index, texts = corpus
    result, _, verified = run(profile, index, [2])
    assert verified[0]["evidence"]["attribution"] == "third_party"
    assert texts[2] in json.dumps(verified[0], ensure_ascii=False)
    row = result["dimensions"][0]
    assert row["evidence"][0]["attribution"] == "third_party"
    assert "direct_request" not in json.dumps(row["evidence"], ensure_ascii=False)


def test_repeated_chunks_count_as_one_meeting_and_generator_is_generic(corpus):
    profile, index, texts = corpus
    result, classified, verified = run(profile, index, [0, 1, 2])
    assert result["dimensions"][0]["independent_meeting_count"] == 2
    assert len(verified) == 3
    assert any(item["dimension_id"] == DIMENSION for item in classified[0]["dimensions"])
    payload = json.dumps(projection.dimensions_for_generator(result), ensure_ascii=False)
    for text in texts:
        assert text not in payload
    for entry in profile["interests"][0]["evidence"]:
        assert entry["fragment"] not in payload


def test_explicit_contradiction_keeps_association_unconfirmed(corpus):
    profile, index, _ = corpus

    def verify(request):
        assert {e["source_id"] for e in request["all_evidence"]} >= {"source-0", "source-3"}
        return {"supported": True, "confidence": "high",
                "conflicting": request["evidence"]["source_id"] == "source-3"}

    result, _, _ = run(profile, index, [0, 3], verify)
    assert not result["dimensions"]
    assert {e["source_id"] for item in result["unconfirmed"] for e in item["evidence"]} == {"source-0", "source-3"}
    assert not projection.dimensions_for_generator(result)["dimensions"]
