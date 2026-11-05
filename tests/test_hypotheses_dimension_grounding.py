"""Synthetic source-grounding regressions; no private corpus or model calls."""

import copy
import hashlib
import json

import pytest

from tof_reliability.vitaly_bot import hypotheses_topic_projection as projection


TOPIC = "techcard_coverage_verification"
MATCH = "repair_techcard_equipment_operations_coverage"
REGISTRY = "repair_techcard_registry_completeness"


@pytest.fixture
def sources(tmp_path):
    excerpts = [
        "Сопоставить отремонтированное оборудование и работы с наличием и различиями технологических карт.",
        "Обсудить порядок хранения документов в архиве.",
        "Не сопоставлять ремонтное оборудование и работы с технологическими картами.",
        "Проверить полноту карт относительно независимого перечня оборудования и работ.",
        "По мнению участника, кто-то мог интересоваться сопоставлением ремонтов и карт.",
        "Осмотреть выставочные образцы и подготовить экскурсию.",
    ]
    index = {"version": "synthetic-corpus", "sources": []}
    evidence = []
    for n, excerpt in enumerate(excerpts):
        path = tmp_path / f"synthetic-{n}.txt"
        text = f"Синтетический ввод. {excerpt} Синтетический конец."
        path.write_text(text, encoding="utf-8")
        start = text.index(excerpt)
        source_id = f"source-{n}"
        index["sources"].append({"id": source_id, "path": str(path),
                                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "date": f"2030-01-0{n + 1}", "kind": "transcript"})
        evidence.append({"source_id": source_id, "locator": f"chars:{start}-{start + len(excerpt)}",
                         "fragment": excerpt, "attribution": "third_party" if n == 4 else "direct_request"})
    return index, evidence


def project(sources, refs, candidates, verifier, requirement="Сопоставление ремонта и технологических карт"):
    index, evidence = sources
    profile = {"version": "synthetic-profile", "corpus_version": index["version"],
               "interests": [{"id": "interest-synthetic", "requirement": requirement,
                              "evidence": [copy.deepcopy(evidence[i]) for i in refs]}]}
    proposals = [{"interest_id": "interest-synthetic", "dimension_id": dimension,
                  "topic_id": TOPIC, "source_id": evidence[i]["source_id"],
                  "locator": evidence[i]["locator"], "confidence": "high"}
                 for dimension, i in candidates]
    calls = []

    def classify(request):
        calls.append(copy.deepcopy(request))
        return copy.deepcopy(proposals)

    result = projection.project_semantic_dimensions(profile, index, classifier=classify, verifier=verifier)
    return result, calls


def test_source_request_does_not_need_methodological_baseline_in_the_quote(sources):
    requests = []

    def verify(request):
        requests.append(copy.deepcopy(request))
        # The excerpt supports comparison, while the independent baseline is
        # a method for establishing completeness, not a quotation requirement.
        return {"supported": True, "confidence": "high", "conflicting": False}

    result, calls = project(sources, [0], [(MATCH, 0)], verify)
    assert [row["dimension_id"] for row in result["dimensions"]] == [MATCH]
    assert result["dimensions"][0]["evidence"][0]["attribution"] == "direct_request"
    catalog = {item["dimension_id"]: item for item in calls[0]["dimensions"]}
    assert MATCH in catalog and REGISTRY in catalog
    context = json.dumps([catalog[MATCH], requests[0]], ensure_ascii=False).casefold()
    assert "независим" in context and any(x in context for x in ("перечень", "реестр", "баз"))
    assert any(x in context for x in ("метод", "полнот", "baseline"))


def test_unrelated_reference_does_not_erase_independent_support(sources):
    def verify(request):
        supported = request["evidence"]["source_id"] == "source-0"
        return {"supported": supported, "confidence": "high", "conflicting": False}

    result, _ = project(sources, [0, 1], [(MATCH, 0), (MATCH, 1)], verify)
    assert result["dimensions"][0]["source_ids"] == ["source-0"]
    assert result["unconfirmed"][0]["evidence"][0]["source_id"] == "source-1"


def test_real_contradiction_preserves_uncertainty(sources):
    def verify(request):
        assert {e["source_id"] for e in request["all_evidence"]} == {"source-0", "source-2"}
        return {"supported": True, "confidence": "high",
                "conflicting": request["evidence"]["source_id"] == "source-2"}

    result, _ = project(sources, [0, 2], [(MATCH, 0), (MATCH, 2)], verify)
    assert not result["dimensions"]
    assert {e["source_id"] for row in result["unconfirmed"] for e in row["evidence"]} == {"source-0", "source-2"}
    assert not projection.dimensions_for_generator(result)["dimensions"]


def test_shared_topic_keeps_distinct_dimensions_and_independent_evidence(sources):
    result, _ = project(sources, [0, 3], [(MATCH, 0), (REGISTRY, 3)],
                        lambda _: {"supported": True, "confidence": "high"})
    rows = {row["dimension_id"]: row for row in result["dimensions"]}
    assert set(rows) == {MATCH, REGISTRY}
    assert rows[MATCH]["source_ids"] == ["source-0"]
    assert rows[REGISTRY]["source_ids"] == ["source-3"]
    assert {row["topic_id"] for row in rows.values()} == {TOPIC}


def test_third_party_paraphrase_does_not_become_a_direct_request(sources):
    seen = []

    def verify(request):
        seen.append(copy.deepcopy(request))
        return {"supported": False, "confidence": "high", "conflicting": False}

    result, _ = project(sources, [4], [(MATCH, 4)], verify)
    assert seen[0]["evidence"]["attribution"] == "third_party"
    assert not result["dimensions"] and result["unconfirmed"]


def test_generator_contains_no_source_facts_or_invented_personal_claims(sources):
    secret = "Иван СЕКРЕТ_КОМПАНИИ 987654321 рублей"
    result, _ = project(sources, [0], [(MATCH, 0)],
                        lambda _: {"supported": True, "confidence": "high"},
                        requirement=f"{secret}: сопоставить ремонт и карты")
    result["dimensions"][0]["description"] = secret
    payload = projection.dimensions_for_generator(result)
    encoded = json.dumps(payload, ensure_ascii=False)
    assert payload["dimensions"]
    for forbidden in (secret, "Иван", "СЕКРЕТ_КОМПАНИИ", "987654321", "chars:",
                      "sha256", sources[1][0]["fragment"], str(sources[0]["sources"][0]["path"])):
        assert forbidden not in encoded
    description = payload["dimensions"][0]["description"].casefold()
    assert "витал" not in description and "встреч" not in description


def test_unrelated_nonrepair_scope_is_not_promoted(sources):
    seen = []

    def verify(request):
        seen.append(request)
        return {"supported": False, "confidence": "high", "reason": "nonrepair"}

    result, _ = project(sources, [5], [(MATCH, 5)], verify, requirement="Экскурсия по выставке")
    assert seen and not result["dimensions"]
    assert not projection.dimensions_for_generator(result)["dimensions"]


def test_previous_version_generator_contract_remains_readable():
    old = {"version": "repair-analytics-dimensions.v1", "dimensions": [{
        "dimension_id": REGISTRY, "topic_id": TOPIC,
        "description": "model text is ignored", "interest_ids": ["interest-old"],
        "source_ids": ["source-old"],
    }]}
    payload = projection.dimensions_for_generator(old)
    assert payload["version"] == old["version"]
    assert payload["dimensions"][0]["dimension_id"] == REGISTRY
    assert payload["dimensions"][0]["description"] != "model text is ignored"
