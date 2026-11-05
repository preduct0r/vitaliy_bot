"""Synthetic, private-data-free contract for the repair dimension projection.

The dimension API is separate from the existing topic projection. Classifier and
verifier are isolated test doubles; no model, production source, or holdout is
accessed here. A projection contains a version, confirmed dimension rows and
unconfirmed candidates; the latter never enter the generator payload.
"""
import copy
import hashlib
import json

import pytest

from tof_reliability.vitaly_bot import hypotheses_topic_projection as projection


EPISODES = "repair_episode_count_by_equipment_period"
REPEAT = "repair_repeat_interval"
TOPIC = "maintenance_history_repeated_episodes"


@pytest.fixture
def corpus(tmp_path):
    texts = [
        "Посчитать эпизоды ремонта оборудования по периодам. Измерить интервалы между повторными ремонтами.",
        "Сравнить число ремонтов оборудования по периодам и интервалы повторных ремонтов.",
        "Закрытая встреча: частная информация НЕ_ПЕРЕДАВАТЬ.",
    ]
    sources = []
    for i, text in enumerate(texts, 1):
        path = tmp_path / f"source-{i}.txt"
        path.write_text(text, encoding="utf-8")
        sources.append({"id": f"source-{i}", "date": f"2030-01-0{i}",
                        "kind": "transcript", "meeting_id": f"meeting-{i}",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "path": str(path)})

    def evidence(i, fragment, attribution="direct_request"):
        start = texts[i - 1].index(fragment)
        return {"source_id": f"source-{i}", "locator": f"chars:{start}-{start + len(fragment)}",
                "fragment": fragment, "attribution": attribution}

    first = evidence(1, "Посчитать эпизоды ремонта оборудования по периодам.")
    second = evidence(1, "Измерить интервалы между повторными ремонтами.")
    independent = evidence(2, texts[1], "interpretation")
    profile = {"version": "synthetic-profile", "corpus_version": "synthetic-corpus",
               "interests": [
                   {"id": "interest-count", "requirement": "Число ремонтных эпизодов по периодам",
                    "evidence": [first, independent]},
                   {"id": "interest-interval", "requirement": "Интервалы между ремонтами",
                    "evidence": [second]},
               ]}
    return profile, {"version": "synthetic-corpus", "sources": sources}, first, second, independent


def candidate(interest, dimension, evidence):
    return {"interest_id": interest, "dimension_id": dimension, "topic_id": TOPIC,
            "source_id": evidence["source_id"], "locator": evidence["locator"],
            "confidence": "high"}


class Component:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, data):
        self.calls.append(copy.deepcopy(data))
        return copy.deepcopy(self.result)


def run(profile, index, candidates, verdict=None):
    assert hasattr(projection, "project_semantic_dimensions"), "dimension projection is not implemented"
    classifier = Component(candidates)
    verifier = Component(verdict if verdict is not None else {"supported": True, "confidence": "high"})
    result = projection.project_semantic_dimensions(profile, index, classifier=classifier, verifier=verifier)
    return result, classifier, verifier


def test_two_dimensions_in_one_topic_and_independent_references_survive(corpus):
    profile, index, first, second, independent = corpus
    result, classifier, verifier = run(profile, index, [
        candidate("interest-count", EPISODES, first),
        candidate("interest-count", EPISODES, independent),
        candidate("interest-interval", REPEAT, second),
    ])
    assert result["version"]
    rows = {row["dimension_id"]: row for row in result["dimensions"]}
    assert set(rows) == {EPISODES, REPEAT}
    assert {row["topic_id"] for row in rows.values()} == {TOPIC}
    assert set(rows[EPISODES]["interest_ids"]) == {"interest-count"}
    assert set(rows[EPISODES]["source_ids"]) == {"source-1", "source-2"}
    assert set(rows[REPEAT]["interest_ids"]) == {"interest-interval"}
    assert len(classifier.calls) == 1 and len(verifier.calls) >= 2
    verifier_data = json.dumps(verifier.calls, ensure_ascii=False)
    assert first["fragment"] in verifier_data and independent["fragment"] in verifier_data
    assert "direct_request" in verifier_data and "interpretation" in verifier_data


@pytest.mark.parametrize("damage", ["fragment", "locator", "hash", "unknown_source"])
def test_invalid_exact_source_evidence_is_rejected_before_model_calls(corpus, damage):
    profile, index, first, _, _ = corpus
    if damage == "fragment":
        profile["interests"][0]["evidence"][0]["fragment"] += " выдумка"
    elif damage == "locator":
        profile["interests"][0]["evidence"][0]["locator"] = "chars:0-9999"
    elif damage == "hash":
        index["sources"][0]["sha256"] = "0" * 64
    else:
        profile["interests"][0]["evidence"][0]["source_id"] = "unknown"
    assert hasattr(projection, "project_semantic_dimensions"), "dimension projection is not implemented"
    classifier, verifier = Component([candidate("interest-count", EPISODES, first)]), Component({"supported": True, "confidence": "high"})
    with pytest.raises(ValueError):
        projection.project_semantic_dimensions(profile, index, classifier=classifier, verifier=verifier)
    assert not classifier.calls and not verifier.calls


@pytest.mark.parametrize("verdict", [
    {"supported": False, "confidence": "high"},
    {"supported": True, "confidence": "low"},
    {"supported": None, "confidence": "unknown"},
    {"supported": True, "confidence": "high", "conflicting": True},
])
def test_rejected_uncertain_and_conflicting_evidence_is_unconfirmed(corpus, verdict):
    profile, index, first, _, _ = corpus
    result, _, _ = run(profile, index, [candidate("interest-count", EPISODES, first)], verdict)
    assert not result["dimensions"]
    assert result["unconfirmed"]
    assert hasattr(projection, "dimensions_for_generator"), "generator boundary is not implemented"
    assert not projection.dimensions_for_generator(result)["dimensions"]


def test_duplicate_quotes_from_one_meeting_do_not_count_as_independent_meetings(corpus):
    profile, index, first, second, independent = corpus
    # Two distinct excerpts from a single meeting are retained as evidence, but
    # do not constitute two independent meetings. A second meeting does.
    profile["interests"][0]["evidence"].append(second)
    result, _, verifier = run(profile, index, [
        candidate("interest-count", EPISODES, first),
        candidate("interest-count", EPISODES, second),
        candidate("interest-count", EPISODES, independent),
    ])
    row = next(row for row in result["dimensions"] if row["dimension_id"] == EPISODES)
    assert set(row["source_ids"]) == {"source-1", "source-2"}
    assert row["independent_meeting_count"] == 2
    seen = json.dumps(verifier.calls, ensure_ascii=False)
    assert first["fragment"] in seen and second["fragment"] in seen


def test_generator_receives_only_curated_generic_descriptions_and_traceable_ids(corpus):
    profile, index, first, _, _ = corpus
    secret = "Иван СЕКРЕТ 987654321 рублей"
    profile["interests"][0]["requirement"] += " " + secret
    proposals = [candidate("interest-count", EPISODES, first)]
    proposals[0]["description"] = secret
    result, _, _ = run(profile, index, proposals)
    assert hasattr(projection, "dimensions_for_generator"), "generator boundary is not implemented"
    payload = projection.dimensions_for_generator(result)
    assert payload["dimensions"]
    encoded = json.dumps(payload, ensure_ascii=False)
    for forbidden in (secret, "Иван", "СЕКРЕТ", "987654321", first["fragment"],
                      first["locator"], "sha256", str(index["sources"][0]["path"])):
        assert forbidden not in encoded
    assert "source-1" in encoded and "interest-count" in encoded


def test_holdout_sources_never_reach_classifier_verifier_or_generator(corpus):
    profile, index, first, _, _ = corpus
    # The source index is caller-selected: passing a reference to an excluded
    # holdout must fail before any private component receives input.
    index["sources"] = index["sources"][:2]
    profile["interests"][0]["evidence"].append({"source_id": "source-3", "locator": "chars:0-5",
                                               "fragment": "Закрыт", "attribution": "third_party"})
    assert hasattr(projection, "project_semantic_dimensions"), "dimension projection is not implemented"
    classifier, verifier = Component([candidate("interest-count", EPISODES, first)]), Component({"supported": True, "confidence": "high"})
    with pytest.raises(ValueError):
        projection.project_semantic_dimensions(profile, index, classifier=classifier, verifier=verifier)
    assert not classifier.calls and not verifier.calls


def test_nonrepair_interest_cannot_be_promoted_even_if_classifier_proposes_it(corpus):
    profile, index, first, _, _ = corpus
    profile["interests"][0]["requirement"] = "План музейных экскурсий"
    result, _, _ = run(profile, index, [candidate("interest-count", EPISODES, first)],
                       {"supported": False, "confidence": "high", "reason": "nonrepair"})
    assert not result["dimensions"]


def test_unsupported_reference_does_not_erase_independent_supported_evidence(corpus):
    profile, index, first, _, independent = corpus
    proposals = [candidate("interest-count", EPISODES, first),
                 candidate("interest-count", EPISODES, independent)]
    def verify(data):
        return {"supported": data["evidence"]["source_id"] == "source-1", "confidence": "high"}
    result = projection.project_semantic_dimensions(profile, index, classifier=Component(proposals), verifier=verify)
    assert len(result["dimensions"]) == 1
    assert result["dimensions"][0]["source_ids"] == ["source-1"]
    assert len(result["unconfirmed"]) == 1
    assert result["unconfirmed"][0]["evidence"][0]["source_id"] == "source-2"


def test_conflict_holds_all_evidence_in_interest_dimension(corpus):
    profile, index, first, _, independent = corpus
    proposals = [candidate("interest-count", EPISODES, first),
                 candidate("interest-count", EPISODES, independent)]
    def verify(data):
        return {"supported": True, "confidence": "high",
                "conflicting": data["evidence"]["source_id"] == "source-2"}
    result = projection.project_semantic_dimensions(profile, index, classifier=Component(proposals), verifier=verify)
    assert not result["dimensions"]
    assert result["unconfirmed"]
