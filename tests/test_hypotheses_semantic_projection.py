"""Synthetic contract for private, source-backed semantic projection.

The callable classifier and verifier below are test doubles; real source data and
model calls do not belong in this test suite.
"""
import copy
import hashlib
import json

import pytest

from tof_reliability.vitaly_bot import hypotheses_topic_projection as projection


TOPIC = "maintenance_history_repeated_episodes"
LABEL = "История ремонтов оборудования и повторные ремонтные эпизоды."


@pytest.fixture
def case(tmp_path):
    source = ("Начало. Вернуться к поломкам одного агрегата и понять, почему "
              "его приходится чинить снова. Компания СИНТЕТИКА, 987654321 рублей.")
    fragment = "Вернуться к поломкам одного агрегата и понять, почему его приходится чинить снова."
    path = tmp_path / "synthetic.txt"
    path.write_text(source, encoding="utf-8")
    start = source.index(fragment)
    locator = f"chars:{start}-{start + len(fragment)}"
    index = {"version": "synthetic-corpus", "sources": [{
        "id": "source-1", "date": "2030-01-01", "kind": "transcript",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "path": str(path),
    }]}
    profile = {"version": "synthetic-profile", "corpus_version": "synthetic-corpus",
               "interests": [{"id": "interest-1", "requirement": "Выяснить причины повторных отказов машины",
                              "evidence": [{"source_id": "source-1", "locator": locator,
                                            "attribution": "direct_request", "fragment": fragment}]}]}
    candidate = {"interest_id": "interest-1", "topic_id": TOPIC,
                 "source_id": "source-1", "locator": locator, "confidence": "high"}
    return profile, index, candidate


def semantic(profile, index, classifier, verifier):
    # Kept distinct from project_topics so its existing narrow API stays intact.
    assert hasattr(projection, "project_semantic_topics"), "semantic projection interface is missing"
    return projection.project_semantic_topics(profile, index, classifier=classifier, verifier=verifier)


class Classifier:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = []

    def __call__(self, private_input):
        self.calls.append(copy.deepcopy(private_input))
        return copy.deepcopy(self.candidates)


class Verifier:
    def __init__(self, verdict=None):
        self.verdict = verdict if verdict is not None else {"supported": True, "confidence": "high"}
        self.calls = []

    def __call__(self, private_input):
        self.calls.append(copy.deepcopy(private_input))
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return copy.deepcopy(self.verdict)


def test_synonyms_require_independent_source_backed_semantic_verification(case):
    profile, index, candidate = case
    assert projection.project_topics(profile, index) == []
    classifier, verifier = Classifier([candidate]), Verifier()
    result = semantic(profile, index, classifier, verifier)
    assert result == [{"topic_id": TOPIC, "interest_id": "interest-1", "description": LABEL}]
    assert len(classifier.calls) == len(verifier.calls) == 1
    verifier_input = json.dumps(verifier.calls, ensure_ascii=False)
    assert profile["interests"][0]["requirement"] in verifier_input
    assert profile["interests"][0]["evidence"][0]["fragment"] in verifier_input
    assert TOPIC in verifier_input


@pytest.mark.parametrize("damage", ["locator", "source", "interest", "topic", "duplicate", "confidence"])
def test_candidate_must_be_bound_to_verified_evidence_and_supported_ids(case, damage):
    profile, index, candidate = case
    if damage == "locator":
        candidate["locator"] = "chars:0-6"
    elif damage == "source":
        candidate["source_id"] = "unknown-source"
    elif damage == "interest":
        candidate["interest_id"] = "unknown-interest"
    elif damage == "topic":
        candidate["topic_id"] = "model-invented-topic"
    elif damage == "confidence":
        candidate["confidence"] = "low"
    candidates = [candidate, copy.deepcopy(candidate)] if damage == "duplicate" else [candidate]
    verifier = Verifier()
    try:
        result = semantic(profile, index, Classifier(candidates), verifier)
    except ValueError:
        return
    assert result == []


@pytest.mark.parametrize("damage", ["hash", "locator", "fragment", "version", "duplicate_source", "duplicate_interest"])
def test_profile_provenance_is_rechecked_before_classification(case, damage):
    profile, index, candidate = case
    if damage == "hash":
        index["sources"][0]["sha256"] = "0" * 64
    elif damage == "locator":
        profile["interests"][0]["evidence"][0]["locator"] = "chars:0-99999"
    elif damage == "fragment":
        profile["interests"][0]["evidence"][0]["fragment"] = "fabricated quotation"
    elif damage == "version":
        profile["corpus_version"] = "different"
    elif damage == "duplicate_source":
        index["sources"].append(copy.deepcopy(index["sources"][0]))
    else:
        profile["interests"].append(copy.deepcopy(profile["interests"][0]))
    classifier, verifier = Classifier([candidate]), Verifier()
    try:
        result = semantic(profile, index, classifier, verifier)
    except ValueError:
        result = []
    assert result == []
    assert classifier.calls == []
    assert verifier.calls == []


@pytest.mark.parametrize("verdict", [
    {"supported": False, "confidence": "high"},
    {"supported": True, "confidence": "low"},
    {"supported": None, "confidence": "unknown"},
    {"supported": True},
    RuntimeError("private verifier failed"),
])
def test_rejected_weak_unknown_or_failed_verification_excludes_candidate(case, verdict):
    profile, index, candidate = case
    try:
        result = semantic(profile, index, Classifier([candidate]), Verifier(verdict))
    except ValueError:
        return
    assert result == []


def test_generic_mining_and_contradictory_evidence_are_not_promoted(case):
    profile, index, candidate = case
    # An adversarial classifier proposes a repair topic despite a verifier finding
    # that the actual evidence contradicts it or is merely generic mining.
    for verdict in ({"supported": False, "confidence": "high", "reason": "generic mining"},
                    {"supported": False, "confidence": "high", "reason": "contradiction"}):
        assert semantic(profile, index, Classifier([candidate]), Verifier(verdict)) == []


def test_downstream_output_contains_only_fixed_labels_and_interest_ids(case):
    profile, index, candidate = case
    candidate["description"] = "СЕКРЕТНАЯ_КОМПАНИЯ 987654321 fabricated label"
    result = semantic(profile, index, Classifier([candidate]), Verifier())
    assert result == [{"topic_id": TOPIC, "interest_id": "interest-1", "description": LABEL}]
    serialized = json.dumps(result, ensure_ascii=False)
    for forbidden in ("СИНТЕТИКА", "987654321", "СЕКРЕТНАЯ_КОМПАНИЯ", "fabricated label",
                      "source-1", "chars:", "fragment", "sha256", "synthetic.txt",
                      profile["interests"][0]["requirement"], profile["interests"][0]["evidence"][0]["fragment"]):
        assert forbidden not in serialized


def test_unreferenced_source_hash_is_also_checked(case, tmp_path):
    profile, index, candidate = case
    path = tmp_path / "other.txt"
    path.write_text("other", encoding="utf-8")
    index["sources"].append({"id": "other", "path": str(path), "sha256": "0" * 64,
                             "date": "2030-01-02", "kind": "transcript"})
    classifier = Classifier([candidate])
    with pytest.raises(ValueError):
        semantic(profile, index, classifier, Verifier())
    assert not classifier.calls


def test_classifier_and_verifier_must_be_distinct(case):
    profile, index, candidate = case
    component = Classifier([candidate])
    with pytest.raises(ValueError):
        semantic(profile, index, component, component)


def test_duplicate_interest_topic_with_different_citation_fails_closed(case):
    profile, index, candidate = case
    other = copy.deepcopy(candidate)
    other["locator"] = "chars:0-6"
    with pytest.raises(ValueError, match="duplicate candidate"):
        semantic(profile, index, Classifier([candidate, other]), Verifier())


def test_verifier_receives_all_verified_evidence_for_contradictions(case, tmp_path):
    profile, index, candidate = case
    source = tmp_path / 'contradiction.txt'
    text = 'Не анализировать повторы отказов.'
    source.write_text(text, encoding='utf-8')
    index['sources'].append({'id': 'source-2', 'date': '2030-01-02', 'kind': 'transcript',
                             'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    profile['interests'][0]['evidence'].append({'source_id': 'source-2', 'locator': f'chars:0-{len(text)}',
                                               'attribution': 'direct_request', 'fragment': text})
    def reject_contradiction(data):
        assert any(e['fragment'] == text for e in data['all_evidence'])
        return {'supported': False, 'confidence': 'high'}
    assert semantic(profile, index, Classifier([candidate]), reject_contradiction) == []
    verifier = Verifier()
    semantic(profile, index, Classifier([candidate]), verifier)
    assert any(e['fragment'] == text for e in verifier.calls[0]['all_evidence'])
