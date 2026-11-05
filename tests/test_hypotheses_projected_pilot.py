"""Offline projected-pilot contract; all values and sources are synthetic.

The generator only sees canonical labels. The separate audit and source map
are private and bind a selected (interest, topic) to a verified citation.
The verifier is an independent test double, not a guarantee of correctness.
"""

import copy
import json

import pytest

from pilot_projection_support import distinct_directions, projection
from test_hypotheses_pilot import inputs as legacy_inputs, cards as legacy_cards

from tof_reliability.vitaly_bot import hypotheses_pilot as pilot


TOPIC = "maintenance_history_repeated_episodes"
LABEL = "История ремонтов оборудования и повторные ремонтные эпизоды."


@pytest.fixture
def case():
    evidence = {"source_id": "meeting-a", "date": "2030-01-01",
                "locator": "chars:0-10", "attribution": "interpretation",
                "sha256": "a" * 64, "reference": "meeting-a:chars:0-10"}
    interests = {"version": "interests-1", "corpus_version": "corpus-1", "interests": [
        {"id": "repeat", "requirement": "PRIVATE_REQUIREMENT фирма СЕКРЕТ 987654321",
         "attributions": ["interpretation"], "evidence": [evidence],
         "fragment": "PRIVATE_TRANSCRIPT", "asset": "PRIVATE_ASSET"},
        {"id": "other", "requirement": "OTHER_CASE_PRIVATE_REQUEST 87654321",
         "attributions": ["interpretation"], "evidence": [dict(evidence)]},
    ]}
    legacy = legacy_inputs.__wrapped__()
    evidence = copy.deepcopy(legacy['interest_context']['interests'][0]['evidence'][0])
    interests['interests'][0]['evidence'] = [evidence]
    interests['interests'][0]['requirement'] += ' ' + legacy['interest_context']['interests'][0]['requirement']
    interests['interests'].extend(copy.deepcopy(legacy['interest_context']['interests'][1:]))
    labels = [{"interest_id": "repeat", "topic_id": TOPIC, "description": LABEL}]
    audit = [{"interest_id": "repeat", "topic_id": TOPIC, "source_id": evidence["source_id"],
              "locator": evidence["locator"], "verdict": {"supported": True, "confidence": "high"}}]
    extra = projection({'interests': interests['interests'][2:]})
    labels.extend(extra['verified_labels'])
    audit.extend(extra['verifier_audit'])
    coverage = legacy['coverage']
    cards = legacy_cards.__wrapped__()
    cards[0]['meeting_interest'] = {'interest_id': 'repeat', 'topic_id': TOPIC}
    return {'interests': interests, 'labels': labels, 'audit': audit,
            'coverage': coverage, 'cards': cards, 'evidence': evidence}



class Verifier:
    def __init__(self, verdict=None):
        self.calls = []
        self.verdict = verdict if verdict is not None else {"supported": True, "confidence": "high"}

    def __call__(self, context):
        self.calls.append(copy.deepcopy(context))
        if isinstance(self.verdict, Exception):
            raise self.verdict
        return copy.deepcopy(self.verdict)


def generate(case, *, verifier=None, generator=None, scope="насосы", **overrides):
    args = dict(scope=scope, knowledge_version="knowledge-1",
                interest_context=case["interests"], verified_labels=case["labels"],
                verifier_audit=case["audit"],
                private_verifier=verifier if verifier is not None else Verifier(),
                coverage_provider=lambda: copy.deepcopy(case["coverage"]),
                generator=generator or (lambda _: copy.deepcopy(case["cards"])))
    args.update(overrides)
    return pilot.generate_pilot(**args)


def test_only_canonical_projection_reaches_generator_and_selected_source_is_attached(case):
    calls = []
    verifier = Verifier()
    run = generate(case, verifier=verifier,
                   generator=lambda context: calls.append(copy.deepcopy(context)) or copy.deepcopy(case["cards"]))
    payload = json.dumps(calls, ensure_ascii=False)
    assert LABEL in payload and TOPIC in payload and "repeat" in payload
    for secret in ("PRIVATE_REQUIREMENT", "СЕКРЕТ", "987654321", "PRIVATE_TRANSCRIPT",
                   "PRIVATE_ASSET", "OTHER_CASE_PRIVATE_REQUEST", "87654321", "meeting-a", "chars:0-10"):
        assert secret not in payload
    assert len(verifier.calls) >= len(run["cards"])
    for card in run["cards"]:
        meeting = card["meeting_interest"]
        item = next(i for i in case["interests"]["interests"] if i["id"] == meeting["interest_id"])
        assert any(all(meeting.get(k) == v for k, v in evidence.items() if k != "fragment")
                   for evidence in item["evidence"])
    assert run["cards"][0]["meeting_interest"]["topic_id"] == TOPIC


@pytest.mark.parametrize("damage", ["missing", "wrong_interest", "wrong_topic", "wrong_label",
                                    "wrong_source", "wrong_locator", "rejected", "duplicate"])
def test_missing_unbound_or_ambiguous_projection_fails_before_generator(case, damage):
    if damage == "missing":
        case["labels"] = []
    elif damage == "wrong_interest":
        case["labels"][0]["interest_id"] = "other"
    elif damage == "wrong_topic":
        case["labels"][0]["topic_id"] = "invented-topic"
    elif damage == "wrong_label":
        case["labels"][0]["description"] = "PRIVATE_REQUIREMENT 987654321"
    elif damage == "wrong_source":
        case["audit"][0]["source_id"] = "unknown"
    elif damage == "wrong_locator":
        case["audit"][0]["locator"] = "chars:10-20"
    elif damage == "rejected":
        case["audit"][0]["verdict"]["supported"] = False
    else:
        case["audit"].append(copy.deepcopy(case["audit"][0]))
    calls = []
    with pytest.raises((ValueError, RuntimeError)):
        generate(case, generator=lambda ctx: calls.append(ctx) or case["cards"])
    assert not calls


@pytest.mark.parametrize("verdict", [None, {}, {"supported": False, "confidence": "high"},
                                     {"supported": True, "confidence": "low"},
                                     RuntimeError("verifier unavailable")])
def test_missing_invalid_weak_or_failed_independent_verifier_rejects(case, verdict):
    verifier = None if verdict is None else Verifier(verdict)
    with pytest.raises((ValueError, RuntimeError)):
        generate(case, private_verifier=verifier)


@pytest.mark.parametrize("field,text", [
    ("management_question", "Как улучшить рецепты приготовления борща?"),
    ("analyst_brief", "В прошлом расходы составили 987654321 рублей."),
    ("decision", "Компания СЕКРЕТ уже отремонтировала оборудование."),
    ("metrics_method", "Использовать чужой запрос OTHER_CASE_PRIVATE_REQUEST и 87654321 рублей."),
])
def test_off_topic_cross_user_and_unverified_historical_claims_are_rejected(case, field, text):
    case["cards"][0][field] = text
    # Even an erroneously optimistic verifier cannot override deterministic
    # topic compatibility or unsupported numeric values.
    with pytest.raises((ValueError, RuntimeError)):
        generate(case, verifier=Verifier())


def test_same_case_request_amount_can_be_used_and_verifier_sees_only_approved_facts(case, tmp_path):
    request = "Как проверить ремонты насосов при бюджете 120000 рублей?"
    case["cards"][0]["management_question"] = request
    verifier = Verifier()
    store = pilot.TrustedRequestStore(tmp_path / 'trusted.sqlite')
    trusted = store.create({'run_id': 'run', 'case_id': 'case', 'user_id': 'user', 'scope': request})
    run = generate(case, scope=request, verifier=verifier, trusted_context=trusted, trusted_store=store)
    assert "120000" in run["cards"][0]["management_question"]
    assert verifier.calls
    context = json.dumps(verifier.calls, ensure_ascii=False)
    assert "120000" in context
    for private in ("OTHER_CASE_PRIVATE_REQUEST", "87654321", "987654321", "PRIVATE_TRANSCRIPT"):
        assert private not in context


def test_generator_and_verifier_must_be_independent(case):
    component = lambda context: copy.deepcopy(case["cards"])
    with pytest.raises((ValueError, RuntimeError)):
        generate(case, generator=component, private_verifier=component)


def test_cross_route_reference_remains_rejected(case):
    case["cards"][0]["source_references"] = ["ledger:unrelated"]
    with pytest.raises((ValueError, RuntimeError)):
        generate(case)


def test_cli_requires_projection_and_private_verifier():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, 'scripts/hypotheses_pilot.py', '--help'],
                            capture_output=True, text=True, check=True)
    for option in ('--verified-labels', '--verifier-audit', '--private-verifier'):
        assert option in result.stdout


def test_verifier_receives_only_case_facts_and_selected_route(case):
    verifier = Verifier()
    generate(case, verifier=verifier)
    for context in verifier.calls:
        assert 'meeting-a' not in json.dumps(context, ensure_ascii=False)
        assert 'chars:0-10' not in json.dumps(context, ensure_ascii=False)


@pytest.mark.parametrize('claim', ['Ранее потрачено миллион рублей.', 'Объём равен 1e6 рублей.'])
def test_written_or_scientific_invented_amount_rejected(case, claim):
    case['cards'][0]['analyst_brief'] = claim
    with pytest.raises(ValueError):
        generate(case)


def test_projection_binding_cannot_be_swapped_after_generation(case):
    run = generate(case)
    run['cards'][0]['meeting_interest']['topic_id'] = 'equipment_downtime'
    with pytest.raises(ValueError):
        pilot.validate_pilot(run, coverage=case['coverage'], interest_context=case['interests'])


def test_private_verifier_executable_json_protocol(tmp_path):
    executable = tmp_path / 'verifier'
    executable.write_text('#!/usr/bin/env python3\nimport json, sys\nvalue=json.load(sys.stdin)\nprint(json.dumps({"supported":value["scope"] == "насосы", "confidence":"high"}))\n')
    executable.chmod(0o700)
    assert pilot.subprocess_verifier(executable)({'scope': 'насосы'}) == {'supported': True, 'confidence': 'high'}
    executable.write_text('#!/usr/bin/env python3\nprint("not json")\n')
    with pytest.raises(ValueError):
        pilot.subprocess_verifier(executable)({'scope': 'насосы'})
    with pytest.raises(ValueError):
        pilot.subprocess_verifier(tmp_path / 'missing')


def test_spelled_invented_numeric_value_rejected(case):
    case['cards'][0]['analyst_brief'] = 'В прошлом было пять ремонтов.'
    with pytest.raises(ValueError):
        generate(case)


def test_extra_projection_fields_never_reach_generator(case):
    case['labels'][0]['private_note'] = 'PRIVATE_LABEL_SECRET'
    seen = []
    generate(case, generator=lambda context: seen.append(context) or copy.deepcopy(case['cards']))
    assert 'PRIVATE_LABEL_SECRET' not in json.dumps(seen, ensure_ascii=False)


def test_projected_run_cannot_drop_topic_binding(case):
    run = generate(case)
    del run['cards'][0]['meeting_interest']['topic_id']
    with pytest.raises(ValueError):
        pilot.validate_pilot(run, coverage=case['coverage'], interest_context=case['interests'])
