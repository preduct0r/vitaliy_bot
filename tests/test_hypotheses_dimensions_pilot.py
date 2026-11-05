"""Offline dimension pilot contracts. All evidence and coverage are synthetic.

The new, versioned entry points deliberately leave the legacy topic pilot intact:
``reconcile_dimension_projections`` verifies private evidence before producing a
projection, and ``generate_dimension_pilot`` consumes that projection. Neither
test inputs nor test doubles access the server, a model, or a historical holdout.
"""

import copy
import hashlib
import json

import pytest

from tof_reliability.vitaly_bot import hypotheses_pilot as pilot
from tof_reliability.vitaly_bot import hypotheses_topic_projection as topics
from test_hypotheses_seven_compatible import case as topic_case


FIRST = "repair_episode_count_by_equipment_period"
SECOND = "repair_repeat_interval"
TOPIC = "maintenance_history_repeated_episodes"
TECHCARD = "repair_techcard_equipment_operations_coverage"


@pytest.fixture
def evidence_case(tmp_path):
    text = ("Посчитать эпизоды ремонта по периодам. "
            "Измерить интервалы повторных ремонтов. "
            "Сопоставить ремонтируемое оборудование и операции с техкартами.")
    path = tmp_path / "source.txt"
    path.write_text(text, encoding="utf-8")
    source = {"id": "source-a", "path": str(path), "kind": "transcript",
              "date": "2030-01-01", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def item(ident, fragment, attribution):
        start = text.index(fragment)
        evidence = {"source_id": "source-a", "locator": f"chars:{start}-{start + len(fragment)}",
                    "fragment": fragment, "attribution": attribution}
        return {"id": ident, "requirement": fragment, "evidence": [evidence]}

    interests = [item("count", "Посчитать эпизоды ремонта по периодам.", "direct_request"),
                 item("interval", "Измерить интервалы повторных ремонтов.", "interpretation"),
                 item("coverage", "Сопоставить ремонтируемое оборудование и операции с техкартами.", "third_party")]
    profile = {"version": "profile-1", "corpus_version": "corpus-1", "interests": interests}
    index = {"version": "corpus-1", "sources": [source]}

    def project(selected):
        candidates = [{"interest_id": interests[i]["id"], "dimension_id": dim,
                       "topic_id": topic, "source_id": "source-a",
                       "locator": interests[i]["evidence"][0]["locator"], "confidence": "high"}
                      for i, dim, topic in selected]
        return topics.project_semantic_dimensions(
            profile, index, classifier=lambda _: candidates,
            verifier=lambda _: {"supported": True, "confidence": "high"})

    v1 = project([(0, FIRST, TOPIC), (1, SECOND, TOPIC)])
    v2 = project([(2, TECHCARD, "techcard_coverage_verification")])

    def audit(selected):
        return [{"interest_id": interests[i]["id"], "dimension_id": dim,
                 "source_id": "source-a", "locator": interests[i]["evidence"][0]["locator"],
                 "verdict": {"supported": True, "confidence": "high", "conflicting": False}}
                for i, dim in selected]

    return {"profile": profile, "index": index, "v1": v1, "v2": v2,
            "v1_audit": audit([(0, FIRST), (1, SECOND)]),
            "v2_audit": audit([(2, TECHCARD)])}


def reconcile(data):
    assert hasattr(pilot, "reconcile_dimension_projections"), "versioned dimension reconciliation is not implemented"
    return pilot.reconcile_dimension_projections(
        profile=data["profile"], source_index=data["index"],
        v1_projection=data["v1"], v1_verdicts=data["v1_audit"],
        v2_projection=data["v2"], v2_verdicts=data["v2_audit"])


def test_reconciliation_keeps_two_dimensions_in_one_topic_and_v2_attribution(evidence_case):
    result = reconcile(evidence_case)
    rows = {r["dimension_id"]: r for r in result["dimensions"]}
    assert {FIRST, SECOND, TECHCARD} <= set(rows)
    assert rows[FIRST]["topic_id"] == rows[SECOND]["topic_id"] == TOPIC
    assert rows[FIRST]["description"] != rows[SECOND]["description"]
    assert rows[TECHCARD]["evidence"][0]["attribution"] == "third_party"
    assert result["version"] and result["version"] != evidence_case["v1"]["version"]


@pytest.mark.parametrize("damage", ["source_hash", "locator", "fragment", "interest", "unknown_version", "missing_verdict"])
def test_reconciliation_rejects_incompatible_or_unverified_sources(evidence_case, damage):
    data = copy.deepcopy(evidence_case)
    if damage == "source_hash":
        data["index"]["sources"][0]["sha256"] = "0" * 64
    elif damage == "locator":
        data["v2"]["dimensions"][0]["evidence"][0]["locator"] = "chars:0-999"
    elif damage == "fragment":
        data["v2"]["dimensions"][0]["evidence"][0]["fragment"] += " fabricated"
    elif damage == "interest":
        data["v2"]["dimensions"][0]["evidence"][0]["interest_id"] = "count"
    elif damage == "unknown_version":
        data["v2"]["version"] = "unsupported-version"
    else:
        data["v2_audit"] = []
    with pytest.raises(ValueError):
        reconcile(data)


@pytest.mark.parametrize("verdict", [
    {"supported": False, "confidence": "high"},
    {"supported": True, "confidence": "low"},
    {"supported": True, "confidence": "high", "conflicting": True},
])
def test_v2_only_added_after_its_own_high_confidence_nonconflicting_verdict(evidence_case, verdict):
    evidence_case["v2_audit"][0]["verdict"] = verdict
    result = reconcile(evidence_case)
    assert TECHCARD not in {row["dimension_id"] for row in result["dimensions"]}
    assert any(row["dimension_id"] == TECHCARD for row in result["unconfirmed"])
    assert {FIRST, SECOND} <= {row["dimension_id"] for row in result["dimensions"]}


@pytest.fixture
def dimension_case(topic_case):
    context, _, _, coverage, cards = copy.deepcopy(topic_case)
    # Reuse real catalog identifiers and synthetic source-bound evidence. The
    # second episode dimension shares a broad topic but has a different method.
    ids = [FIRST, SECOND, "repair_plan_fact_reasons", "repair_material_cost_supply",
           "repair_downtime", "repair_criticality_decisions"]
    catalog = topics._CATALOGS[topics.DIMENSION_VERSION]
    assert all(ident in catalog for ident in ids)
    select = [0, 0, 1, 2, 3, 5]
    rows, selected_cards = [], []
    for number, (ident, source_number) in enumerate(zip(ids, select), 1):
        item = context["interests"][source_number]
        topic, description = catalog[ident]
        evidence = item["evidence"][0]
        rows.append({"dimension_id": ident, "topic_id": topic, "description": description,
                     "interest_ids": [item["id"]], "source_ids": [evidence["source_id"]],
                     "evidence": [{"interest_id": item["id"], **evidence}],
                     "independent_meeting_count": 1})
        card = copy.deepcopy(cards[source_number])
        card["number"] = number
        card["meeting_interest"] = {"interest_id": item["id"], "topic_id": topic, "dimension_id": ident}
        card["decision"] = f"Выбрать проверку: {description}"
        card["metrics_method"] = f"Проверить методом: {description}"
        selected_cards.append(card)
    projection = {"version": pilot.DIMENSION_PILOT_VERSION, "dimensions": rows, "unconfirmed": []}
    return context, coverage, projection, selected_cards


def generate(case, *, cards=None, verifier=None, generator=None, scope="насосы", **kwargs):
    assert hasattr(pilot, "generate_dimension_pilot"), "offline versioned dimension pilot is not implemented"
    context, coverage, projection, defaults = copy.deepcopy(case)
    return pilot.generate_dimension_pilot(
        scope=scope, knowledge_version="k1", interest_context=context,
        dimension_projection=projection, coverage_provider=lambda: copy.deepcopy(coverage),
        generator=generator or (lambda _: copy.deepcopy(defaults if cards is None else cards)),
        private_verifier=verifier or (lambda _: {"supported": True, "confidence": "high"}), **kwargs)


def test_same_topic_distinct_dimensions_are_accepted_and_citations_attached(dimension_case):
    run = generate(dimension_case)
    selected = [c for c in run["cards"] if c["meeting_interest"]["topic_id"] == TOPIC]
    assert {c["meeting_interest"]["dimension_id"] for c in selected} == {FIRST, SECOND}
    assert len(run["cards"]) == 6
    assert all(c["meeting_interest"]["reference"] and c["meeting_interest"]["attribution"] for c in run["cards"])
    assert run["generation_id"] == dimension_case[1]["generation"]["run_id"]


def test_generator_only_sees_curated_descriptions_not_private_evidence(dimension_case):
    dimension_case[0]["interests"][0]["requirement"] += " PRIVATE_COMPANY 987654321"
    dimension_case[2]["dimensions"][0]["private_note"] = "PRIVATE_FRAGMENT"
    seen = []
    generate(dimension_case, generator=lambda ctx: seen.append(ctx) or copy.deepcopy(dimension_case[3]))
    payload = json.dumps(seen, ensure_ascii=False)
    assert FIRST in payload and SECOND in payload
    for forbidden in ("PRIVATE_COMPANY", "987654321", "PRIVATE_FRAGMENT", "chars:", "source-0", "sha256"):
        assert forbidden not in payload


@pytest.mark.parametrize("field", ["decision", "metrics_method"])
def test_reworded_or_generic_duplicate_does_not_become_a_distinct_dimension(dimension_case, field):
    cards = copy.deepcopy(dimension_case[3])
    cards[1][field] = cards[0][field]
    run = generate(dimension_case, cards=cards)
    assert cards[1]['meeting_interest']['dimension_id'] not in {c['meeting_interest']['dimension_id'] for c in run['cards']}
    assert len(run['cards']) == 5
    assert any('duplicate' in r.get('reason', '') for r in run['rejected_cards'])


@pytest.mark.parametrize("damage", ["unavailable", "wrong_route", "unknown_dimension", "stale_generation"])
def test_dimension_eligibility_and_generation_pin_fail_closed(dimension_case, damage):
    context, coverage, projection, cards = dimension_case
    if damage == "unavailable":
        next(r for r in coverage["routes"] if r["capability_id"] == cards[0]["capability_id"])["availability"] = "unavailable"
    elif damage == "wrong_route":
        cards[0]["capability_id"] = "AN-DOWNTIME"
    elif damage == "unknown_dimension":
        cards[0]["meeting_interest"]["dimension_id"] = "invented"
    else:
        coverage["generation"]["run_id"] = "stale"
        projection["generation_id"] = "current"
    if damage in ('unavailable', 'stale_generation'):
        with pytest.raises(ValueError):
            generate(dimension_case)
    else:
        run = generate(dimension_case)
        assert len(run['cards']) == 5
        assert all(c['meeting_interest']['dimension_id'] != cards[0]['meeting_interest']['dimension_id'] for c in run['cards'])
        assert run['rejected_cards']


@pytest.mark.parametrize("text", ["Ранее потрачено 987654321 рублей.", "Компания СЕКРЕТ уже завершила ремонт.",
                                  "Сценарий гарантированно снизит затраты."])
def test_unsupported_old_facts_and_scenario_effect_are_rejected(dimension_case, text):
    dimension_case[3][0]["analyst_brief"] = text
    run = generate(dimension_case)
    assert len(run['cards']) == 5
    assert all(c['analyst_brief'] != text for c in run['cards'])
    assert run['rejected_cards']


def test_scope_cannot_be_widened_and_verifier_must_be_independent(dimension_case):
    dimension_case[3][0]["scope"] = "всё оборудование"
    with pytest.raises(ValueError):
        generate(dimension_case, scope="насосы цеха A")
    component = lambda _: copy.deepcopy(dimension_case[3])
    with pytest.raises(ValueError):
        generate(dimension_case, generator=component, verifier=component)


@pytest.mark.parametrize("verdict", [{}, {"supported": True, "confidence": "low"},
                                     {"supported": False, "confidence": "high", "reason": "не подтверждено"}])
def test_semantic_verifier_cannot_leave_fewer_than_five_accepted(dimension_case, verdict):
    with pytest.raises((ValueError, RuntimeError)):
        generate(dimension_case, verifier=lambda _: copy.deepcopy(verdict))


def test_fewer_than_five_eligible_dimensions_stops_before_generator(dimension_case):
    dimension_case[2]["dimensions"] = dimension_case[2]["dimensions"][:4]
    calls = []
    with pytest.raises(ValueError):
        generate(dimension_case, generator=lambda ctx: calls.append(ctx) or dimension_case[3])
    assert calls == []


def test_reconciliation_requires_verdict_for_unconfirmed_reference(evidence_case):
    data = copy.deepcopy(evidence_case)
    row = data['v1']['dimensions'].pop(0)
    entry = row['evidence'][0]
    data['v1']['unconfirmed'].append({'interest_id': entry['interest_id'], 'dimension_id': row['dimension_id'],
                                      'topic_id': row['topic_id'], 'evidence': [entry]})
    data['v1_audit'] = [v for v in data['v1_audit'] if v['dimension_id'] != row['dimension_id']]
    with pytest.raises(ValueError):
        reconcile(data)


def test_verifier_sees_other_dimensions_to_detect_semantic_duplicates(dimension_case):
    seen = []
    def verifier(request):
        seen.append(request)
        return {'supported': True, 'confidence': 'high'}
    generate(dimension_case, verifier=verifier)
    assert all(len(req['other_directions']) == 5 for req in seen)
    assert all('decision' in d and 'metrics_method' in d and 'dimension_id' in d
               for req in seen for d in req['other_directions'])


def test_dimension_specific_missing_data_is_not_treated_as_available(dimension_case):
    run = generate(dimension_case)
    by_dim = {c['meeting_interest']['dimension_id']: c for c in run['cards']}
    assert any('полнот' in text.casefold() for text in by_dim[SECOND]['missing_evidence'])
    assert any('постав' in text.casefold() for text in by_dim['repair_material_cost_supply']['missing_evidence'])
    assert all(set(next(r for r in dimension_case[1]['routes'] if r['capability_id'] == card['capability_id'])['missing_evidence']) <= set(card['missing_evidence']) for card in run['cards'])


def test_verifier_receives_selected_route_and_dimension_data_limits(dimension_case):
    seen = []
    def verifier(request):
        seen.append(request)
        return {'supported': True, 'confidence': 'high'}
    generate(dimension_case, verifier=verifier)
    assert all(req['route']['capability_id'] and req['route']['availability'] == 'partial' for req in seen)
    assert all(req['route']['missing_evidence'] for req in seen)


def test_generated_missing_evidence_is_preserved_and_sent_to_verifier(dimension_case):
    dimension_case[3][0]['missing_evidence'].append('Проверить привязку операций к оборудованию')
    seen = []
    def verifier(request):
        seen.append(request)
        return {'supported': True, 'confidence': 'high'}
    run = generate(dimension_case, verifier=verifier)
    assert any('Проверить привязку операций к оборудованию' in c['missing_evidence'] for c in run['cards'])
    assert any('Проверить привязку операций к оборудованию' in r['missing_evidence'] for r in seen)


def test_validator_rejects_unsupported_numbers_in_modified_missing_evidence(dimension_case):
    run = generate(dimension_case)
    run['cards'][0]['missing_evidence'].append('Ранее потрачено 987654321 рублей')
    with pytest.raises(ValueError):
        pilot.validate_pilot(run, coverage=dimension_case[1], interest_context=dimension_case[0])
