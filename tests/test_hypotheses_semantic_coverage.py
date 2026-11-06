"""Regression boundaries for the published semantic coverage contract.

The event rows below are deliberately misleading: repeated order rows are not
repair episodes and matching raw order IDs are not a plan/fact methodology.
They contain no published semantic derivations and cannot prove coverage.
"""
import csv
import json

import pytest

from tof_reliability.vitaly_bot import server_coverage as server


ROUTES = {
    "AN-REPAIR-EPISODES", "AN-PLAN-FACT", "AN-MTR-COST", "AN-DOWNTIME",
    "AN-TECHCARD-LABOR", "AN-EQUIPMENT-CRITICALITY",
}


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def published(tmp_path, monkeypatch):
    from tof_reliability.vitaly_bot import server_coverage
    empty = lambda generation, ids: {"generation": generation["run_id"], "columns": ["source_document_id"], "rows": [], "truncated": False}
    monkeypatch.setattr(server_coverage, "_criticality_query", empty)
    monkeypatch.setattr(server_coverage, "_techcard_query", empty)
    ledger = tmp_path / "ledger" / "g1"
    ledger.mkdir(parents=True)
    catalog = tmp_path / "inventory" / "g1" / "data_catalog.csv"
    write_csv(catalog, ["source_document_id", "catalog_status", "parser_status"], [
        {"source_document_id": "accepted", "catalog_status": "ready_for_parse", "parser_status": "supported"},
        {"source_document_id": "review-only", "catalog_status": "needs_review", "parser_status": "supported"},
    ])
    write_csv(ledger / "source_documents.csv", ["source_document_id", "parser_status"], [
        {"source_document_id": "accepted", "parser_status": "supported"},
        {"source_document_id": "review-only", "parser_status": "supported"},
    ])
    columns = ["source_document_id", "order_id", "equipment_id", "event_date"]
    rows = [
        {"source_document_id": "accepted", "order_id": "same-order", "equipment_id": "eq1", "event_date": "2026-01-10"},
        {"source_document_id": "accepted", "order_id": "same-order", "equipment_id": "eq1", "event_date": "2026-01-10"},
        {"source_document_id": "review-only", "order_id": "review-order", "equipment_id": "eq1", "event_date": "2026-02-10"},
    ]
    for table in ("work_order_events", "maintenance_plan_events", "maintenance_fact_events"):
        write_csv(ledger / (table + ".csv"), columns, rows)
    state = tmp_path / "current.json"
    state.write_text(json.dumps({"run_id": "g1", "ledger_path": str(ledger),
                                 "catalog_path": str(catalog)}), encoding="utf-8")
    root = tmp_path / "root"
    capabilities = root / "harness" / "capabilities" / "analytics.json"
    capabilities.parent.mkdir(parents=True)
    capabilities.write_text(json.dumps({"capabilities": [
        {"id": name, "route_id": name, "lifecycle": "available"} for name in ROUTES
    ] + [{"id": "AN-SCENARIO-EFFECT", "route_id": "scenario_effect", "lifecycle": "planned"}]}), encoding="utf-8")
    return state, root


def build(published):
    state, root = published
    return server.build_coverage(
        {"version": 1, "scope": {"equipment_ids": ["eq1"]},
         "period": {"start": "2026-01-01", "end": "2026-02-28"}},
        generation_state_path=state, root=root,
    )


def route(result, capability):
    return next(item for item in result["routes"] if item["capability_id"] == capability)


def test_raw_event_rows_are_not_published_semantic_counts_or_equipment_links(published):
    item = route(build(published), "AN-REPAIR-EPISODES")
    # Two identical physical records cannot establish either two orders or a
    # repair episode. No derived_repair_episodes fixture is published here.
    assert item["availability"] == "unavailable"
    assert "work_order_events" not in item.get("tables", {})
    assert item["missing_evidence"]


def test_matching_raw_order_ids_do_not_prove_plan_fact_join(published):
    item = route(build(published), "AN-PLAN-FACT")
    assert item["availability"] == "unavailable"
    assert item["missing_evidence"]
    assert any(any(word in text.lower() for word in ("метод", "сопостав", "ключ"))
               for text in item["missing_evidence"])


def test_six_routes_explain_unavailable_methodology_and_scenario_is_not_available(published):
    result = build(published)
    by_id = {item["capability_id"]: item for item in result["routes"]}
    assert ROUTES <= set(by_id)
    assert all(by_id[name]["missing_evidence"] for name in ROUTES)
    assert all(by_id[name]["availability"] == "unavailable" for name in ROUTES)
    assert by_id.get("AN-SCENARIO-EFFECT", {}).get("availability") != "available"
    assert result["coverage_version"] == 1
    assert result["generation"]["run_id"] == "g1"


def test_review_source_and_unverified_period_are_not_cited(published):
    result = build(published)
    encoded = json.dumps(result["routes"], ensure_ascii=False)
    assert "review-only" not in encoded
    assert "review-order" not in encoded
    assert all(item["availability"] != "available" for item in result["routes"])
    assert all(item["missing_evidence"] for item in result["routes"])


def test_semantic_criticality_uses_generation_pinned_query_and_preserves_provenance(published, monkeypatch):
    calls = []
    def query(generation, ids):
        calls.append((generation['run_id'], ids))
        return {'generation': generation['run_id'], 'columns': ['source_document_id'],
                'rows': [['accepted']], 'truncated': False}
    monkeypatch.setattr(server, '_criticality_query', query)
    result = build(published)
    item = route(result, 'AN-EQUIPMENT-CRITICALITY')
    assert calls == [('g1', {'eq1'})]
    assert item['availability'] == 'partial'
    assert item['tables']['equipment']['source_document_ids'] == ['accepted']
    assert item['missing_evidence']


def test_semantic_query_generation_mismatch_is_not_accepted(published, monkeypatch):
    monkeypatch.setattr(server, '_criticality_query', lambda generation, ids: {
        'generation': 'other', 'columns': ['source_document_id'],
        'rows': [['accepted']], 'truncated': False})
    with pytest.raises((ValueError, RuntimeError), match='generation'):
        build(published)


def test_semantic_techcard_sources_exclude_review_and_keep_limitations(published, monkeypatch):
    monkeypatch.setattr(server, '_criticality_query', lambda generation, ids: {'generation': generation['run_id'], 'columns': ['source_document_id'], 'rows': [], 'truncated': False})
    monkeypatch.setattr(server, '_techcard_query', lambda generation, ids: {'generation': generation['run_id'], 'columns': ['source_document_id'], 'rows': [['accepted'], ['review-only']], 'truncated': False})
    item = route(build(published), 'AN-TECHCARD-LABOR')
    assert item['availability'] == 'partial'
    assert item['tables']['tech_card']['source_document_ids'] == ['accepted']
    assert item['missing_evidence']
    assert 'review-only' not in json.dumps(item)


def test_truncated_semantic_sources_are_not_claimed(published, monkeypatch):
    monkeypatch.setattr(server, '_criticality_query', lambda generation, ids: {'generation': generation['run_id'], 'columns': ['source_document_id'], 'rows': [['accepted']], 'truncated': True})
    monkeypatch.setattr(server, '_techcard_query', lambda generation, ids: {'generation': generation['run_id'], 'columns': ['source_document_id'], 'rows': [], 'truncated': False})
    item = route(build(published), 'AN-EQUIPMENT-CRITICALITY')
    assert item['availability'] == 'unavailable'
    assert item['missing_evidence']
