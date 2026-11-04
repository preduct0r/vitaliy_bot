"""Synthetic contract based on the published TOF catalog and ledger schemas.

No production data is copied into these fixtures. Catalog status is distinct
from the source parser classification; neither proves period completeness.
"""
import csv
import json
from pathlib import Path

import pytest

from tof_reliability.vitaly_bot import server_coverage as server


COLUMNS = ["source_document_id", "order_id", "equipment_id", "event_date"]


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def source(identifier, parser="supported"):
    return {"source_document_id": identifier, "parser_status": parser}


def catalog(identifier, status="ready_for_parse", parser="supported"):
    return {"source_document_id": identifier, "catalog_status": status, "parser_status": parser}


def event(identifier, order="o1", equipment="eq1", day="2026-01-10"):
    return {"source_document_id": identifier, "order_id": order,
            "equipment_id": equipment, "event_date": day}


@pytest.fixture
def case(tmp_path):
    ledger = tmp_path / "ledger" / "g1"
    ledger.mkdir(parents=True)
    cat = tmp_path / "inventory" / "g1" / "data_catalog.csv"
    state = tmp_path / "current_generation.json"
    state.write_text(json.dumps({"run_id": "g1", "ledger_path": str(ledger),
                                 "catalog_path": str(cat)}), encoding="utf-8")
    root = tmp_path / "root"
    capability = root / "harness" / "capabilities" / "analytics.json"
    capability.parent.mkdir(parents=True)
    capability.write_text(json.dumps({"capabilities": [
        {"id": name, "route_id": name, "lifecycle": "available"}
        for name in server.ROUTES
    ]}), encoding="utf-8")
    write_csv(ledger / "source_documents.csv", ["source_document_id", "parser_status"], [source("ok")])
    write_csv(cat, ["source_document_id", "catalog_status", "parser_status"], [catalog("ok")])
    return {"ledger": ledger, "catalog": cat, "state": state, "root": root}


def build(case, scope=None, period=None):
    request = {"version": 1, "scope": {"equipment_ids": scope or ["eq1"]},
               "period": period or {"start": "2026-01-01", "end": "2026-02-28"}}
    return server.build_coverage(request, generation_state_path=case["state"], root=case["root"])


def route(result, name="AN-REPAIR-EPISODES"):
    return next(row for row in result["routes"] if row["capability_id"] == name)


def set_sources(case, sources, catalogs):
    write_csv(case["ledger"] / "source_documents.csv", ["source_document_id", "parser_status"], sources)
    write_csv(case["catalog"], ["source_document_id", "catalog_status", "parser_status"], catalogs)


def set_table(case, table, rows):
    write_csv(case["ledger"] / (table + ".csv"), COLUMNS, rows)


@pytest.mark.parametrize("status", ["ready_for_parse", "auto_parsed"])
def test_supported_with_published_catalog_and_actual_fact_is_partial(case, status):
    set_sources(case, [source("ok")], [catalog("ok", status)])
    set_table(case, "work_order_events", [event("ok")])
    result = build(case)
    item = route(result)
    assert result["generation"]["run_id"] == "g1"
    assert item["availability"] == "partial"
    assert item["tables"]["work_order_events"]["count"] == 1
    assert item["tables"]["work_order_events"]["months"] == ["2026-01"]
    assert item["missing_evidence"]  # presence is not completeness


@pytest.mark.parametrize("catalog_rows,source_rows", [
    ([catalog("ok", "needs_review")], [source("ok")]),
    ([], [source("ok")]),
    ([catalog("ok", "ready_for_parse"), catalog("ok", "needs_review")], [source("ok")]),
    ([catalog("ok", "ready_for_parse")], [source("ok"), source("ok", "skip")]),
    ([catalog("ok", "ready_for_parse", "unknown")], [source("ok")]),
    ([catalog("ok", "ready_for_parse")], []),
])
def test_unresolved_conflicting_or_review_provenance_never_counts(case, catalog_rows, source_rows):
    set_sources(case, source_rows, catalog_rows)
    set_table(case, "work_order_events", [event("ok")])
    item = route(build(case))
    assert item["availability"] == "unavailable"
    assert "work_order_events" not in item["tables"]


@pytest.mark.parametrize("parser,status", [
    ("skip", "registered_skipped"), ("unknown", "auto_parse_queued"),
    ("inventory_only", "registered_reference"), ("supported", "auto_parse_failed"),
])
def test_catalog_inventory_and_queue_are_not_published_facts(case, parser, status):
    set_sources(case, [source("ok", parser)], [catalog("ok", status, parser)])
    set_table(case, "work_order_events", [event("ok")])
    assert route(build(case))["availability"] == "unavailable"


def test_scope_period_and_join_use_only_admissible_rows(case):
    set_sources(case, [source("ok"), source("review")],
                [catalog("ok"), catalog("review", "needs_review")])
    set_table(case, "maintenance_plan_events", [event("ok", "match"), event("ok", "other", "eq2")])
    set_table(case, "maintenance_fact_events", [event("review", "match"),
              event("ok", "outside", day="2025-12-31"), event("ok", "other", "eq2")])
    item = route(build(case), "AN-PLAN-FACT")
    assert item["tables"]["maintenance_plan_events"]["count"] == 1
    assert "maintenance_fact_events" not in item["tables"]
    assert any("ключ" in text.lower() for text in item["missing_evidence"])
    set_table(case, "maintenance_fact_events", [event("ok", "match", day="2026-02-01")])
    item = route(build(case), "AN-PLAN-FACT")
    assert item["tables"]["maintenance_fact_events"]["months"] == ["2026-02"]
    assert not any("совпадения ключей" in text.lower() for text in item["missing_evidence"])
    assert item["availability"] == "partial"


def test_catalog_is_part_of_pinned_generation_and_drift_retries(case, monkeypatch, tmp_path):
    set_table(case, "work_order_events", [event("ok")])
    alternate = tmp_path / "other" / "data_catalog.csv"
    write_csv(alternate, ["source_document_id", "catalog_status", "parser_status"],
              [catalog("ok", "needs_review")])
    original = server._generation
    calls = []
    def changed(path):
        value = original(path)
        calls.append(None)
        return {**value, "catalog_path": str(case["catalog"] if len(calls) == 1 else alternate)}
    monkeypatch.setattr(server, "_generation", changed)
    result = build(case)
    assert len(calls) == 4
    assert route(result)["availability"] == "unavailable"


def test_missing_catalog_does_not_fall_back_to_parser_status(case):
    case["catalog"].unlink()
    set_table(case, "work_order_events", [event("ok")])
    try:
        result = build(case)
    except (ValueError, OSError, RuntimeError):
        return
    assert route(result)["availability"] == "unavailable"


def test_response_is_bounded_for_many_distinct_keys_and_sources(case):
    # Cardinality is deliberately larger than any reasonable evidence sample.
    n = 6000
    set_sources(case, [source(f"s{i}") for i in range(n)],
                [catalog(f"s{i}") for i in range(n)])
    set_table(case, "work_order_events", [event(f"s{i}", f"order{i}") for i in range(n)])
    result = build(case)
    item = route(result)
    assert item["tables"]["work_order_events"]["count"] == n
    assert len(json.dumps(result, ensure_ascii=False).encode()) < 50000
    assert "order5999" not in json.dumps(result)


@pytest.mark.parametrize("parser", ["supported", "inventory_only", "unknown"])
def test_auto_parsed_catalog_can_publish_facts_for_original_classification(case, parser):
    set_sources(case, [source("ok", parser)], [catalog("ok", "auto_parsed", parser)])
    set_table(case, "work_order_events", [event("ok")])
    assert route(build(case))["tables"]["work_order_events"]["count"] == 1


def test_generation_fingerprint_change_retries_even_when_paths_are_unchanged(case, monkeypatch):
    set_table(case, "work_order_events", [event("ok")])
    original = server._generation
    calls = []
    def changed(path):
        value = original(path)
        calls.append(None)
        return {**value, "catalog_sha256": "old" if len(calls) == 1 else "new"}
    monkeypatch.setattr(server, "_generation", changed)
    assert build(case)["generation"]["catalog_sha256"] == "new"
    assert len(calls) == 4


def test_undated_rows_do_not_emit_null_months(case):
    set_table(case, "work_order_events", [event("ok", day="")])
    result = server.build_coverage({"version": 1}, generation_state_path=case["state"], root=case["root"])
    assert route(result)["tables"]["work_order_events"]["months"] == []
