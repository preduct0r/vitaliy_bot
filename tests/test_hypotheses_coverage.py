"""Contract for S2's fixed, read-only coverage lookup.

The server entry point is ``server_coverage.build_coverage(coverage_request, ...)``;
its client is ``coverage.RemoteCoverageProvider.build(coverage_request)``.  The test
ledger below is synthetic and never touches the production server or database.
"""

import asyncio
import csv
import importlib
import json
from pathlib import Path

import pytest


ROUTES = {
    "AN-REPAIR-EPISODES", "AN-PLAN-FACT", "AN-MTR-COST", "AN-DOWNTIME",
    "AN-TECHCARD-LABOR", "AN-EQUIPMENT-CRITICALITY",
}


@pytest.fixture
def server():
    try:
        return importlib.import_module("tof_reliability.vitaly_bot.server_coverage")
    except ModuleNotFoundError as exc:
        if exc.name == "tof_reliability.vitaly_bot.server_coverage":
            class MissingServer:
                def __getattr__(self, _name):
                    pytest.fail("S2 server coverage operation is not implemented")
            return MissingServer()
        raise


@pytest.fixture
def client():
    try:
        return importlib.import_module("tof_reliability.vitaly_bot.coverage")
    except ModuleNotFoundError as exc:
        if exc.name == "tof_reliability.vitaly_bot.coverage":
            class MissingClient:
                def __getattr__(self, _name):
                    pytest.fail("S2 remote coverage client is not implemented")
            return MissingClient()
        raise


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def snapshot(tmp_path):
    ledger = tmp_path / "generations" / "g1" / "ledger"
    ledger.mkdir(parents=True)
    state = tmp_path / "current_generation.json"
    state.write_text(json.dumps({"run_id": "g1", "ledger_path": str(ledger)}), encoding="utf-8")
    write_csv(ledger / "source_documents.csv", ["source_document_id", "relative_path", "parser_status"], [
        {"source_document_id": "good", "relative_path": "good.csv", "parser_status": "published"},
        {"source_document_id": "review", "relative_path": "review.csv", "parser_status": "needs_review"},
    ])
    write_csv(ledger / "work_order_events.csv",
              ["event_id", "order_id", "equipment_id", "event_date", "source_document_id"], [
        {"event_id": "e1", "order_id": "o1", "equipment_id": "eq1", "event_date": "2026-01-10", "source_document_id": "good"},
        {"event_id": "e2", "order_id": "o2", "equipment_id": "eq1", "event_date": "2026-02-10", "source_document_id": "review"},
        {"event_id": "e3", "order_id": "o3", "equipment_id": "eq2", "event_date": "2026-01-11", "source_document_id": "good"},
    ])
    root = tmp_path / "server-root"
    capabilities = root / "harness" / "capabilities"
    capabilities.mkdir(parents=True)
    (capabilities / "analytics.json").write_text(json.dumps({"capabilities": [
        {"id": route, "route_id": route, "lifecycle": "available"} for route in sorted(ROUTES)
    ]}), encoding="utf-8")
    return {"ledger": ledger, "state": state, "root": root}


@pytest.fixture
def coverage_request():
    return {"version": 1, "scope": {"equipment_ids": ["eq1"]},
            "period": {"start": "2026-01-01", "end": "2026-02-28"}}


def build(server, snapshot, coverage_request):
    return server.build_coverage(coverage_request, generation_state_path=snapshot["state"], root=snapshot["root"])


def route(result, route_id):
    return next(item for item in result["routes"] if item["capability_id"] == route_id)


def test_fixed_routes_and_generation_even_without_case_files(server, snapshot, coverage_request):
    result = build(server, snapshot, coverage_request)
    assert result["generation"]["run_id"] == "g1"
    assert {item["capability_id"] for item in result["routes"]} == ROUTES
    assert all("availability" in item and "missing_evidence" in item for item in result["routes"])
    assert route(result, "AN-REPAIR-EPISODES")["availability"] != "available"


def test_empty_tables_and_unmatched_scope_are_not_proof(server, snapshot, coverage_request):
    coverage_request["scope"] = {"equipment_ids": ["missing"]}
    result = build(server, snapshot, coverage_request)
    assert all(item["availability"] != "available" for item in result["routes"])
    assert route(result, "AN-REPAIR-EPISODES")["missing_evidence"]


def test_needs_review_and_other_equipment_do_not_fill_period_gap(server, snapshot, coverage_request):
    result = build(server, snapshot, coverage_request)
    repairs = route(result, "AN-REPAIR-EPISODES")
    assert repairs["availability"] != "available"
    assert repairs["missing_evidence"]
    # Neither the reviewed February row nor the out-of-scope row may be cited.
    assert "e2" not in json.dumps(repairs, ensure_ascii=False)
    assert "e3" not in json.dumps(repairs, ensure_ascii=False)


def test_unresolved_source_is_not_confirmed_coverage(server, snapshot, coverage_request):
    write_csv(snapshot["ledger"] / "work_order_events.csv",
              ["event_id", "order_id", "equipment_id", "event_date", "source_document_id"], [
        {"event_id": "unknown", "order_id": "o1", "equipment_id": "eq1", "event_date": "2026-01-10", "source_document_id": "missing"},
    ])
    result = build(server, snapshot, coverage_request)
    repairs = route(result, "AN-REPAIR-EPISODES")
    assert repairs["availability"] != "available"
    assert repairs["missing_evidence"]


def test_plan_and_fact_without_matching_keys_do_not_prove_join(server, snapshot, coverage_request):
    columns = ["event_id", "order_id", "equipment_id", "event_date", "source_document_id"]
    for table, order in [("maintenance_plan_events", "plan-only"), ("maintenance_fact_events", "fact-only")]:
        write_csv(snapshot["ledger"] / (table + ".csv"), columns, [
            {"event_id": table, "order_id": order, "equipment_id": "eq1", "event_date": "2026-01-10", "source_document_id": "good"},
        ])
    item = route(build(server, snapshot, coverage_request), "AN-PLAN-FACT")
    assert item["availability"] != "available"
    assert item["missing_evidence"]


def test_missing_or_invalid_current_generation_fails_without_old_fallback(server, snapshot, coverage_request):
    snapshot["state"].unlink()
    with pytest.raises((ValueError, OSError), match="(?i)generation|current|no such file"):
        build(server, snapshot, coverage_request)
    snapshot["state"].write_text('{"run_id":"g1"}', encoding="utf-8")
    with pytest.raises((ValueError, OSError), match="(?i)generation|ledger"):
        build(server, snapshot, coverage_request)


def test_snapshot_files_are_read_only(server, snapshot, coverage_request):
    paths = [snapshot["state"], *snapshot["ledger"].rglob("*")]
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths if p.is_file()}
    build(server, snapshot, coverage_request)
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths if p.is_file()}
    assert after == before
    assert set(snapshot["ledger"].rglob("*")) == set(paths[1:])


def test_generation_drift_retries_once_then_uses_stable_generation(server, snapshot, coverage_request, monkeypatch):
    original = server._generation
    calls = []
    def generation(path):
        value = original(path)
        calls.append(None)
        return {**value, "run_id": "g1" if len(calls) == 1 else "g2"}
    monkeypatch.setattr(server, "_generation", generation)
    result = build(server, snapshot, coverage_request)
    assert result["generation"]["run_id"] == "g2"
    assert len(calls) == 4


def test_repeated_generation_drift_fails_after_one_retry(server, snapshot, coverage_request, monkeypatch):
    original = server._generation
    calls = []
    def generation(path):
        value = original(path)
        calls.append(None)
        return {**value, "run_id": f"g{len(calls)}"}
    monkeypatch.setattr(server, "_generation", generation)
    with pytest.raises((ValueError, RuntimeError), match="(?i)generation.*(drift|chang)|(?:drift|chang).*generation"):
        build(server, snapshot, coverage_request)
    assert len(calls) == 4


def test_rejects_untrusted_query_or_snapshot_override(server, snapshot, coverage_request):
    for extra in ({"sql": "DROP TABLE work_order_events"}, {"ledger_path": "/tmp/old-ledger"}, {"route": "../../secret"}):
        with pytest.raises(ValueError):
            build(server, snapshot, {**coverage_request, **extra})


def test_remote_outage_is_explicit_and_never_falls_back(client, coverage_request, monkeypatch):
    async def unavailable(*args, **kwargs):
        raise OSError("synthetic SSH outage")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", unavailable)
    provider = client.RemoteCoverageProvider()
    with pytest.raises(Exception, match="(?i)unavailable|недоступ|ssh|outage"):
        asyncio.run(provider.build(coverage_request))
