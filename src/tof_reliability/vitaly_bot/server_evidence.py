"""Fixed read-only lookup sent to the TOF server over SSH.

This module is self-contained so the running bot does not depend on a synced
developer worktree.  Its caller supplies a bounded JSON request; table names,
paths and readable columns are fixed here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


EVIDENCE_VERSION = 1
MAX_CANDIDATES = 40
MAX_ROWS_PER_TABLE = 20
GENERATION_STATE = Path("/srv/tof/state/current_generation.json")
TOF_ROOT = Path("/srv/tof/repos/tof_v1")
SAFE_COLUMNS = (
    "event_id",
    "criticality_record_id",
    "source_file",
    "source_sheet",
    "source_row",
    "order_id",
    "equipment_id",
    "equipment_name",
    "inventory_number",
    "material_id",
    "operation_id",
    "fixed_asset_id",
    "work_type_raw",
    "work_type_norm",
    "actual_start",
    "actual_end",
    "planned_start",
    "planned_end",
    "event_date",
    "actual_cost",
    "planned_cost",
    "released_quantity",
    "unit_price",
    "duration_hours",
    "planned_work_hours",
    "actual_work_hours",
    "work_hours",
    "abc_code",
    "is_reliability_downtime",
)
MATCH_COLUMNS = (
    "event_id",
    "criticality_record_id",
    "order_id",
    "equipment_id",
    "inventory_number",
    "material_id",
    "operation_id",
    "fixed_asset_id",
)
CAPABILITY_TABLES = {
    "AN-REPAIR-EPISODES": ("work_order_events", "maintenance_fact_events", "work_order_operation_events"),
    "AN-PLAN-FACT": ("maintenance_plan_events", "maintenance_fact_events", "work_order_events"),
    "AN-MTR-COST": ("mtr_writeoff_events", "work_order_events", "material_requirements"),
    "AN-DOWNTIME": ("downtime_events", "sap_message_events", "work_order_events"),
    "AN-TECHCARD-LABOR": ("tech_card_operations", "work_order_operation_events", "material_requirements"),
    "AN-EQUIPMENT-CRITICALITY": ("equipment_criticality_records", "equipment_registry", "fixed_asset_records"),
}


def _generation(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("ledger_path") or not value.get("run_id"):
        raise ValueError("published_generation_incomplete")
    if not Path(str(value["ledger_path"])).is_dir():
        raise ValueError("published_ledger_missing")
    return value


def _source_statuses(ledger_dir: Path) -> Dict[str, str]:
    path = ledger_dir / "source_documents.csv"
    if not path.is_file():
        return {}
    statuses: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            document_id = str(row.get("source_document_id") or "")
            relative = str(row.get("relative_path") or "")
            status = str(row.get("parser_status") or "")
            for key in (document_id, relative, Path(relative).name if relative else ""):
                if key:
                    statuses[key] = status
    return statuses


def _available_capabilities(root: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads((root / "harness/capabilities/analytics.json").read_text(encoding="utf-8"))
    return {
        str(row["id"]): dict(row)
        for row in payload.get("capabilities", [])
        if isinstance(row, dict) and row.get("id") in CAPABILITY_TABLES and row.get("lifecycle") == "available"
    }


def _header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle), [])


def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def _matching_rows(ledger_dir: Path, candidates: Sequence[str]) -> List[tuple[str, Dict[str, str], List[str]]]:
    if not candidates:
        return []
    import duckdb

    selected_tables = list(dict.fromkeys(table for tables in CAPABILITY_TABLES.values() for table in tables))
    matches: List[tuple[str, Dict[str, str], List[str]]] = []
    connection = duckdb.connect(":memory:", config={"threads": "2", "memory_limit": "512MB"})
    try:
        placeholders = ", ".join("?" for _ in candidates)
        for table in selected_tables:
            path = ledger_dir / (table + ".csv")
            if not path.is_file():
                continue
            header = set(_header(path))
            match_columns = [column for column in MATCH_COLUMNS if column in header]
            columns = [column for column in SAFE_COLUMNS if column in header]
            if "source_document_id" in header:
                columns.append("source_document_id")
            if not match_columns or not columns:
                continue
            where = " OR ".join(f"{_quoted(column)} IN ({placeholders})" for column in match_columns)
            sort_columns = [
                column for column in (
                    "event_id", "criticality_record_id", "order_id", "equipment_id",
                    "material_id", "fixed_asset_id", "source_file", "source_row",
                ) if column in columns
            ]
            ordering = ", ".join(_quoted(column) for column in sort_columns)
            query = (
                "SELECT " + ", ".join(_quoted(column) for column in columns)
                + " FROM read_csv(?, all_varchar=true, header=true) WHERE "
                + where + (" ORDER BY " + ordering if ordering else "") + " LIMIT ?"
            )
            parameters: List[Any] = [str(path)]
            for _column in match_columns:
                parameters.extend(candidates)
            parameters.append(MAX_ROWS_PER_TABLE)
            result = connection.execute(query, parameters)
            names = [item[0] for item in result.description]
            for values in result.fetchall():
                record = {
                    name: str(value)
                    for name, value in zip(names, values)
                    if value is not None and str(value)
                }
                direct_source = record.pop("source_document_id", "")
                matches.append((table, record, [direct_source] if direct_source else []))
        missing_event_ids = list(dict.fromkeys(
            record.get("event_id", "") for _table, record, sources in matches
            if not sources and record.get("event_id")
        ))
        if missing_event_ids:
            provenance = _provenance(connection, ledger_dir, missing_event_ids)
            matches = [
                (table, record, sources or provenance.get(record.get("event_id", ""), []))
                for table, record, sources in matches
            ]
    finally:
        connection.close()
    return matches


def _provenance(connection: Any, ledger_dir: Path, event_ids: Sequence[str]) -> Dict[str, List[str]]:
    path = ledger_dir / "source_records.csv"
    if not path.is_file():
        return {}
    header = set(_header(path))
    if not {"event_id", "source_document_id"}.issubset(header):
        return {}
    placeholders = ", ".join("?" for _ in event_ids)
    rows = connection.execute(
        "SELECT event_id, source_document_id FROM read_csv(?, all_varchar=true, header=true) "
        + f"WHERE event_id IN ({placeholders})",
        [str(path), *event_ids],
    ).fetchall()
    result: Dict[str, List[str]] = {}
    for event_id, document_id in rows:
        if event_id and document_id:
            result.setdefault(str(event_id), []).append(str(document_id))
    return {key: list(dict.fromkeys(value)) for key, value in result.items()}


def build_evidence_pack(
    request: Dict[str, Any],
    *,
    generation_state_path: Path = GENERATION_STATE,
    root: Path = TOF_ROOT,
) -> Dict[str, Any]:
    """Return bounded facts only from the current published generation."""

    if request.get("version") != EVIDENCE_VERSION:
        raise ValueError("request_version")
    generation = _generation(generation_state_path)
    ledger_dir = Path(str(generation["ledger_path"]))
    candidates = list(dict.fromkeys(
        str(value).strip() for value in request.get("candidate_ids", []) if str(value).strip()
    ))[:MAX_CANDIDATES]
    source_status = _source_statuses(ledger_dir)
    capabilities = _available_capabilities(root)
    rows: List[Dict[str, Any]] = []
    for table, record, source_ids in _matching_rows(ledger_dir, candidates):
        source_file = record.get("source_file", "")
        provenance_statuses = [source_status.get(value, "unresolved") for value in source_ids]
        if "needs_review" in provenance_statuses:
            continue
        parser_status = (
            next((status for status in provenance_statuses if status != "unresolved"), "")
            or source_status.get(source_file)
            or source_status.get(Path(source_file).name)
            or "unresolved"
        )
        if parser_status == "needs_review":
            continue
        rows.append({
            "ledger_table": table,
            "record": record,
            "source_document_ids": source_ids,
            "provenance": "resolved" if source_ids else "unresolved",
            "parser_status": parser_status,
        })
    found_tables = {row["ledger_table"] for row in rows}
    selected_capabilities = []
    for capability_id, capability in capabilities.items():
        tables = CAPABILITY_TABLES[capability_id]
        if found_tables.intersection(tables):
            selected_capabilities.append({
                "capability_id": capability_id,
                "route_id": capability.get("route_id"),
                "tables": [table for table in tables if table in found_tables],
                "method": capability.get("behavior"),
            })
    return {
        "evidence_version": EVIDENCE_VERSION,
        "generation": {
            "run_id": generation["run_id"],
            "ledger_path": str(ledger_dir),
            "promoted_at": generation.get("promoted_at", ""),
        },
        "candidate_ids": candidates,
        "periods": [str(value) for value in request.get("periods", [])][:MAX_CANDIDATES],
        "material_claims": [str(value) for value in request.get("claims", [])][:80],
        "material_numbers": [str(value) for value in request.get("numbers", [])][:80],
        "capabilities": selected_capabilities,
        "records": rows,
        "limitations": [
            "В пакет попадают только точные совпадения идентификаторов.",
            "Строки needs_review исключены.",
            "Строка без source_document_id не является подтверждённым фактом.",
        ],
    }
