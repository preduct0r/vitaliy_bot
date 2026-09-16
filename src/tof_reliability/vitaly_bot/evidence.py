"""Read-only evidence packs for the Vitaly simulator.

The Telegram process never reads a local ledger.  It asks its personal server
workspace to run ``scripts/vitaly_ledger_evidence.py`` against the current
published generation and receives a deliberately small JSON evidence pack.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence

from .models import BotConfig, EvidenceError


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_VERSION = 1
MAX_CANDIDATES = 40
MAX_ROWS_PER_TABLE = 20
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


class EvidenceProvider(Protocol):
    async def build(self, material_text: str, artifact_hashes: Sequence[str]) -> Dict[str, Any]:
        ...


def extract_hints(material_text: str) -> Dict[str, List[str]]:
    """Extract a bounded material map; only exact identifiers become queries."""

    values: List[str] = []
    patterns = (
        r"(?i)(?:заказ|order)\s*[№#:]?\s*([A-ZА-Я0-9][A-ZА-Я0-9_-]{3,})",
        r"(?i)(?:ео|equipment(?:[_ ]id)?|оборудовани[ея])\s*[№#:]?\s*([A-ZА-Я0-9][A-ZА-Я0-9_-]{2,})",
        r"(?i)(?:мтр|material(?:[_ ]id)?)\s*[№#:]?\s*([A-ZА-Я0-9][A-ZА-Я0-9_-]{2,})",
        r"(?<!\d)(\d{6,12})(?!\d)",
    )
    for pattern in patterns:
        values.extend(match.group(1).strip() for match in re.finditer(pattern, material_text))
    candidate_ids = list(dict.fromkeys(value for value in values if value))[:MAX_CANDIDATES]
    periods = list(
        dict.fromkeys(re.findall(r"\b20\d{2}(?:[-./](?:0[1-9]|1[0-2]))?\b", material_text))
    )[:MAX_CANDIDATES]
    numbers = list(
        dict.fromkeys(re.findall(r"(?<!\w)\d{1,3}(?:[ \u00a0]\d{3})*(?:[.,]\d+)?\s*(?:%|₽|руб\.?|ч(?:ас(?:а|ов)?)?)?", material_text))
    )[:80]
    claims = [" ".join(value.split()) for value in re.split(r"[\n.!?]+", material_text) if len(value.strip()) >= 12]
    return {
        "candidate_ids": candidate_ids,
        "periods": periods,
        "numbers": numbers,
        "claims": list(dict.fromkeys(claims))[:80],
    }


class RemoteLedgerEvidenceProvider:
    """Run the fixed evidence helper through the personal server workspace."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config

    async def build(self, material_text: str, artifact_hashes: Sequence[str]) -> Dict[str, Any]:
        hints = extract_hints(material_text)
        request = {
            "version": EVIDENCE_VERSION,
            "artifact_hashes": list(artifact_hashes),
            **hints,
        }
        encoded = base64.urlsafe_b64encode(json.dumps(request, ensure_ascii=False).encode("utf-8")).decode("ascii")
        external_root = os.environ.get("VITALY_BOT_TOF_ROOT")
        client_root = Path(external_root).expanduser().resolve() if external_root else ROOT
        # The TOF client has its own dependencies and an identically named
        # `scripts` package. Never import it with the standalone bot interpreter.
        client_python = str(client_root / ".venv/bin/python") if external_root else sys.executable
        command = [
            client_python,
            str(client_root / "scripts" / "tof_remote_workspace.py"),
            "run",
            "tof_v1",
            "--",
            ".venv/bin/python",
            "scripts/vitaly_ledger_evidence.py",
            "--request-b64",
            encoded,
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(client_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=self.config.evidence_timeout_seconds
            )
        except asyncio.TimeoutError as error:
            raise EvidenceError("tof_unavailable", "timeout") from error
        except OSError as error:
            raise EvidenceError("tof_unavailable", str(error)) from error
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise EvidenceError("tof_unavailable", (stderr + "\n" + stdout)[-4000:])
        for line in reversed(stdout.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("evidence_version") == EVIDENCE_VERSION:
                return value
        raise EvidenceError("tof_evidence_invalid", (stderr + "\n" + stdout)[-4000:])


def _generation(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError("tof_unavailable", str(error)) from error
    if not isinstance(value, dict) or not value.get("ledger_path") or not value.get("run_id"):
        raise EvidenceError("tof_unavailable", "published_generation_incomplete")
    return value


def _source_statuses(ledger_dir: Path) -> Dict[str, str]:
    from tof_reliability.data import load_table

    documents = load_table(
        "source_documents",
        ledger_dir=ledger_dir,
        columns=("source_document_id", "relative_path", "parser_status"),
        missing_ok=True,
    )
    result: Dict[str, str] = {}
    for row in documents.to_dict("records"):
        document_id = str(row.get("source_document_id") or "")
        relative = str(row.get("relative_path") or "")
        status = str(row.get("parser_status") or "")
        if document_id:
            result[document_id] = status
        if relative:
            result[relative] = status
            result[Path(relative).name] = status
    return result


def _provenance(ledger_dir: Path, event_ids: Iterable[str]) -> Dict[str, List[str]]:
    from tof_reliability.data import load_table

    wanted = {value for value in event_ids if value}
    if not wanted:
        return {}
    records = load_table(
        "source_records",
        ledger_dir=ledger_dir,
        columns=("event_id", "source_document_id"),
        missing_ok=True,
    )
    result: Dict[str, List[str]] = {}
    for row in records.to_dict("records"):
        event_id = str(row.get("event_id") or "")
        document_id = str(row.get("source_document_id") or "")
        if event_id in wanted and document_id:
            result.setdefault(event_id, []).append(document_id)
    return {key: list(dict.fromkeys(value)) for key, value in result.items()}


def _available_capabilities(root: Path) -> Dict[str, Dict[str, Any]]:
    path = root / "harness" / "capabilities" / "analytics.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError("tof_evidence_invalid", str(error)) from error
    return {
        str(row["id"]): dict(row)
        for row in payload.get("capabilities", [])
        if isinstance(row, dict) and row.get("id") in CAPABILITY_TABLES and row.get("lifecycle") == "available"
    }


def build_evidence_pack(
    request: Dict[str, Any],
    *,
    generation_state_path: Path = Path("/srv/tof/state/current_generation.json"),
    root: Path = ROOT,
) -> Dict[str, Any]:
    """Return only exact row excerpts with current-generation provenance.

    This is intentionally a lookup helper, not a generic analytics engine.  A
    claim without an exact identifier is returned as unconfirmed rather than
    being converted into an unsafe aggregate.
    """

    if request.get("version") != EVIDENCE_VERSION:
        raise EvidenceError("tof_evidence_invalid", "request_version")
    generation = _generation(generation_state_path)
    ledger_dir = Path(str(generation["ledger_path"]))
    candidates = [str(value).strip() for value in request.get("candidate_ids", []) if str(value).strip()]
    candidates = list(dict.fromkeys(candidates))[:MAX_CANDIDATES]
    candidate_set = set(candidates)
    source_status = _source_statuses(ledger_dir)
    capabilities = _available_capabilities(root)
    rows: List[Dict[str, Any]] = []
    selected_capabilities: List[Dict[str, Any]] = []
    selected_tables = list(dict.fromkeys(table for tables in CAPABILITY_TABLES.values() for table in tables))
    raw_rows: List[tuple[str, Dict[str, Any]]] = []
    from tof_reliability.data import load_table

    for table in (selected_tables if candidate_set else ()):
        frame = load_table(table, ledger_dir=ledger_dir, columns=SAFE_COLUMNS, missing_ok=True)
        if not candidate_set or frame.empty:
            continue
        mask = frame[list(MATCH_COLUMNS)].astype(str).apply(lambda column: column.isin(candidate_set)).any(axis=1)
        for record in frame[mask].head(MAX_ROWS_PER_TABLE).to_dict("records"):
            raw_rows.append((table, {key: str(value) for key, value in record.items() if str(value)}))
    provenance = _provenance(ledger_dir, (row.get("event_id", "") for _table, row in raw_rows))
    for table, record in raw_rows:
        event_id = record.get("event_id", "")
        source_ids = provenance.get(event_id, [])
        source_file = record.get("source_file", "")
        provenance_statuses = [source_status.get(value, "unresolved") for value in source_ids]
        parser_status = (
            next((value for value in provenance_statuses if value), "")
            or source_status.get(source_file)
            or source_status.get(Path(source_file).name)
            or "unresolved"
        )
        if parser_status == "needs_review" or "needs_review" in provenance_statuses:
            continue
        rows.append(
            {
                "ledger_table": table,
                "record": record,
                "source_document_ids": source_ids,
                "provenance": "resolved" if source_ids else "unresolved",
                "parser_status": parser_status,
            }
        )
    found_tables = {row["ledger_table"] for row in rows}
    for capability_id, capability in capabilities.items():
        tables = CAPABILITY_TABLES[capability_id]
        if found_tables.intersection(tables):
            selected_capabilities.append(
                {
                    "capability_id": capability_id,
                    "route_id": capability.get("route_id"),
                    "tables": [table for table in tables if table in found_tables],
                    "method": capability.get("behavior"),
                }
            )
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
