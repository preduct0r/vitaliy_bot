"""Доступ к ledger: один загрузчик на таблицу вместо `pd.read_csv` с угадыванием колонок.

Схемы таблиц — `schemas.py`, чтение — `loaders.py`, контракты таблиц —
`docs/contracts/`, карта данных — `DATA_MAP.md`. Назначение, границы и порядок
добавления таблицы — в `README.md` рядом.
"""
from __future__ import annotations

from tof_reliability.data.loaders import (
    LedgerSchemaError,
    LedgerTableNotFoundError,
    load_downtime_events,
    load_equipment_criticality,
    load_maintenance_plan_events,
    load_mtr_writeoffs,
    load_work_order_operations,
    load_table,
    load_tech_card_headers,
    load_tech_card_operations,
    load_work_orders,
    resolve_ledger_dir,
    summarize_table,
    table_path,
)
from tof_reliability.data.schemas import LEDGER_TABLES, LedgerTableSchema, table_schema

__all__ = [
    "LEDGER_TABLES",
    "LedgerSchemaError",
    "LedgerTableNotFoundError",
    "LedgerTableSchema",
    "load_downtime_events",
    "load_equipment_criticality",
    "load_maintenance_plan_events",
    "load_mtr_writeoffs",
    "load_work_order_operations",
    "load_table",
    "load_tech_card_headers",
    "load_tech_card_operations",
    "load_work_orders",
    "resolve_ledger_dir",
    "summarize_table",
    "table_path",
    "table_schema",
]
