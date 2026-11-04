"""Fixed, read-only coverage observations for the published TOF generation.

Event presence is not evidence of complete observation over a period.  Until a
published completeness contract exists, routes remain partial or unavailable.
This file is self-contained so it can be sent to the server over SSH.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

GENERATION_STATE = Path('/srv/tof/state/current_generation.json')
TOF_ROOT = Path('/srv/tof/repos/tof_v1')
ROUTES = {
    'AN-REPAIR-EPISODES': ('work_order_events', 'maintenance_fact_events', 'work_order_operation_events'),
    'AN-PLAN-FACT': ('maintenance_plan_events', 'maintenance_fact_events', 'work_order_events'),
    'AN-MTR-COST': ('mtr_writeoff_events', 'work_order_events', 'material_requirements'),
    'AN-DOWNTIME': ('downtime_events', 'sap_message_events', 'work_order_events'),
    'AN-TECHCARD-LABOR': ('tech_card_operations', 'work_order_operation_events', 'material_requirements'),
    'AN-EQUIPMENT-CRITICALITY': ('equipment_criticality_records', 'equipment_registry', 'fixed_asset_records'),
}
DATE_COLUMNS = ('event_date', 'actual_start', 'planned_start')


def _generation(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or not isinstance(value.get('run_id'), str) or not value['run_id'] or not isinstance(value.get('ledger_path'), str) or not value['ledger_path'] or not isinstance(value.get('catalog_path'), str) or not value['catalog_path']:
        raise ValueError('published_generation_incomplete')
    if not Path(value['ledger_path']).is_dir():
        raise ValueError('published_generation_ledger_missing')
    if not Path(value['catalog_path']).is_file():
        raise ValueError('published_generation_catalog_missing')
    return value


def _request(request: dict[str, Any]) -> tuple[set[str], date | None, date | None]:
    if not isinstance(request, dict) or set(request) - {'version', 'scope', 'period'} or request.get('version') != 1 or isinstance(request.get('version'), bool):
        raise ValueError('invalid_coverage_request')
    scope = request.get('scope', {})
    period = request.get('period', {})
    if not isinstance(scope, dict) or set(scope) - {'equipment_ids'} or not isinstance(period, dict) or set(period) - {'start', 'end'}:
        raise ValueError('invalid_coverage_scope_or_period')
    ids = scope.get('equipment_ids', [])
    if not isinstance(ids, list) or len(ids) > 100 or any(not isinstance(x, str) or not x.strip() or len(x) > 200 for x in ids):
        raise ValueError('invalid_equipment_ids')
    try:
        start = date.fromisoformat(period['start']) if period.get('start') else None
        end = date.fromisoformat(period['end']) if period.get('end') else None
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid_coverage_period') from exc
    if start and end and start > end:
        raise ValueError('invalid_coverage_period')
    return set(ids), start, end


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _collect(generation: dict[str, Any], root: Path, ids: set[str], start: date | None, end: date | None) -> list[dict[str, Any]]:
    ledger = Path(generation['ledger_path'])
    capabilities = json.loads((root / 'harness/capabilities/analytics.json').read_text(encoding='utf-8'))
    active = {x.get('id'): x for x in capabilities.get('capabilities', []) if isinstance(x, dict) and x.get('lifecycle') == 'available'}
    connection = duckdb.connect(':memory:', config={'threads': '2', 'memory_limit': '2GB', 'temp_directory': ''})
    try:
        source_path = ledger / 'source_documents.csv'
        if not source_path.is_file():
            raise ValueError('published_generation_sources_missing')
        source = _literal(str(source_path))
        catalog = _literal(generation['catalog_path'])
        # Both inventories must agree, including duplicate records. Auto-parsed
        # sources may originally have been inventory_only or unknown.
        connection.execute(f"""
            CREATE TEMP VIEW eligible AS
            WITH s AS (
                SELECT source_document_id, min(coalesce(parser_status, '')) AS parser_kind
                FROM read_csv({source}, all_varchar=true)
                WHERE source_document_id IS NOT NULL AND source_document_id <> ''
                GROUP BY source_document_id HAVING count(DISTINCT coalesce(parser_status, '')) = 1
            ), c AS (
                SELECT source_document_id, min(coalesce(parser_status, '')) AS parser_kind,
                       min(coalesce(catalog_status, '')) status
                FROM read_csv({catalog}, all_varchar=true)
                WHERE source_document_id IS NOT NULL AND source_document_id <> ''
                GROUP BY source_document_id
                HAVING count(DISTINCT coalesce(parser_status, '')) = 1
                   AND count(DISTINCT coalesce(catalog_status, '')) = 1
            )
            SELECT s.source_document_id FROM s JOIN c USING (source_document_id)
            WHERE s.parser_kind = c.parser_kind AND
              ((s.parser_kind = 'supported' AND c.status = 'ready_for_parse') OR
               (s.parser_kind IN ('supported', 'inventory_only', 'unknown') AND c.status = 'auto_parsed'))
        """)
        observations: dict[str, dict[str, Any]] = {}
        for table in dict.fromkeys(t for tables in ROUTES.values() for t in tables):
            path = ledger / (table + '.csv')
            empty = {'count': 0, 'months': [], 'equipment_ids': [], 'source_document_ids': []}
            if not path.is_file():
                observations[table] = empty
                continue
            file_sql = _literal(str(path))
            columns = {row[0] for row in connection.execute(f'DESCRIBE SELECT * FROM read_csv({file_sql}, all_varchar=true)').fetchall()}
            if 'source_document_id' not in columns:
                observations[table] = empty
                continue
            equipment = 't.equipment_id' if 'equipment_id' in columns else 'NULL::VARCHAR'
            order = 't.order_id' if 'order_id' in columns else 'NULL::VARCHAR'
            date_parts = [f"nullif(t.{name}, '')" for name in DATE_COLUMNS if name in columns]
            raw_date = 'coalesce(' + ', '.join(date_parts + ['NULL']) + ')'
            day = f'try_cast(substr({raw_date}, 1, 10) AS DATE)'
            filters = []
            if ids:
                filters.append(f"{equipment} IN (" + ','.join(_literal(x) for x in sorted(ids)) + ')')
            if start:
                filters.append(f"{day} >= DATE {_literal(start.isoformat())}")
            if end:
                filters.append(f"{day} <= DATE {_literal(end.isoformat())}")
            where = ' AND '.join(filters) if filters else 'true'
            connection.execute(f"""CREATE TEMP VIEW {table} AS
                SELECT {equipment} AS equipment_id, {order} AS order_id,
                       {day} AS day, t.source_document_id
                FROM read_csv({file_sql}, all_varchar=true) t
                SEMI JOIN eligible e ON t.source_document_id = e.source_document_id
                WHERE {where}""")
            # Source evidence is a single deterministic example, not an
            # unbounded list of identifiers. Scope output is request-bounded.
            row = connection.execute(f"""SELECT count(*),
                list_slice(list_sort(list(DISTINCT strftime(day, '%Y-%m')) FILTER (WHERE day IS NOT NULL)), 1, 120),
                min(source_document_id), count(DISTINCT strftime(day, '%Y-%m')) > 120,
                min(source_document_id) <> max(source_document_id)
                FROM {table}""").fetchone()
            seen = []
            if ids and row[0]:
                seen = [x[0] for x in connection.execute(f'SELECT DISTINCT equipment_id FROM {table} WHERE equipment_id IS NOT NULL ORDER BY 1').fetchall()]
            observations[table] = {'count': row[0], 'months': row[1] or [], 'equipment_ids': seen,
                                   'source_document_ids': [row[2]] if row[2] else [],
                                   'months_truncated': row[3], 'source_document_ids_sampled': row[4]}
        plan_join = False
        if observations['maintenance_plan_events']['count'] and observations['maintenance_fact_events']['count']:
            plan_join = connection.execute("""SELECT EXISTS (
                SELECT 1 FROM maintenance_plan_events p
                SEMI JOIN maintenance_fact_events f ON p.order_id = f.order_id
                WHERE p.order_id IS NOT NULL AND p.order_id <> '')""").fetchone()[0]
    finally:
        connection.close()
    results = []
    for capability, tables in ROUTES.items():
        present = {table: observations[table] for table in tables if observations[table]['count']}
        missing = ['Полнота наблюдений за период и периметр не подтверждена источником.']
        if not ids:
            missing.append('Не задан точный периметр оборудования.')
        if not start or not end:
            missing.append('Не задан полный период исследования.')
        if not present:
            missing.append('Нет подтверждённых строк для заданного периметра и периода.')
        if capability == 'AN-PLAN-FACT':
            if not plan_join:
                missing.append('Нет подтверждённого совпадения ключей плана и факта.')
        if capability not in active:
            missing.append('Маршрут не зарегистрирован как доступный.')
        results.append({'capability_id': capability, 'route_id': active.get(capability, {}).get('route_id'),
                        'availability': 'partial' if present and capability in active else 'unavailable',
                        'tables': {name: {k: v for k, v in item.items() if k != 'order_ids'} for name, item in present.items()},
                        'missing_evidence': missing})
    return results


def build_coverage(request: dict[str, Any], *, generation_state_path: Path = GENERATION_STATE, root: Path = TOF_ROOT) -> dict[str, Any]:
    ids, start, end = _request(request)
    for _ in range(2):
        before = _generation(generation_state_path)
        # Resolve once: a moving symlink must not mix snapshot files.
        snapshot = {**before, 'ledger_path': str(Path(before['ledger_path']).resolve(strict=True)), 'catalog_path': str(Path(before['catalog_path']).resolve(strict=True))}
        routes = _collect(snapshot, root, ids, start, end)
        after = _generation(generation_state_path)
        fingerprints = ('catalog_sha256', 'source_documents_sha256', 'ledger_schema_sha256')
        same_fingerprints = all(before.get(key) == after.get(key) for key in fingerprints)
        if same_fingerprints and (before['run_id'], snapshot['ledger_path'], snapshot['catalog_path']) == (after['run_id'], str(Path(after['ledger_path']).resolve(strict=True)), str(Path(after['catalog_path']).resolve(strict=True))):
            return {'coverage_version': 1, 'generation': snapshot, 'routes': routes}
    raise RuntimeError('published_generation_changed_during_coverage')
