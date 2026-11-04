"""Fixed, read-only coverage observations for the published TOF generation.

Event presence is not evidence of complete observation over a period.  Until a
published completeness contract exists, routes remain partial or unavailable.
This file is self-contained so it can be sent to the server over SSH.
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

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
    if not isinstance(value, dict) or not isinstance(value.get('run_id'), str) or not value['run_id'] or not isinstance(value.get('ledger_path'), str) or not value['ledger_path']:
        raise ValueError('published_generation_incomplete')
    if not Path(value['ledger_path']).is_dir():
        raise ValueError('published_generation_ledger_missing')
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


def _rows(path: Path):
    if path.is_file():
        with path.open(encoding='utf-8-sig', newline='') as handle:
            yield from csv.DictReader(handle)


def _collect(generation: dict[str, Any], root: Path, ids: set[str], start: date | None, end: date | None) -> list[dict[str, Any]]:
    ledger = Path(generation['ledger_path'])
    capabilities = json.loads((root / 'harness/capabilities/analytics.json').read_text(encoding='utf-8'))
    active = {x.get('id'): x for x in capabilities.get('capabilities', []) if isinstance(x, dict) and x.get('lifecycle') == 'available'}
    # Duplicate document IDs with conflicting statuses are not trusted.
    statuses: dict[str, str] = {}
    for row in _rows(ledger / 'source_documents.csv'):
        key, status = row.get('source_document_id', ''), row.get('parser_status', '')
        if key:
            statuses[key] = status if key not in statuses or statuses[key] == status else ''
    observations: dict[str, dict[str, Any]] = {}
    for table in dict.fromkeys(t for tables in ROUTES.values() for t in tables):
        count = 0
        dates: set[str] = set()
        keys: set[str] = set()
        scope_seen: set[str] = set()
        sources: set[str] = set()
        for row in _rows(ledger / (table + '.csv')):
            source = row.get('source_document_id', '')
            if not source or statuses.get(source, '').lower() not in {'published', 'parsed', 'ready', 'complete', 'completed', 'accepted'}:
                continue
            equipment = row.get('equipment_id', '')
            if ids and equipment not in ids:
                continue
            raw_date = next((row.get(c, '') for c in DATE_COLUMNS if row.get(c)), '')
            try:
                day = date.fromisoformat(raw_date[:10])
            except ValueError:
                day = None
            if (start or end) and (day is None or (start and day < start) or (end and day > end)):
                continue
            count += 1
            if day:
                dates.add(day.isoformat()[:7])
            if equipment:
                scope_seen.add(equipment)
            if row.get('order_id'):
                keys.add(row['order_id'])
            sources.add(source)
        observations[table] = {'count': count, 'months': sorted(dates), 'equipment_ids': sorted(scope_seen), 'order_ids': keys, 'source_document_ids': sorted(sources)}
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
            plan = observations['maintenance_plan_events']['order_ids']
            fact = observations['maintenance_fact_events']['order_ids']
            if not plan.intersection(fact):
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
        snapshot = {**before, 'ledger_path': str(Path(before['ledger_path']).resolve(strict=True))}
        routes = _collect(snapshot, root, ids, start, end)
        after = _generation(generation_state_path)
        if (before['run_id'], snapshot['ledger_path']) == (after['run_id'], str(Path(after['ledger_path']).resolve(strict=True))):
            return {'coverage_version': 1, 'generation': snapshot, 'routes': routes}
    raise RuntimeError('published_generation_changed_during_coverage')
