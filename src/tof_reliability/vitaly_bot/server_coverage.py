"""Fixed, read-only coverage observations for the published TOF generation.

Event presence is not evidence of complete observation over a period.  Until a
published completeness contract exists, routes remain partial or unavailable.
This file is self-contained so it can be sent to the server over SSH.
"""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import date
from pathlib import Path
from typing import Any


GENERATION_STATE = Path('/srv/tof/state/current_generation.json')
TOF_ROOT = Path('/srv/tof/repos/tof_v1')
ROUTES = {
    'AN-REPAIR-EPISODES': 'Производные эпизоды не раскрывают полный состав источников и их статусы needs_review.',
    'AN-PLAN-FACT': 'Методика сопоставления плановых позиций и ремонтных эпизодов с проверяемыми ключами и происхождением не опубликована.',
    'AN-MTR-COST': 'Полная цепочка источников производных затрат и методика связи со списаниями МТР не подтверждены.',
    'AN-DOWNTIME': 'Методика связи простоя с заказом и полное происхождение для периметра не подтверждены.',
    'AN-TECHCARD-LABOR': 'Источники техкарт не подтверждают полноту норм времени или соответствие фактическим ремонтам за период.',
    'AN-EQUIPMENT-CRITICALITY': 'Источники критичности не подтверждают полноту или актуальность оценок для всего периметра и периода.',
}


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


def _semantic_query(generation: dict[str, Any], ids: set[str], table: str, role: str = '') -> dict[str, Any]:
    # Only fixed tables and fixed projection; values are escaped, never model SQL.
    where = []
    if ids:
        where.append('equipment_key IN (' + ','.join('equipment_key(' + _literal(x) + ')' for x in sorted(ids)) + ')')
    if role:
        where.append("role = 'criticality'")
    sql = 'SELECT DISTINCT source_document_id FROM ' + table
    if where:
        sql += ' WHERE ' + ' AND '.join(where)
    sql += ' ORDER BY source_document_id'
    completed = subprocess.run(['tof-data', 'query', '--generation', generation['run_id'], '--limit', '10000', '--json', sql],
                               check=True, capture_output=True, text=True, timeout=90)
    return json.loads(completed.stdout)


def _criticality_query(generation: dict[str, Any], ids: set[str]) -> dict[str, Any]:
    return _semantic_query(generation, ids, 'equipment_sources', 'criticality')


def _techcard_query(generation: dict[str, Any], ids: set[str]) -> dict[str, Any]:
    return _semantic_query(generation, ids, 'tech_card_sources')


def _eligible_sources(generation: dict[str, Any]) -> set[str]:
    def rows(path):
        with Path(path).open(encoding='utf-8-sig', newline='') as stream:
            return list(csv.DictReader(stream))
    sources, catalog = {}, {}
    for row in rows(Path(generation['ledger_path']) / 'source_documents.csv'):
        sources.setdefault(row.get('source_document_id'), set()).add(row.get('parser_status', ''))
    for row in rows(generation['catalog_path']):
        catalog.setdefault(row.get('source_document_id'), set()).add((row.get('parser_status', ''), row.get('catalog_status', '')))
    eligible = set()
    for ident, statuses in sources.items():
        if not ident or len(statuses) != 1 or len(catalog.get(ident, ())) != 1:
            continue
        parser = next(iter(statuses))
        cat_parser, status = next(iter(catalog[ident]))
        if parser == cat_parser and ((parser == 'supported' and status == 'ready_for_parse') or
                                     (parser in ('supported', 'inventory_only', 'unknown') and status == 'auto_parsed')):
            eligible.add(ident)
    return eligible


def _source_observation(result: dict[str, Any], generation: dict[str, Any], eligible: set[str]) -> list[str]:
    if not isinstance(result, dict) or result.get('generation') != generation['run_id']:
        raise ValueError('semantic_query_generation_mismatch')
    if result.get('columns') != ['source_document_id'] or not isinstance(result.get('rows'), list) or type(result.get('truncated')) is not bool:
        raise ValueError('invalid_semantic_query_result')
    if result['truncated']:
        return []
    if any(not isinstance(row, list) or len(row) != 1 or (row[0] is not None and not isinstance(row[0], str)) for row in result['rows']):
        raise ValueError('invalid_semantic_query_result')
    return sorted({row[0] for row in result['rows'] if row[0] in eligible})


def _collect(generation: dict[str, Any], root: Path, ids: set[str], start: date | None, end: date | None) -> list[dict[str, Any]]:
    capabilities = json.loads((root / 'harness/capabilities/analytics.json').read_text(encoding='utf-8'))
    active = {x.get('id'): x for x in capabilities.get('capabilities', []) if isinstance(x, dict) and x.get('lifecycle') == 'available'}
    eligible = _eligible_sources(generation)
    observations = {
        'AN-EQUIPMENT-CRITICALITY': ('equipment', _source_observation(_criticality_query(generation, ids), generation, eligible)),
        'AN-TECHCARD-LABOR': ('tech_card', _source_observation(_techcard_query(generation, ids), generation, eligible)),
    }
    results = []
    for capability, limitation in ROUTES.items():
        missing = [limitation, 'Источники со статусом needs_review исключены; полнота наблюдений не подтверждена.']
        if not ids:
            missing.append('Не задан точный периметр оборудования; источники не подтверждают охват всего оборудования.')
        if not start or not end:
            missing.append('Не задан полный период исследования.')
        table, sources = observations.get(capability, ('', []))
        if capability not in active:
            missing.append('Маршрут не зарегистрирован как доступный.')
        present = bool(sources and capability in active)
        results.append({'capability_id': capability, 'route_id': active.get(capability, {}).get('route_id'),
                        'availability': 'partial' if present else 'unavailable',
                        'tables': {table: {'source_document_ids': sources[:20], 'source_document_ids_sampled': len(sources) > 20}} if present else {},
                        'missing_evidence': missing})
    return results


def build_coverage(request: dict[str, Any], *, generation_state_path: Path = GENERATION_STATE, root: Path = TOF_ROOT,
                   max_generation_retries: int = 1) -> dict[str, Any]:
    ids, start, end = _request(request)
    if type(max_generation_retries) is not int or max_generation_retries not in (0, 1):
        raise ValueError('invalid_generation_retry_limit')
    for _ in range(1 + max_generation_retries):
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
