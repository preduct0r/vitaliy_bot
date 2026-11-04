"""Evidence-bounded hypothesis cards and deterministic private pilot exports."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from .hypotheses_interests import ATTRIBUTIONS
from .hypotheses_topic_projection import _TOPICS
from .models import ModelError

TEXT_FIELDS = ('management_question', 'decision', 'scope', 'metrics_method',
               'falsification', 'analyst_brief', 'novelty')
LIST_FIELDS = ('source_references', 'missing_evidence', 'hard_questions')
SCORES = ('utility', 'feasibility', 'interest', 'novelty')


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} required')
    return value


def _interests(context: Mapping[str, Any]) -> dict:
    version = _text(context.get('version'), 'interest version')
    corpus_version = _text(context.get('corpus_version'), 'corpus version')
    items = context.get('interests')
    if not isinstance(items, list) or not items:
        raise ValueError('interests required')
    projected = []
    for item in items:
        ident = _text(item.get('id'), 'interest id')
        requirement = _text(item.get('requirement'), 'requirement')
        attributions = item.get('attributions')
        if not isinstance(attributions, list) or not attributions or any(a not in ATTRIBUTIONS for a in attributions):
            raise ValueError('invalid interest attribution')
        if any(i['id'] == ident for i in projected):
            raise ValueError('duplicate interest')
        evidence = item.get('evidence')
        if not isinstance(evidence, list) or not evidence:
            raise ValueError('verified interest evidence required')
        safe = []
        for entry in evidence:
            if not isinstance(entry, Mapping):
                raise ValueError('invalid evidence')
            sid, locator = entry.get('source_id'), entry.get('locator')
            if not isinstance(sid, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', sid) or not isinstance(locator, str) or not re.fullmatch(r'chars:[0-9]+-[0-9]+', locator):
                raise ValueError('unsafe evidence reference')
            when = _text(entry.get('date'), 'source date')
            try:
                if date.fromisoformat(when).isoformat() != when:
                    raise ValueError('invalid date')
            except ValueError as exc:
                raise ValueError('invalid source date') from exc
            digest = entry.get('sha256')
            if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ValueError('invalid source hash')
            if entry.get('attribution') not in attributions or entry.get('reference') != f'{sid}:{locator}':
                raise ValueError('invalid evidence attribution or reference')
            safe.append({k: entry[k] for k in ('source_id', 'date', 'locator', 'attribution', 'sha256', 'reference')})
        projected.append({'id': ident, 'requirement': requirement, 'attributions': list(attributions), 'evidence': safe})
    return {'version': version, 'corpus_version': corpus_version, 'interests': projected}


def _coverage(value: Mapping[str, Any]) -> dict:
    if not isinstance(value, Mapping) or value.get('coverage_version') != 1:
        raise ValueError('invalid coverage')
    generation = value.get('generation')
    if not isinstance(generation, Mapping):
        raise ValueError('missing generation')
    _text(generation.get('run_id'), 'generation id')
    routes = value.get('routes')
    if not isinstance(routes, list) or not routes:
        raise ValueError('coverage routes required')
    seen = set()
    for route in routes:
        ident = _text(route.get('capability_id'), 'capability id')
        if ident in seen:
            raise ValueError(f'duplicate coverage capability id: {ident}')
        seen.add(ident)
        if route.get('availability') not in ('available', 'partial', 'unavailable'):
            raise ValueError('invalid coverage availability')
        if not isinstance(route.get('missing_evidence'), list) or any(not isinstance(x, str) or not x.strip() for x in route['missing_evidence']):
            raise ValueError('missing evidence list required')
    return copy.deepcopy(value)


def _references(route: Mapping[str, Any]) -> set[str]:
    references = set(route.get('source_references', []))
    for table in route.get('tables', {}):
        references.add('ledger:' + table)
    return references


def validate_pilot(run: Mapping[str, Any], *, coverage: Mapping[str, Any], interest_context: Mapping[str, Any]) -> None:
    coverage = _coverage(coverage)
    interests = _interests(interest_context)
    if run.get('generation_id') != coverage['generation']['run_id'] or run.get('interest_version') != interests['version']:
        raise ValueError('snapshot version mismatch')
    _text(run.get('knowledge_version'), 'knowledge version')
    cards = run.get('cards')
    if not isinstance(cards, list) or not 5 <= len(cards) <= 7:
        raise ValueError('five to seven cards required')
    ids = {i['id']: i for i in interests['interests']}
    projection_bindings = None
    if 'verified_labels' in run or 'verifier_audit' in run:
        projection_bindings = _projection(interests, run.get('verified_labels'), run.get('verifier_audit'))
    routes = {r['capability_id']: r for r in coverage['routes']}
    for number, card in enumerate(cards, 1):
        if not isinstance(card, dict) or card.get('number') != number:
            raise ValueError('cards must be numbered consecutively')
        for field in TEXT_FIELDS:
            _text(card.get(field), field)
        for field in LIST_FIELDS:
            value = card.get(field)
            if not isinstance(value, list) or (field != 'missing_evidence' and not value):
                raise ValueError(f'{field} must be a list')
            for entry in value:
                _text(entry, field)
        meeting = card.get('meeting_interest')
        if not isinstance(meeting, dict) or not isinstance(meeting.get('interest_id'), str) or meeting['interest_id'] not in ids:
            raise ValueError('unknown meeting interest')
        evidence = ids[meeting['interest_id']]['evidence']
        if projection_bindings is not None and 'topic_id' not in meeting:
            raise ValueError('missing projected topic')
        if 'topic_id' in meeting:
            if projection_bindings is None or (meeting['interest_id'], meeting['topic_id']) not in projection_bindings:
                raise ValueError('unknown projection binding')
            if not all(meeting.get(k) == v for k, v in projection_bindings[(meeting['interest_id'], meeting['topic_id'])].items()):
                raise ValueError('projection evidence mismatch')
        fields = ('reference', 'attribution', 'source_id', 'date', 'locator', 'sha256')
        # Historical exports carry reference and attribution only. If any metadata
        # is present, require the entire binding to match one evidence entry.
        present = fields if any(k in meeting for k in fields[2:]) else fields[:2]
        if not any(all(meeting.get(k) == e[k] for k in present) for e in evidence):
            raise ValueError('unsupported meeting reference or attribution')
        capability = _text(card.get('capability_id'), f'card {number} capability_id')
        route = routes.get(capability)
        if route is None:
            raise ValueError(f'card {number}: unknown capability_id {capability}')
        refs = set(card['source_references'])
        unsupported = refs - _references(route)
        if unsupported:
            raise ValueError(f'card {number}: references unsupported by {capability}: {sorted(unsupported)}')
        if card.get('source_readiness') != route['availability']:
            raise ValueError(f'card {number}: readiness mismatches {capability}')
        if not set(route['missing_evidence']) <= set(card['missing_evidence']):
            raise ValueError(f'card {number}: missing evidence omitted for {capability}')
        ranking = card.get('ranking')
        if not isinstance(ranking, dict) or any(type(ranking.get(k)) not in (int, float) or not 0 <= ranking[k] <= 5 for k in SCORES):
            raise ValueError('invalid ranking')
        if card['novelty'].casefold() in ('unknown', 'неизвестна', 'неизвестно') and ranking['novelty'] != 0:
            raise ValueError('unknown novelty cannot be scored as proven')



def _projection(interests, labels, audit):
    """Bind each canonical label to exactly one high-confidence audited source."""
    if not isinstance(labels, list) or not labels or not isinstance(audit, list):
        raise ValueError('verified labels and audit required')
    canonical = {topic: description for topic, description, _ in _TOPICS}
    by_id = {item['id']: item for item in interests['interests']}
    entries = {}
    for label in labels:
        if not isinstance(label, Mapping):
            raise ValueError('invalid label')
        ident, topic = label.get('interest_id'), label.get('topic_id')
        key = (ident, topic)
        if key in entries or ident not in by_id or topic not in canonical or label.get('description') != canonical[topic]:
            raise ValueError('invalid or duplicate label')
        matches = [a for a in audit if isinstance(a, Mapping) and (a.get('interest_id'), a.get('topic_id')) == key]
        if len(matches) != 1:
            raise ValueError('missing or ambiguous projection audit')
        record = matches[0]
        verdict = record.get('verdict')
        if not isinstance(verdict, Mapping) or verdict.get('supported') is not True or verdict.get('confidence') != 'high':
            raise ValueError('projection audit rejected')
        matching = [e for e in by_id[ident]['evidence'] if e['source_id'] == record.get('source_id') and e['locator'] == record.get('locator')]
        if len(matching) != 1:
            raise ValueError('projection source not bound')
        entries[key] = matching[0]
    return entries


def _validate_card_content(card, scope, topic, private_interests, verifier):
    # Numeric facts in prose must occur in this request. IDs, scores and
    # route metadata are structural and never evidence for numeric claims.
    prose = [card.get(k, '') for k in TEXT_FIELDS] + sum((card.get(k, []) for k in ('hard_questions',) if isinstance(card.get(k), list)), [])
    prose = [v for v in prose if isinstance(v, str)]
    text = ' '.join(prose)
    numeric = r'\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?|\b(?:один|одна|одно|одного|одной|два|две|двух|три|трёх|трех|четыре|четырёх|четырех|пять|пяти|шесть|шести|семь|семи|восемь|восьми|девять|девяти|ноль|нуль|миллион\w*|миллиард\w*|тысяч\w*|сотен|сотня\w*|сто|двести|триста|четыреста|пятьсот|шестьсот|семьсот|восемьсот|девятьсот|десять|двадцать|тридцать|сорок|пятьдесят|шестьдесят|семьдесят|восемьдесят|девяносто)\b'
    numbers = lambda value: set(re.findall(numeric, value.casefold(), re.I))
    if numbers(text) - numbers(scope):
        raise ValueError('unsupported numerical value')
    # Avoid allowing other case identifiers and distinctive private tokens to
    # cross into this case even when a semantic verifier makes a mistake.
    approved = set(re.findall(r'[\w-]+', scope.casefold()))
    private_tokens = set()
    for item in private_interests['interests']:
        for value in item.values():
            if isinstance(value, str):
                private_tokens.update(t.casefold() for t in re.findall(r'[A-ZА-ЯЁ_]{4,}', value) if t.casefold() not in approved)
    if any(token in text.casefold() for token in private_tokens):
        raise ValueError('private source token in card')
    # Each question needs a positive repair-topic signal, not just a generic
    # label on an unrelated question. Remaining semantics are independently checked.
    question = card.get('management_question', '')
    markers = {
      'maintenance_history_repeated_episodes': r'ремонт|повтор|эпизод|отказ|полом',
      'maintenance_plan_fact': r'ремонт|план|факт',
      'maintenance_scheduling_resource_load': r'ремонт|график|ресурс|бригад',
      'mtr_cost_supply': r'ремонт|мтр|постав|материал',
      'equipment_downtime': r'ремонт|просто',
      'labor_norm_observed_measurement': r'ремонт|трудо|норм|замер',
      'equipment_criticality': r'ремонт|критичност|оборудован',
      'techcard_coverage_verification': r'ремонт|техкарт|технологическ',
      'maintenance_data_quality': r'ремонт|данн',
      'implementation_measurability': r'ремонт|измеримост|внедрен',
    }
    if not isinstance(question, str) or not re.search(markers[topic], question, re.I):
        raise ValueError('question incompatible with repair topic')
    try:
        safe_card = {k: copy.deepcopy(card[k]) for k in TEXT_FIELDS + ('hard_questions',) if k in card}
        verdict = verifier({'card': safe_card, 'scope': scope,
                            'topic_id': topic,
                            'description': dict((t, d) for t, d, _ in _TOPICS)[topic]})
    except Exception as exc:
        raise ValueError('private verifier failed') from exc
    if not isinstance(verdict, Mapping) or verdict.get('supported') is not True or verdict.get('confidence') != 'high':
        raise ValueError('private verifier rejected or uncertain')

def generate_pilot(*, scope: str, knowledge_version: str, interest_context: Mapping[str, Any], coverage_provider, generator, verified_labels=None, verifier_audit=None, private_verifier=None) -> dict:
    _text(knowledge_version, 'knowledge version')
    if not isinstance(scope, str):
        raise ValueError('scope must be text')
    interests = _interests(interest_context)
    if not callable(generator) or not callable(private_verifier) or generator is private_verifier:
        raise ValueError('independent generator and private verifier required')
    bindings = _projection(interests, verified_labels, verifier_audit)
    # Fetch before invoking the model; there is deliberately no local fallback.
    try:
        coverage = _coverage(coverage_provider())
    except Exception as exc:
        raise RuntimeError('published server coverage unavailable: ' + str(exc)) from exc
    safe_routes = [{k: copy.deepcopy(r[k]) for k in ('capability_id', 'route_id', 'availability', 'missing_evidence') if k in r}
                   | {'source_references': sorted(_references(r))} for r in coverage['routes']]
    model_interests = [{k: label[k] for k in ('interest_id', 'topic_id', 'description')}
                       for label in verified_labels]
    context = {'scope': scope, 'knowledge_version': knowledge_version, 'interests': model_interests,
               'coverage': {'generation_id': coverage['generation']['run_id'], 'routes': safe_routes},
               'instructions': ('Предложите 5–7 гипотез, не выполняя анализ. Не выдумывайте источники, готовность или эффект. '
                                'Укажите ограничения, проверку опровержения и вопросы аналитика. '
                                'Различайте заказ и эпизод, длительность и трудоёмкость, план и факт, норму и оценку. '
                                'Выберите точный capability_id одного маршрута и только его source_references; '
                                'не заявляйте планируемый эффект сценария как установленный факт. '
                                'Эффект сценария — только непроверенное исследование с недостающей методикой. '
                                'Сформулируйте различные аналитические управленческие вопросы: по одной гипотезе на вопрос. '
                                'Рассмотрите качество данных, покрытие техкарт, проверку нормативной и измеренной трудоёмкости, '
                                'ресурсные ограничения графика с привязкой к обучающим интересам. '
                                'Не утверждайте непроверенных фактов; укажите interest_id и topic_id выбранной пары. '
                                'Интерпретация и пересказ третьего лица не являются подтверждённым прямым запросом. '
                                'Неизвестную новизну укажите явно и оцените нулём.')}
    cards = generator(copy.deepcopy(context))
    run = {'generation_id': coverage['generation']['run_id'], 'knowledge_version': knowledge_version,
           'interest_version': interests['version'], 'scope': scope, 'coverage': coverage,
           'interest_context': interests, 'verified_labels': copy.deepcopy(verified_labels),
           'verifier_audit': copy.deepcopy(verifier_audit), 'cards': copy.deepcopy(cards)}
    routes = {r['capability_id']: r for r in coverage['routes']}
    by_id = {item['id']: item for item in interests['interests']}
    if isinstance(run['cards'], list):
        for number, card in enumerate(run['cards'], 1):
            if not isinstance(card, dict):
                raise ValueError(f'card {number}: expected object')
            capability = _text(card.get('capability_id'), f'card {number} capability_id')
            if capability not in routes:
                raise ValueError(f'card {number}: unknown capability_id {capability}')
            route = routes[capability]
            card['source_readiness'] = route['availability']
            card['missing_evidence'] = copy.deepcopy(route['missing_evidence'])
            meeting = card.get('meeting_interest')
            if not isinstance(meeting, dict) or not isinstance(meeting.get('interest_id'), str) or meeting['interest_id'] not in by_id:
                raise ValueError('unknown meeting interest')
            topic = meeting.get('topic_id')
            if (meeting['interest_id'], topic) not in bindings:
                raise ValueError('unknown projected interest and topic')
            entry = bindings[(meeting['interest_id'], topic)]
            fields = ('reference', 'attribution', 'source_id', 'date', 'locator', 'sha256')
            supplied = [k for k in fields if k in meeting]
            if supplied and (not {'reference', 'attribution'} <= set(supplied) or
                             any(meeting[k] != entry[k] for k in supplied)):
                raise ValueError('unsupported meeting reference or attribution')
            if set(card.get('source_references', [])) - _references(route):
                raise ValueError('references unsupported by selected route')
            _validate_card_content(card, scope, topic, interests, private_verifier)
            card['meeting_interest'] = {'interest_id': meeting['interest_id'], 'topic_id': topic, **entry}

    validate_pilot(run, coverage=coverage, interest_context=interests)
    run['cards'].sort(key=lambda c: sum(c['ranking'][k] for k in SCORES), reverse=True)
    for number, card in enumerate(run['cards'], 1):
        card['number'] = number
    return run


def export_pilot(run: Mapping[str, Any], *, output_dir: Path, source_verification: Mapping[str, Any] | None) -> None:
    if not isinstance(source_verification, Mapping) or source_verification.get('verified') is not True:
        raise ValueError('verified source report required')
    checks = source_verification.get('checks')
    if not isinstance(checks, list) or not checks or any(not isinstance(c, dict) or c.get('verified') is not True for c in checks):
        raise ValueError('all source checks must be verified')
    if source_verification.get('interest_version') != run['interest_version'] or source_verification.get('corpus_version') != run['interest_context'].get('corpus_version'):
        raise ValueError('source verification version mismatch')
    validate_pilot(run, coverage=run['coverage'], interest_context=run['interest_context'])
    for item in _interests(run['interest_context'])['interests']:
        for entry in item['evidence']:
            if not any(c.get('interest_id') == item['id'] and all(c.get(k) == entry[k] for k in ('source_id', 'locator', 'date', 'sha256', 'attribution')) and c.get('verified') is True for c in checks):
                raise ValueError('source verification does not match evidence')
    output_dir = Path(output_dir).resolve()
    repo = Path(__file__).resolve().parents[3]
    if output_dir == repo or repo in output_dir.parents:
        raise ValueError('pilot outputs must be outside the checkout')
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    for name, value in (('pilot.json', run), ('source-verification.json', source_verification)):
        data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
        fd = os.open(output_dir / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)


def codex_generator(runner):
    """Adapt the existing schema-constrained runner to the offline generator."""
    string = {'type': 'string'}
    string_list = {'type': 'array', 'items': string}
    meeting = {'type': 'object', 'additionalProperties': False,
               'properties': {'interest_id': string, 'topic_id': string}, 'required': ['interest_id', 'topic_id']}
    ranking = {'type': 'object', 'additionalProperties': False,
               'properties': {k: {'type': 'number'} for k in SCORES}, 'required': list(SCORES)}
    properties = {k: string for k in TEXT_FIELDS + ('source_readiness', 'capability_id')}
    properties.update({k: string_list for k in LIST_FIELDS})
    properties.update({'number': {'type': 'integer'}, 'meeting_interest': meeting, 'ranking': ranking})
    schema = {'type': 'object', 'additionalProperties': False,
              'properties': {'cards': {'type': 'array', 'minItems': 5, 'maxItems': 7,
                  'items': {'type': 'object', 'additionalProperties': False,
                            'properties': properties, 'required': list(properties)}}},
              'required': ['cards']}
    def generate(context):
        prompt = 'Верните JSON с карточками гипотез на русском языке. Контекст:\n' + json.dumps(context, ensure_ascii=False, sort_keys=True)
        try:
            return asyncio.run(runner.run(prompt, schema)).value['cards']
        except ModelError as exc:
            raise RuntimeError(f'model error {exc.code}: {exc.detail}') from exc
    return generate


def subprocess_verifier(path: Path):
    """A separately supplied private verifier using JSON stdin/stdout, no shell."""
    path = Path(path)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError('executable private verifier required')
    def verify(context):
        try:
            result = subprocess.run([str(path)], input=json.dumps(context, ensure_ascii=False),
                                    text=True, capture_output=True, timeout=480, check=True)
            return json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            raise ValueError('private verifier failed') from exc
    return verify
