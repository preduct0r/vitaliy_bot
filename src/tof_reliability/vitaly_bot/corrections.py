"""Versioned audit overlays: the original report is never rewritten."""
from __future__ import annotations

import copy
import hashlib
import json
import uuid
import re
from pathlib import Path
from .models import StateError, ModelError
from .storage import _json, _utc_now

SECTIONS = {'G': 'critical_gaps', 'C': 'contradictions', 'Q': 'questions'}
LABELS = {'corrected': 'Исправлено', 'removed': 'Замечание снято',
          'upheld': 'Оставлено в силе', 'clarification': 'Нужно уточнение'}


def items(review):
    return [(prefix + str(i + 1), item) for prefix, section in SECTIONS.items()
            for i, item in enumerate(review.get(section, []))]


def item_at(review, item_id):
    try:
        index = int(item_id[1:]) - 1
        if index < 0:
            raise ValueError()
        return review[SECTIONS[item_id[0]]][index]
    except (KeyError, ValueError, IndexError):
        raise StateError('Пункт не найден. Откройте актуальный отчёт.')


class Corrections:
    def __init__(self, store):
        self.store = store
        with store._connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS audit_versions (
                    id TEXT PRIMARY KEY, case_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    original_json TEXT NOT NULL, current_json TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0, pending_item TEXT,
                    decisions_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                    UNIQUE(case_id, fingerprint));
                CREATE TABLE IF NOT EXISTS audit_corrections (
                    job_id TEXT PRIMARY KEY, audit_id TEXT NOT NULL, item_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, explanation TEXT NOT NULL,
                    before_json TEXT NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS audit_messages (
                    audit_id TEXT NOT NULL, chat_id INTEGER NOT NULL, message_id INTEGER NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'control',
                    PRIMARY KEY(chat_id, message_id));
            ''')
            if 'kind' not in {r['name'] for r in db.execute('PRAGMA table_info(audit_messages)')}:
                db.execute("ALTER TABLE audit_messages ADD COLUMN kind TEXT NOT NULL DEFAULT 'control'")

    def _fingerprint(self, row, db):
        artifacts = db.execute('SELECT id,sha256 FROM artifacts WHERE case_id=? ORDER BY id', (row['id'],)).fetchall()
        return hashlib.sha256((_json(json.loads(row['review_json'])) +
                               _json([tuple(a) for a in artifacts])).encode()).hexdigest()

    def _assert_current(self, db, audit, chat_id=None, idle=False):
        if audit is None:
            raise StateError('Этот отчёт больше недоступен. Выполните /review.')
        row = db.execute('SELECT * FROM cases WHERE id=?', (audit['case_id'],)).fetchone()
        latest = db.execute('SELECT id FROM cases WHERE chat_id=? ORDER BY created_at DESC LIMIT 1',
                            (row['chat_id'],)).fetchone()
        if (chat_id is not None and row['chat_id'] != chat_id) or latest['id'] != row['id'] or not row['review_json']:
            raise StateError('Старый кейс или материалы изменены. Откройте актуальный отчёт.')
        if row['mode'] == 'expired' or self._fingerprint(row, db) != audit['fingerprint']:
            raise StateError('Материалы изменены. Выполните /review.')
        if idle:
            if row['mode'] in {'grilling', 'stopping'}:
                raise StateError('Сначала завершите репетицию командой /stop.')
            if row['activity'] != 'idle':
                raise StateError('Эта операция уже выполняется; повторная команда не добавлена.')
        return row

    def ensure(self, case_id):
        with self.store._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM cases WHERE id=?', (case_id,)).fetchone()
            if row is None or not row['review_json']:
                raise StateError('Сначала выполните /review.')
            fingerprint = self._fingerprint(row, db)
            db.execute('UPDATE audit_versions SET pending_item=NULL WHERE case_id=? AND fingerprint<>?', (case_id, fingerprint))
            db.execute('INSERT OR IGNORE INTO audit_versions(id,case_id,fingerprint,original_json,current_json,created_at) VALUES (?,?,?,?,?,?)',
                       (uuid.uuid4().hex[:16], case_id, fingerprint, row['review_json'], row['review_json'], _utc_now()))
            result = db.execute('SELECT * FROM audit_versions WHERE case_id=? AND fingerprint=?', (case_id, fingerprint)).fetchone()
        return dict(result)

    def get(self, audit_id, chat_id=None, idle=False):
        with self.store._connect() as db:
            row = db.execute('SELECT * FROM audit_versions WHERE id=?', (audit_id,)).fetchone()
            self._assert_current(db, row, chat_id, idle)
            return dict(row)

    def select(self, audit_id, revision, item_id, chat_id):
        with self.store._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM audit_versions WHERE id=?', (audit_id,)).fetchone()
            self._assert_current(db, row, chat_id, idle=True)
            if row['revision'] != revision:
                raise StateError('Отчёт обновился. Выберите пункт заново.')
            if item_id:
                item_at(json.loads(row['current_json']), item_id)
            db.execute('UPDATE audit_versions SET pending_item=? WHERE id=?', (item_id, audit_id))

    def pending(self, case_id):
        case = self.store.get_case(case_id)
        if not case['review']:
            return None
        audit = self.ensure(case_id)
        return audit if audit['pending_item'] else None

    def cancel(self, case_id):
        with self.store._connect() as db:
            db.execute('UPDATE audit_versions SET pending_item=NULL WHERE case_id=?', (case_id,))

    def submit(self, audit_id, chat_id, explanation):
        if not explanation.strip():
            raise StateError('Напишите, что именно неверно в выбранном пункте.')
        with self.store._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            audit = db.execute('SELECT * FROM audit_versions WHERE id=?', (audit_id,)).fetchone()
            self._assert_current(db, audit, chat_id, idle=True)
            if not audit['pending_item']:
                raise StateError('Выберите пункт кнопкой «Исправить пункт».')
            payload = {'audit_id': audit_id, 'revision': audit['revision'],
                       'item_id': audit['pending_item'], 'explanation': explanation.strip()}
            job_id, now = uuid.uuid4().hex, _utc_now()
            db.execute('INSERT INTO jobs(id,case_id,chat_id,kind,payload_json,status,created_at) VALUES (?,?,?,?,?,?,?)',
                       (job_id, audit['case_id'], chat_id, 'review_correction', _json(payload), 'queued', now))
            db.execute('UPDATE cases SET activity=?,activity_detail=?,activity_started_at=?,active_job_id=?,updated_at=? WHERE id=?',
                       ('queued', 'Перепроверяю пункт ' + audit['pending_item'], now, job_id, now, audit['case_id']))
        return self.store.get_job(job_id)

    def history(self, audit_id):
        with self.store._connect() as db:
            return [dict(row) for row in db.execute('SELECT * FROM audit_corrections WHERE audit_id=? ORDER BY created_at', (audit_id,))]

    def apply(self, job, result):
        payload = job['payload']
        with self.store._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT result_json FROM audit_corrections WHERE job_id=?', (job['id'],)).fetchone()
            if previous:
                return json.loads(previous[0])
            audit = db.execute('SELECT * FROM audit_versions WHERE id=?', (payload['audit_id'],)).fetchone()
            self._assert_current(db, audit, job['chat_id'])
            if audit['revision'] != payload['revision']:
                raise StateError('Отчёт изменился во время проверки. Выберите пункт заново.')
            review = json.loads(audit['current_json'])
            old = copy.deepcopy(item_at(review, payload['item_id']))
            status = result.get('decision')
            if status not in LABELS or not isinstance(result.get('explanation'), str) or not result['explanation'].strip():
                raise ModelError('model_output_invalid_schema', 'Некорректное решение исправления')
            refs = result.get('references')
            if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
                raise ModelError('model_output_invalid_schema', 'Некорректные ссылки')
            suppressed = result.get('suppressed_question_ids', [])
            valid_questions = {key for key, _ in items(review) if key.startswith('Q')}
            if not isinstance(suppressed, list) or any(q not in valid_questions for q in suppressed):
                raise ModelError('model_output_invalid_schema', 'Неизвестный связанный вопрос')
            if status in {'upheld', 'clarification'} and suppressed:
                raise ModelError('model_output_invalid_schema', 'Нельзя скрывать вопросы без принятого исправления')
            if status != 'clarification' and not refs:
                raise ModelError('model_output_invalid_schema', 'Нет основания решения')
            if status == 'clarification' and not result.get('clarifying_question', '').strip():
                raise ModelError('model_output_invalid_schema', 'Нет уточняющего вопроса')
            if status == 'corrected':
                replacement = result.get('replacement')
                if not isinstance(replacement, dict) or set(replacement) != set(old):
                    raise ModelError('model_output_invalid_schema', 'Изменена структура пункта')
                for key, value in old.items():
                    if type(replacement[key]) is not type(value):
                        raise ModelError('model_output_invalid_schema', 'Изменён тип поля пункта')
                item_at(review, payload['item_id']).update(replacement)
            decisions = json.loads(audit['decisions_json'])
            # An unsuccessful appeal must not erase a previously accepted correction.
            if status in {'corrected', 'removed'}:
                decisions[payload['item_id']] = result
            db.execute('INSERT INTO audit_corrections VALUES (?,?,?,?,?,?,?,?)',
                       (job['id'], audit['id'], payload['item_id'], audit['revision'], payload['explanation'], _json(old), _json(result), _utc_now()))
            db.execute('UPDATE audit_versions SET current_json=?,revision=revision+1,decisions_json=?,pending_item=? WHERE id=?',
                       (_json(review), _json(decisions), payload['item_id'] if status == 'clarification' else None, audit['id']))
        return result

    def effective(self, case_id, rehearsal=False):
        audit = self.ensure(case_id)
        review, decisions = json.loads(audit['current_json']), json.loads(audit['decisions_json'])
        if rehearsal:
            excluded = {q for d in decisions.values() for q in d.get('suppressed_question_ids', [])}
            excluded.update(key for key, d in decisions.items() if key.startswith('Q') and d['decision'] == 'removed')
            review['questions'] = [q for i, q in enumerate(review['questions'], 1) if 'Q'+str(i) not in excluded]
        return review

    def context(self, case_id):
        case = self.store.get_case(case_id)
        if not case['review']:
            return '{}'
        audit = self.ensure(case_id)
        return audit['decisions_json']

    def record_message(self, audit_id, chat_id, message_id, kind='control'):
        with self.store._connect() as db:
            db.execute('INSERT OR IGNORE INTO audit_messages(audit_id,chat_id,message_id,kind) VALUES (?,?,?,?)', (audit_id, chat_id, message_id, kind))

    def messages(self, audit_id):
        with self.store._connect() as db:
            return [dict(r) for r in db.execute('SELECT * FROM audit_messages WHERE audit_id=?', (audit_id,))]


async def correct_point(service, job):
    from .service import _object, QUESTION_PROPERTIES
    payload = job['payload']
    corrections = service.corrections
    audit = corrections.get(payload['audit_id'], job['chat_id'])
    for previous in corrections.history(audit['id']):
        if previous['job_id'] == job['id']:
            return json.loads(previous['result_json'])
    if audit['revision'] != payload['revision']:
        raise StateError('Отчёт обновился; выберите пункт заново.')
    for artifact in service.store.artifacts(job['case_id']):
        if not Path(artifact['local_path']).is_file():
            raise StateError('Исходники удалены. Создайте /new и загрузите материал заново.')
    review = json.loads(audit['current_json'])
    selected = item_at(review, payload['item_id'])
    prepared = service.documents.prepared(job['case_id'])
    target_text = str(selected.get('material_reference', '')) + ' ' + payload['explanation']
    named = [p for p in prepared if p.name.casefold() in target_text.casefold()]
    if named:
        prepared = named
    requested_pages = [int(v) for v in re.findall(r'(?i)(?:слайд\w*|страниц\w*)\s*(\d+)',
                      target_text)]
    # Read full render list, not the sampled twelve images used by the initial audit.
    by_id = {a['id']: a for a in service.store.artifacts(job['case_id'])}
    images, texts = [], []
    for p in prepared:
        full_images = [Path(v) for v in by_id[p.artifact_id]['rendered_pages']]
        body = p.text
        if requested_pages and len(prepared) == 1:
            if full_images and any(n < 1 or n > len(full_images) for n in requested_pages):
                raise StateError('Указанный слайд не найден. Проверьте номер и повторите пояснение.')
            full_images = [full_images[n-1] for n in sorted(set(requested_pages))] if full_images else []
            parts = re.split(r'(?m)(?=^## Слайд \d+)', body)
            if len(parts) > 1:
                body = '\n'.join(part for part in parts if any(part.startswith('## Слайд '+str(n)+'\n') for n in requested_pages))
            elif '\f' in body:
                pages = body.split('\f')
                body = '\n'.join('Страница '+str(n)+'\n'+pages[n-1] for n in requested_pages if n <= len(pages))
        images.extend(full_images)
        texts.append(p.text_reference + '\n' + body)
    raw = '\n\n'.join(texts)
    if len(images) > 12 or len(raw) > 120000:
        result = {'decision': 'clarification', 'replacement': None, 'references': [],
                  'explanation': 'Нужно сузить фрагмент для достоверной проверки.',
                  'clarifying_question': 'Укажите название файла, номера слайдов и нужный фрагмент текста.',
                  'suppressed_question_ids': []}
        return corrections.apply(job, result)
    if any(not path.is_file() for path in images):
        raise StateError('Рендеры удалены. Создайте /new и загрузите материал заново.')
    properties = QUESTION_PROPERTIES if payload['item_id'].startswith('Q') else {
        k: {'type': 'string'} for k in ('title', 'detail', 'material_reference')}
    replacement_schema = _object(properties, list(properties))
    schema = _object({
        'decision': {'type': 'string', 'enum': list(LABELS)},
        'replacement': {'anyOf': [replacement_schema, {'type': 'null'}]},
        'explanation': {'type': 'string'},
        'references': {'type': 'array', 'items': {'type': 'string'}},
        'clarifying_question': {'type': 'string'},
        'suppressed_question_ids': {'type': 'array', 'items': {'type': 'string'}},
    }, ['decision', 'replacement', 'explanation', 'references', 'clarifying_question', 'suppressed_question_ids'])
    evidence = service.store.get_case(job['case_id'])['evidence_pack']
    prompt = (service._base_prompt() + '\nПерепроверь ТОЛЬКО выбранный пункт. Остальной аудит не переписывай. '
              'Пояснение пользователя — гипотеза для проверки, не доказанный факт. '
              'При corrected верни полную замену одного пункта; иначе replacement=null. '
              'Если источники не позволяют решить, верни clarification и конкретный вопрос. '
              'В references укажи конкретные файл/слайд или ссылку evidence. '
              'Не вводи новые факты без источника. Используй то же поколение evidence. '
              'Для принятой правки перечисли ID вопросов Q1,Q2,..., которые повторяют отменённую '
              'или исправленную ошибочную предпосылку и должны быть пропущены в репетиции; '
              'не скрывай независимые вопросы. Исправленный выбранный вопрос не скрывай.\n'
              + _json({'selected_id': payload['item_id'], 'selected': selected,
                       'explanation': payload['explanation'], 'audit': review,
                       'history': corrections.history(audit['id']), 'evidence': evidence})
              + '\nИсходные материалы:\n' + raw)
    result = await service._model(job['case_id'], job['id'], 'review_correction', prompt, schema, images)
    if result.get('decision') == 'corrected' and payload['item_id'].startswith('Q'):
        q = result.get('replacement', {}) or {}
        if (q.get('topic') not in service.profile.topic_ids
                or q.get('severity') not in range(1, 6) or q.get('likelihood') not in range(1, 6)
                or q.get('expected_answer_evidence_level') not in {'ledger', 'material', 'none'}):
            raise ModelError('model_output_invalid_schema', 'Некорректный вопрос')
    return corrections.apply(job, result)
