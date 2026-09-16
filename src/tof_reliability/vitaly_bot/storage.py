"""SQLite state for cases, files, queue jobs, turns, and model diagnostics."""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import InputError, StateError


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_json(value: Optional[str], fallback: Any) -> Any:
    return json.loads(value) if value else fallback


class VitalyStore:
    """Persistent state with a single process writer and restart-safe job state."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.case_root = self.state_dir / "cases"
        self.database_path = self.state_dir / "vitaly_bot.sqlite3"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.case_root.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS cases (
                    id TEXT PRIMARY KEY,
                    chat_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    profile_version TEXT NOT NULL,
                    material_map_json TEXT,
                    review_json TEXT,
                    evidence_pack_json TEXT,
                    evidence_fingerprint TEXT,
                    generation_id TEXT,
                    current_question_json TEXT,
                    planned_question_index INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    last_error_detail TEXT,
                    activity TEXT NOT NULL DEFAULT 'idle',
                    activity_detail TEXT NOT NULL DEFAULT '',
                    activity_started_at TEXT,
                    active_job_id TEXT,
                    progress_message_id INTEGER,
                    stop_requested INTEGER NOT NULL DEFAULT 0,
                    closed_by_stop INTEGER NOT NULL DEFAULT 0,
                    last_user_activity_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_cases_chat ON cases(chat_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
                    telegram_file_id TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    stored_name TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    suffix TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    local_path TEXT NOT NULL,
                    normalized_text_path TEXT,
                    rendered_pages_json TEXT NOT NULL DEFAULT '[]',
                    unit_count INTEGER NOT NULL DEFAULT 0,
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
                    chat_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS turns (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
                    sequence INTEGER NOT NULL,
                    question_json TEXT NOT NULL,
                    answer_text TEXT NOT NULL,
                    feedback_json TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE(case_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS model_runs (
                    id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
                    job_id TEXT REFERENCES jobs(id) ON DELETE SET NULL,
                    kind TEXT NOT NULL,
                    model TEXT NOT NULL,
                    reasoning_effort TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT
                );
                """
            )
            self._migrate_case_columns(connection)

    @staticmethod
    def _migrate_case_columns(connection: sqlite3.Connection) -> None:
        """Keep existing launchd state compatible without a destructive migration."""

        known = {str(row["name"]) for row in connection.execute("PRAGMA table_info(cases)").fetchall()}
        additions = {
            "evidence_pack_json": "TEXT",
            "evidence_fingerprint": "TEXT",
            "generation_id": "TEXT",
            "last_error_detail": "TEXT",
            "activity": "TEXT NOT NULL DEFAULT 'idle'",
            "activity_detail": "TEXT NOT NULL DEFAULT ''",
            "activity_started_at": "TEXT",
            "active_job_id": "TEXT",
            "progress_message_id": "INTEGER",
            "stop_requested": "INTEGER NOT NULL DEFAULT 0",
            "closed_by_stop": "INTEGER NOT NULL DEFAULT 0",
            "last_user_activity_at": "TEXT",
        }
        for name, definition in additions.items():
            if name not in known:
                connection.execute("ALTER TABLE cases ADD COLUMN " + name + " " + definition)

    def create_case(self, chat_id: int, profile_version: str) -> Dict[str, Any]:
        case_id = uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as connection:
            self._insert_case(connection, case_id, chat_id, profile_version, now)
        (self.case_root / case_id / "artifacts").mkdir(parents=True, exist_ok=True)
        return self.get_case(case_id)

    @staticmethod
    def _insert_case(
        connection: sqlite3.Connection, case_id: str, chat_id: int, profile_version: str, now: str
    ) -> None:
        connection.execute(
            """INSERT INTO cases(
                   id, chat_id, status, mode, profile_version, closed_by_stop, last_user_activity_at, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)""",
            (case_id, chat_id, "collecting", "collecting", profile_version, now, now, now),
        )

    def new_case_if_idle(self, chat_id: int, profile_version: str) -> Dict[str, Any]:
        """Create a new case without silently abandoning an active operation."""

        case_id = uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM cases WHERE chat_id = ? ORDER BY created_at DESC LIMIT 1", (chat_id,)
            ).fetchone()
            if current is not None and (current["activity"] != "idle" or current["mode"] in {"grilling", "stopping"}):
                raise self._busy_error(current)
            self._insert_case(connection, case_id, chat_id, profile_version, now)
        (self.case_root / case_id / "artifacts").mkdir(parents=True, exist_ok=True)
        return self.get_case(case_id)

    def get_case(self, case_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        if row is None:
            raise StateError("Кейс не найден")
        return self._case_row(row)

    def active_case(self, chat_id: int, profile_version: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cases WHERE chat_id = ? ORDER BY created_at DESC LIMIT 1", (chat_id,)
            ).fetchone()
        return self._case_row(row) if row is not None else self.create_case(chat_id, profile_version)

    def activate_case_for_user(
        self, chat_id: int, profile_version: str, context_ttl_days: int
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Return a usable case, replacing a stale non-stopped context atomically.

        Resetting creates a new case instead of deleting the old one. The existing
        30-day cleanup policy still controls file deletion, but the previous
        artifacts cannot reach a new review or rehearsal.
        """

        if context_ttl_days < 1:
            raise ValueError("context_ttl_days must be positive")
        now = _utc_now()
        cutoff = datetime.now(timezone.utc) - timedelta(days=context_ttl_days)
        expired_case_id: Optional[str] = None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM cases WHERE chat_id = ? ORDER BY created_at DESC LIMIT 1", (chat_id,)
            ).fetchone()
            if row is None:
                case_id = uuid.uuid4().hex
                self._insert_case(connection, case_id, chat_id, profile_version, now)
            else:
                case_id = str(row["id"])
                last_activity = self._as_utc_datetime(
                    str(row["last_user_activity_at"] or row["updated_at"] or row["created_at"])
                )
                active_job = connection.execute(
                    "SELECT 1 FROM jobs WHERE case_id = ? AND status IN ('queued', 'running') LIMIT 1", (case_id,)
                ).fetchone()
                should_expire = (
                    not int(row["closed_by_stop"] or 0)
                    and active_job is None
                    and last_activity is not None
                    and last_activity < cutoff
                )
                if should_expire:
                    connection.execute(
                        """UPDATE cases
                           SET status = 'expired', mode = 'expired', activity = 'idle',
                               activity_detail = ?,
                               activity_started_at = NULL, active_job_id = NULL, current_question_json = NULL,
                               stop_requested = 0, updated_at = ?
                           WHERE id = ?""",
                        ("Контекст истёк: более " + str(context_ttl_days) + " суток без /stop", now, case_id),
                    )
                    expired_case_id = case_id
                    case_id = uuid.uuid4().hex
                    self._insert_case(connection, case_id, chat_id, profile_version, now)
                else:
                    connection.execute(
                        "UPDATE cases SET last_user_activity_at = ?, updated_at = ? WHERE id = ?",
                        (now, now, case_id),
                    )
        (self.case_root / case_id / "artifacts").mkdir(parents=True, exist_ok=True)
        return self.get_case(case_id), expired_case_id

    @staticmethod
    def _as_utc_datetime(value: str) -> Optional[datetime]:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

    def _case_row(self, row: sqlite3.Row) -> Dict[str, Any]:
        value = dict(row)
        for key, fallback in (
            ("material_map_json", {}),
            ("review_json", {}),
            ("evidence_pack_json", {}),
            ("current_question_json", None),
        ):
            value[key[:-5]] = _load_json(value.pop(key), fallback)
        return value

    def set_activity(self, case_id: str, activity: str, detail: str = "") -> None:
        started_at: Optional[str] = _utc_now() if activity not in {"idle", "awaiting_answer"} else None
        self._update_case(case_id, activity=activity, activity_detail=detail, activity_started_at=started_at)

    def set_progress_message(self, case_id: str, message_id: Optional[int]) -> None:
        self._update_case(case_id, progress_message_id=message_id)

    def active_cases(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM cases WHERE activity NOT IN ('idle', 'awaiting_answer') OR active_job_id IS NOT NULL"
            ).fetchall()
        return [self._case_row(row) for row in rows]

    def _busy_error(self, row: sqlite3.Row) -> StateError:
        detail = str(row["activity_detail"] or "операция ещё не завершена")
        return StateError("Бот занят: " + detail + ". Дождитесь завершения; доступен /status.")

    def require_idle(self, case_id: str) -> None:
        """Give an immediate, human-readable busy response before input validation."""

        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        if row is None:
            raise StateError("Кейс не найден")
        if row["activity"] != "idle":
            raise self._busy_error(row)

    def begin_upload(self, case_id: str) -> None:
        """Atomically reserve a case before Telegram downloads a file."""

        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
            if row is None:
                raise StateError("Кейс не найден")
            if row["activity"] != "idle" or row["mode"] == "grilling":
                raise self._busy_error(row)
            connection.execute(
                "UPDATE cases SET activity = ?, activity_detail = ?, activity_started_at = ?, updated_at = ? WHERE id = ?",
                ("uploading", "Загружаю и разбираю файл", now, now, case_id),
            )

    def finish_upload(self, case_id: str, error: Optional[str] = None) -> None:
        self._update_case(
            case_id,
            activity="idle",
            activity_detail="" if error is None else "Ошибка обработки файла",
            activity_started_at=None,
            last_error=error,
        )

    def case_dir(self, case_id: str) -> Path:
        self.get_case(case_id)
        return self.case_root / case_id

    def artifact_count(self, case_id: str) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM artifacts WHERE case_id = ?", (case_id,)).fetchone()[0])

    def total_units(self, case_id: str) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COALESCE(SUM(unit_count), 0) FROM artifacts WHERE case_id = ?", (case_id,)).fetchone()[0])

    def add_artifact(
        self,
        case_id: str,
        telegram_file_id: str,
        original_name: str,
        stored_name: str,
        sha256: str,
        suffix: str,
        size_bytes: int,
        local_path: Path,
        normalized_text_path: Optional[Path],
        rendered_pages: Sequence[Path],
        unit_count: int,
        warnings: Sequence[str],
    ) -> Dict[str, Any]:
        if self.artifact_count(case_id) >= 10:
            raise InputError("В одном кейсе можно использовать не более 10 файлов")
        if self.total_units(case_id) + unit_count > 200:
            raise InputError("Суммарный лимит кейса — 200 страниц, слайдов или листов")
        artifact_id = uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO artifacts(
                    id, case_id, telegram_file_id, original_name, stored_name, sha256, suffix, size_bytes,
                    local_path, normalized_text_path, rendered_pages_json, unit_count, warnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    artifact_id,
                    case_id,
                    telegram_file_id,
                    original_name,
                    stored_name,
                    sha256,
                    suffix,
                    size_bytes,
                    str(local_path),
                    str(normalized_text_path) if normalized_text_path else None,
                    _json([str(path) for path in rendered_pages]),
                    unit_count,
                    _json(list(warnings)),
                    now,
                ),
            )
            connection.execute(
                """UPDATE cases
                   SET updated_at = ?, status = 'collecting', mode = 'collecting', material_map_json = NULL,
                       review_json = NULL, evidence_pack_json = NULL, evidence_fingerprint = NULL, generation_id = NULL,
                       current_question_json = NULL, planned_question_index = 0, last_error = NULL,
                       last_error_detail = NULL, stop_requested = 0, closed_by_stop = 0
                   WHERE id = ?""",
                (now, case_id),
            )
            connection.execute("DELETE FROM turns WHERE case_id = ?", (case_id,))
        return self.get_artifact(artifact_id)

    def get_artifact(self, artifact_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise StateError("Файл не найден")
        value = dict(row)
        value["rendered_pages"] = _load_json(value.pop("rendered_pages_json"), [])
        value["warnings"] = _load_json(value.pop("warnings_json"), [])
        return value

    def artifacts(self, case_id: str) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM artifacts WHERE case_id = ? ORDER BY created_at", (case_id,)).fetchall()
        return [self.get_artifact(row["id"]) for row in rows]

    def set_case_material_map(self, case_id: str, material_map: Dict[str, Any]) -> None:
        self._update_case(case_id, material_map_json=_json(material_map))

    def set_evidence_pack(self, case_id: str, pack: Dict[str, Any], fingerprint: str) -> None:
        generation = pack.get("generation", {}) if isinstance(pack, dict) else {}
        self._update_case(
            case_id,
            evidence_pack_json=_json(pack),
            evidence_fingerprint=fingerprint,
            generation_id=str(generation.get("run_id") or ""),
        )

    def set_review(self, case_id: str, review: Dict[str, Any]) -> None:
        self._update_case(case_id, review_json=_json(review), status="reviewed", mode="ready")

    def set_question(self, case_id: str, question: Optional[Dict[str, Any]], index: int, mode: str) -> None:
        self._update_case(
            case_id,
            current_question_json=_json(question) if question is not None else None,
            planned_question_index=index,
            mode=mode,
            activity="awaiting_answer" if question is not None and mode == "grilling" else "idle",
            activity_detail="" if question is not None else "",
            activity_started_at=None,
            active_job_id=None,
        )

    def set_error(self, case_id: str, error: Optional[str], detail: str = "") -> None:
        self._update_case(case_id, last_error=error, last_error_detail=detail[-4000:])

    def set_stop_requested(self, case_id: str, value: bool) -> None:
        self._update_case(case_id, stop_requested=1 if value else 0)

    def _update_case(self, case_id: str, **changes: Any) -> None:
        if not changes:
            return
        changes["updated_at"] = _utc_now()
        columns = ", ".join(f"{column} = ?" for column in changes)
        with self._connect() as connection:
            result = connection.execute(
                f"UPDATE cases SET {columns} WHERE id = ?", tuple(changes.values()) + (case_id,)
            )
        if result.rowcount != 1:
            raise StateError("Кейс не найден")

    def add_turn(self, case_id: str, question: Dict[str, Any], answer_text: str) -> Dict[str, Any]:
        with self._connect() as connection:
            sequence = int(
                connection.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM turns WHERE case_id = ?", (case_id,)).fetchone()[0]
            )
            turn_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO turns(id, case_id, sequence, question_json, answer_text, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (turn_id, case_id, sequence, _json(question), answer_text, _utc_now()),
            )
        return self.get_turn(turn_id)

    def accept_answer_and_enqueue(self, case_id: str, chat_id: int, answer_text: str) -> Dict[str, Any]:
        """Store exactly one answer and its evaluation job in one transaction."""

        answer = answer_text.strip()
        if not answer:
            raise StateError("Нужен текстовый ответ на вопрос")
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
            if row is None:
                raise StateError("Кейс не найден")
            question = _load_json(row["current_question_json"], None)
            if row["mode"] != "grilling" or row["activity"] != "awaiting_answer" or not question:
                raise self._busy_error(row)
            sequence = int(
                connection.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM turns WHERE case_id = ?", (case_id,)).fetchone()[0]
            )
            turn_id = uuid.uuid4().hex
            job_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO turns(id, case_id, sequence, question_json, answer_text, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (turn_id, case_id, sequence, _json(question), answer, now),
            )
            connection.execute(
                "INSERT INTO jobs(id, case_id, chat_id, kind, payload_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, case_id, chat_id, "grill_answer", _json({"turn_id": turn_id}), "queued", now),
            )
            connection.execute(
                "UPDATE cases SET activity = ?, activity_detail = ?, activity_started_at = ?, active_job_id = ?, updated_at = ? WHERE id = ?",
                ("queued", "Ответ принят; ожидаю модель", now, job_id, now, case_id),
            )
        return self.get_job(job_id)

    def get_turn(self, turn_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()
        if row is None:
            raise StateError("Ответ не найден")
        value = dict(row)
        value["question"] = _load_json(value.pop("question_json"), {})
        value["feedback"] = _load_json(value.pop("feedback_json"), {})
        return value

    def turns(self, case_id: str) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id FROM turns WHERE case_id = ? ORDER BY sequence", (case_id,)).fetchall()
        return [self.get_turn(row["id"]) for row in rows]

    def set_turn_feedback(self, turn_id: str, feedback: Dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE turns SET feedback_json = ? WHERE id = ?", (_json(feedback), turn_id))

    def clear_turns(self, case_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM turns WHERE case_id = ?", (case_id,))

    def enqueue_job(self, case_id: str, chat_id: int, kind: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO jobs(id, case_id, chat_id, kind, payload_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, case_id, chat_id, kind, _json(payload or {}), "queued", now),
            )
        return self.get_job(job_id)

    def begin_job(self, case_id: str, chat_id: int, kind: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create the only active case job, rejecting duplicate commands atomically."""

        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
            if row is None:
                raise StateError("Кейс не найден")
            if row["activity"] != "idle":
                raise self._busy_error(row)
            job_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO jobs(id, case_id, chat_id, kind, payload_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, case_id, chat_id, kind, _json(payload or {}), "queued", now),
            )
            connection.execute(
                "UPDATE cases SET activity = ?, activity_detail = ?, activity_started_at = ?, active_job_id = ?, updated_at = ? WHERE id = ?",
                ("queued", "Ожидаю общей очереди", now, job_id, now, case_id),
            )
        return self.get_job(job_id)

    def queue_finish(self, case_id: str, chat_id: int, *, detail: str = "Готовлю итог репетиции") -> Optional[Dict[str, Any]]:
        """Schedule one final scorecard after stopping or the final question."""

        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
            if row is None:
                raise StateError("Кейс не найден")
            if row["active_job_id"]:
                active = connection.execute(
                    "SELECT status FROM jobs WHERE id = ?", (row["active_job_id"],)
                ).fetchone()
                if active is not None and active["status"] in {"queued", "running"}:
                    return None
            count = int(connection.execute("SELECT COUNT(*) FROM turns WHERE case_id = ?", (case_id,)).fetchone()[0])
            if not count:
                connection.execute(
                    "UPDATE cases SET mode = ?, activity = ?, activity_detail = ?, activity_started_at = NULL, stop_requested = 0, closed_by_stop = ?, updated_at = ? WHERE id = ?",
                    (
                        "stopped",
                        "idle",
                        "",
                        1 if int(row["stop_requested"] or 0) else int(row["closed_by_stop"] or 0),
                        now,
                        case_id,
                    ),
                )
                return None
            job_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO jobs(id, case_id, chat_id, kind, payload_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, case_id, chat_id, "finish", "{}", "queued", now),
            )
            connection.execute(
                "UPDATE cases SET mode = ?, activity = ?, activity_detail = ?, activity_started_at = ?, active_job_id = ?, updated_at = ? WHERE id = ?",
                ("stopping", "summarizing", detail, now, job_id, now, case_id),
            )
        return self.get_job(job_id)

    def request_stop(self, case_id: str, chat_id: int) -> Optional[Dict[str, Any]]:
        """Immediately suppress future questions and queue one final result if possible."""

        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
            if row is None:
                raise StateError("Кейс не найден")
            active = connection.execute(
                "SELECT * FROM jobs WHERE case_id = ? AND status IN ('queued', 'running') ORDER BY created_at LIMIT 1", (case_id,)
            ).fetchone()
            grill_active = row["mode"] in {"grilling", "stopping"} or (active is not None and active["kind"] in {"grill_start", "grill_answer", "finish"})
            if not grill_active:
                raise StateError("Репетиция ещё не запущена")
            if int(row["stop_requested"] or 0):
                raise StateError("Остановка уже запрошена; повторная команда не добавлена")
            connection.execute(
                "UPDATE cases SET mode = ?, stop_requested = 1, closed_by_stop = 1, activity = ?, activity_detail = ?, updated_at = ? WHERE id = ?",
                ("stopping", "stopping", "Останавливаю репетицию", now, case_id),
            )
            if active is not None:
                return None
            count = int(connection.execute("SELECT COUNT(*) FROM turns WHERE case_id = ?", (case_id,)).fetchone()[0])
            if not count:
                connection.execute(
                    "UPDATE cases SET mode = ?, activity = ?, activity_detail = ?, activity_started_at = NULL, stop_requested = 0, closed_by_stop = 1, updated_at = ? WHERE id = ?",
                    ("stopped", "idle", "", now, case_id),
                )
                return None
            job_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO jobs(id, case_id, chat_id, kind, payload_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (job_id, case_id, chat_id, "finish", "{}", "queued", now),
            )
            connection.execute(
                "UPDATE cases SET activity = ?, activity_detail = ?, activity_started_at = ?, active_job_id = ?, updated_at = ? WHERE id = ?",
                ("summarizing", "Готовлю итог репетиции", now, job_id, now, case_id),
            )
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise StateError("Задача не найдена")
        value = dict(row)
        value["payload"] = _load_json(value.pop("payload_json"), {})
        return value

    def next_queued_job(self) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
        return self.get_job(row["id"]) if row else None

    def mark_job_running(self, job_id: str) -> None:
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE jobs SET status = 'running', attempts = attempts + 1, started_at = ? WHERE id = ? AND status = 'queued'",
                (_utc_now(), job_id),
            )
        if result.rowcount != 1:
            raise StateError("Задача уже не ожидает запуска")
        job = self.get_job(job_id)
        self._update_case(
            job["case_id"],
            activity="thinking",
            activity_detail="Выполняю задачу",
            activity_started_at=_utc_now(),
            active_job_id=job_id,
        )

    def finish_job(self, job_id: str, error: Optional[str] = None) -> None:
        status = "failed" if error else "completed"
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, error, _utc_now(), job_id),
            )

    def settle_job(self, case_id: str, *, activity: str = "idle", detail: str = "", mode: Optional[str] = None) -> None:
        changes: Dict[str, Any] = {
            "active_job_id": None,
            "activity": activity,
            "activity_detail": detail,
            "activity_started_at": None,
        }
        if mode is not None:
            changes["mode"] = mode
        self._update_case(case_id, **changes)

    def recover_jobs(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            connection.execute("UPDATE jobs SET status = 'queued', started_at = NULL WHERE status = 'running'")
            connection.execute(
                """UPDATE cases
                   SET activity = 'queued', activity_detail = 'Восстанавливаю задачу после перезапуска',
                       activity_started_at = ?, active_job_id = (
                         SELECT id FROM jobs WHERE jobs.case_id = cases.id AND status = 'queued'
                         ORDER BY created_at, id LIMIT 1
                       )
                   WHERE id IN (SELECT case_id FROM jobs WHERE status = 'queued')""",
                (_utc_now(),),
            )
            rows = connection.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at, id").fetchall()
        return [self.get_job(row["id"]) for row in rows]

    def queue_position(self, job_id: str) -> int:
        job = self.get_job(job_id)
        if job["status"] != "queued":
            return 0
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE status = 'queued' AND (created_at < ? OR (created_at = ? AND id <= ?))",
                    (job["created_at"], job["created_at"], job_id),
                ).fetchone()[0]
            )

    def queue_summary(self) -> Dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute("SELECT status, COUNT(*) AS count FROM jobs GROUP BY status").fetchall()
        return {row["status"]: int(row["count"]) for row in rows}

    def create_model_run(self, case_id: str, job_id: Optional[str], kind: str, model: str, reasoning_effort: str) -> str:
        run_id = uuid.uuid4().hex
        with self._connect() as connection:
            if job_id is not None:
                known_job = connection.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if known_job is None:
                    job_id = None
            connection.execute(
                "INSERT INTO model_runs(id, case_id, job_id, kind, model, reasoning_effort, status, started_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, case_id, job_id, kind, model, reasoning_effort, "running", _utc_now()),
            )
        return run_id

    def finish_model_run(self, run_id: str, error: Optional[str] = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE model_runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                ("failed" if error else "completed", error, _utc_now(), run_id),
            )

    def cleanup_expired_files(self, days: int) -> List[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        removed: List[str] = []
        with self._connect() as connection:
            rows = connection.execute("SELECT id, created_at FROM cases").fetchall()
        for row in rows:
            created = datetime.fromisoformat(row["created_at"])
            case_dir = self.case_root / row["id"]
            if created < cutoff and case_dir.exists():
                shutil.rmtree(case_dir)
                removed.append(row["id"])
        return removed
