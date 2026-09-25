"""SQLite state for cases, files, queue jobs, turns, and model diagnostics."""

from __future__ import annotations

import json
import hashlib
import shutil
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import InputError, StateError


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_now(now: Optional[datetime] = None) -> datetime:
    """Return the Mac-local clock, while keeping injected test zones stable."""

    if now is None:
        return datetime.now().astimezone()
    return now.astimezone() if now.tzinfo is None else now


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_json(value: Optional[str], fallback: Any) -> Any:
    return json.loads(value) if value else fallback


def _duration_ms(started_at: Optional[str]) -> Optional[int]:
    if not started_at:
        return None
    try:
        started = datetime.fromisoformat(started_at)
    except ValueError:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return max(0, int((datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds() * 1000))


class VitalyStore:
    """Persistent state with a single process writer and restart-safe job state."""

    def __init__(self, state_dir: Path, knowledge_dir: Optional[Path] = None) -> None:
        self.state_dir = state_dir
        self.case_root = self.state_dir / "cases"
        self.knowledge_root = knowledge_dir or (self.state_dir / "knowledge")
        self.database_path = self.state_dir / "vitaly_bot.sqlite3"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.case_root.mkdir(parents=True, exist_ok=True)
        self.knowledge_root.mkdir(parents=True, exist_ok=True)
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
                    source_locations_json TEXT NOT NULL DEFAULT '[]',
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
            self._initialize_learning_schema(connection)

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
            "knowledge_package_id": "TEXT",
        }
        for name, definition in additions.items():
            if name not in known:
                connection.execute("ALTER TABLE cases ADD COLUMN " + name + " " + definition)
        artifact_columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(artifacts)").fetchall()}
        if "source_locations_json" not in artifact_columns:
            connection.execute("ALTER TABLE artifacts ADD COLUMN source_locations_json TEXT NOT NULL DEFAULT '[]'")

    @staticmethod
    def _initialize_learning_schema(connection: sqlite3.Connection) -> None:
        """Add-only storage for the independent learning domain.

        Learning work deliberately does not borrow a user case: doing so would
        mutate its activity state and could accidentally put a transcript into a
        user's review.  The worker still consumes both domains serially.
        """

        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_packages (
                id TEXT PRIMARY KEY,
                version TEXT NOT NULL UNIQUE,
                parent_id TEXT,
                package_json TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                verification_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                enabled_at TEXT
            );
            CREATE TABLE IF NOT EXISTS learning_control (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                active_package_id TEXT,
                auto_update_paused INTEGER NOT NULL DEFAULT 0,
                last_cycle_at TEXT,
                updated_at TEXT NOT NULL
            );
            INSERT OR IGNORE INTO learning_control(id, active_package_id, auto_update_paused, updated_at)
            VALUES (1, NULL, 0, CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS learning_jobs (
                id TEXT PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                user_id INTEGER,
                kind TEXT NOT NULL,
                submission_id TEXT,
                payload_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_learning_jobs_status ON learning_jobs(status, created_at);
            CREATE TABLE IF NOT EXISTS learning_model_runs (
                id TEXT PRIMARY KEY,
                learning_job_id TEXT REFERENCES learning_jobs(id) ON DELETE SET NULL,
                kind TEXT NOT NULL,
                model TEXT NOT NULL,
                reasoning_effort TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );
            CREATE TABLE IF NOT EXISTS feedback_submissions (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
                audit_id TEXT,
                chat_id INTEGER NOT NULL,
                user_id INTEGER,
                status TEXT NOT NULL,
                text TEXT,
                text_sha256 TEXT,
                result_json TEXT NOT NULL DEFAULT '{}',
                learning_job_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_feedback_pending ON feedback_submissions(chat_id, status, created_at);
            CREATE TABLE IF NOT EXISTS meeting_submissions (
                id TEXT PRIMARY KEY,
                chat_id INTEGER NOT NULL,
                user_id INTEGER,
                status TEXT NOT NULL,
                step TEXT NOT NULL,
                meeting_date TEXT,
                title TEXT,
                vitaly_designation TEXT,
                source_name TEXT,
                source_path TEXT,
                normalized_path TEXT,
                source_sha256 TEXT,
                normalized_sha256 TEXT,
                classification_json TEXT NOT NULL DEFAULT '{}',
                duplicate_of TEXT,
                continuation_of TEXT,
                learning_job_id TEXT,
                last_metadata_step TEXT,
                last_metadata_sha256 TEXT,
                last_metadata_message_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_meetings_user ON meeting_submissions(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_meetings_hash ON meeting_submissions(source_sha256, normalized_sha256);
            CREATE TABLE IF NOT EXISTS learning_candidates (
                id TEXT PRIMARY KEY,
                source_ids_json TEXT NOT NULL,
                patch_json TEXT NOT NULL,
                baseline_package_id TEXT,
                status TEXT NOT NULL,
                scope TEXT NOT NULL,
                expected_improvement TEXT NOT NULL DEFAULT '',
                generated_at TEXT NOT NULL,
                verified_at TEXT,
                verification_json TEXT NOT NULL DEFAULT '{}',
                package_id TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS learning_regression_runs (
                id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL REFERENCES learning_candidates(id) ON DELETE CASCADE,
                baseline_package_id TEXT NOT NULL,
                result_json TEXT NOT NULL,
                passing INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS learning_events (
                id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                object_type TEXT NOT NULL,
                object_id TEXT NOT NULL,
                chat_id INTEGER,
                user_id INTEGER,
                status TEXT NOT NULL,
                duration_ms INTEGER,
                error_code TEXT,
                detail_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_learning_events_object ON learning_events(object_type, object_id, created_at);
            """
        )
        candidate_columns = {
            str(row["name"]) for row in connection.execute("PRAGMA table_info(learning_candidates)").fetchall()
        }
        if "baseline_package_id" not in candidate_columns:
            connection.execute("ALTER TABLE learning_candidates ADD COLUMN baseline_package_id TEXT")
        meeting_columns = {
            str(row["name"]) for row in connection.execute("PRAGMA table_info(meeting_submissions)").fetchall()
        }
        for name, definition in {
            "last_metadata_step": "TEXT",
            "last_metadata_sha256": "TEXT",
            "last_metadata_message_id": "INTEGER",
        }.items():
            if name not in meeting_columns:
                connection.execute("ALTER TABLE meeting_submissions ADD COLUMN " + name + " " + definition)

    def create_case(
        self, chat_id: int, profile_version: str, knowledge_package_id: Optional[str] = None
    ) -> Dict[str, Any]:
        case_id = uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as connection:
            self._insert_case(connection, case_id, chat_id, profile_version, now, knowledge_package_id)
        (self.case_root / case_id / "artifacts").mkdir(parents=True, exist_ok=True)
        case = self.get_case(case_id)
        self.record_learning_event(
            "case_created", "case", case_id, status="collecting", chat_id=chat_id,
            detail={"profile_version": profile_version, "knowledge_package_id": knowledge_package_id or ""},
        )
        return case

    @staticmethod
    def _insert_case(
        connection: sqlite3.Connection,
        case_id: str,
        chat_id: int,
        profile_version: str,
        now: str,
        knowledge_package_id: Optional[str] = None,
    ) -> None:
        connection.execute(
            """INSERT INTO cases(
                   id, chat_id, status, mode, profile_version, knowledge_package_id, closed_by_stop, last_user_activity_at, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?)""",
            (case_id, chat_id, "collecting", "collecting", profile_version, knowledge_package_id, now, now, now),
        )

    def new_case_if_idle(
        self, chat_id: int, profile_version: str, knowledge_package_id: Optional[str] = None
    ) -> Dict[str, Any]:
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
            self._insert_case(connection, case_id, chat_id, profile_version, now, knowledge_package_id)
        (self.case_root / case_id / "artifacts").mkdir(parents=True, exist_ok=True)
        case = self.get_case(case_id)
        self.record_learning_event(
            "case_created", "case", case_id, status="collecting", chat_id=chat_id,
            detail={"profile_version": profile_version, "knowledge_package_id": knowledge_package_id or ""},
        )
        return case

    def get_case(self, case_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
        if row is None:
            raise StateError("Кейс не найден")
        return self._case_row(row)

    def active_case(
        self, chat_id: int, profile_version: str, knowledge_package_id: Optional[str] = None
    ) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM cases WHERE chat_id = ? ORDER BY created_at DESC LIMIT 1", (chat_id,)
            ).fetchone()
        return self._case_row(row) if row is not None else self.create_case(chat_id, profile_version, knowledge_package_id)

    def activate_case_for_user(
        self,
        chat_id: int,
        profile_version: str,
        context_ttl_days: int,
        knowledge_package_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Return a usable case, replacing a stale non-stopped context atomically.

        Resetting creates a new case instead of deleting the old one.  The old
        evidence remains in the local archive, but cannot reach a new review or
        rehearsal.
        """

        if context_ttl_days < 1:
            raise ValueError("context_ttl_days must be positive")
        now = _utc_now()
        cutoff = datetime.now(timezone.utc) - timedelta(days=context_ttl_days)
        expired_case_id: Optional[str] = None
        created_case = False
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM cases WHERE chat_id = ? ORDER BY created_at DESC LIMIT 1", (chat_id,)
            ).fetchone()
            if row is None:
                case_id = uuid.uuid4().hex
                self._insert_case(connection, case_id, chat_id, profile_version, now, knowledge_package_id)
                created_case = True
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
                    self._insert_case(connection, case_id, chat_id, profile_version, now, knowledge_package_id)
                    created_case = True
                else:
                    connection.execute(
                        "UPDATE cases SET last_user_activity_at = ?, updated_at = ? WHERE id = ?",
                        (now, now, case_id),
                    )
        (self.case_root / case_id / "artifacts").mkdir(parents=True, exist_ok=True)
        case = self.get_case(case_id)
        if created_case:
            self.record_learning_event(
                "case_created", "case", case_id, status="collecting", chat_id=chat_id,
                detail={
                    "profile_version": profile_version,
                    "knowledge_package_id": knowledge_package_id or "",
                    "replaced_expired_case_id": expired_case_id or "",
                },
            )
        return case, expired_case_id

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
        case = self.get_case(case_id)
        self.record_learning_event("upload_started", "case", case_id, status="uploading", chat_id=int(case["chat_id"]))

    def finish_upload(self, case_id: str, error: Optional[str] = None) -> None:
        case = self.get_case(case_id)
        duration_ms = _duration_ms(case.get("activity_started_at"))
        self._update_case(
            case_id,
            activity="idle",
            activity_detail="" if error is None else "Ошибка обработки файла",
            activity_started_at=None,
            last_error=error,
        )
        self.record_learning_event(
            "upload_finished", "case", case_id, status="failed" if error else "completed", chat_id=int(case["chat_id"]),
            duration_ms=duration_ms, error_code=error,
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
        source_locations: Optional[Sequence[Dict[str, Any]]] = None,
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
                    local_path, normalized_text_path, rendered_pages_json, source_locations_json, unit_count, warnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    _json(list(source_locations or [])),
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
        artifact = self.get_artifact(artifact_id)
        case = self.get_case(case_id)
        self.record_learning_event(
            "artifact_uploaded", "artifact", artifact_id, status="stored", chat_id=int(case["chat_id"]),
            detail={"sha256": artifact["sha256"], "size_bytes": artifact["size_bytes"]},
        )
        return artifact

    def get_artifact(self, artifact_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM artifacts WHERE id = ?", (artifact_id,)).fetchone()
        if row is None:
            raise StateError("Файл не найден")
        value = dict(row)
        value["rendered_pages"] = _load_json(value.pop("rendered_pages_json"), [])
        value["source_locations"] = _load_json(value.pop("source_locations_json", "[]"), [])
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
        case = self.get_case(case_id)
        self.record_learning_event(
            "ledger_checked", "case", case_id, status="checked", chat_id=int(case["chat_id"]),
            detail={"generation_id": str(generation.get("run_id") or "")},
        )

    def set_review(self, case_id: str, review: Dict[str, Any]) -> None:
        before = self.get_case(case_id)
        self._update_case(case_id, review_json=_json(review), status="reviewed", mode="ready")
        case = self.get_case(case_id)
        self.record_learning_event(
            "review_completed", "case", case_id, status="reviewed", chat_id=int(case["chat_id"]),
            duration_ms=_duration_ms(before.get("activity_started_at")),
            detail={"knowledge_package_id": case.get("knowledge_package_id") or ""},
        )

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
        if question is not None and mode == "grilling" and index == 0:
            case = self.get_case(case_id)
            self.record_learning_event("grill_started", "case", case_id, status="grilling", chat_id=int(case["chat_id"]))

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
        turn = self.get_turn(turn_id)
        case = self.get_case(str(turn["case_id"]))
        self.record_learning_event(
            "turn_evaluated", "turn", turn_id, status=str(feedback.get("answer_status") or "evaluated"), chat_id=int(case["chat_id"]),
            duration_ms=_duration_ms(case.get("activity_started_at")),
        )

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

    # --- One serial worker, with user jobs always selected before learning ---

    @staticmethod
    def _insert_learning_job(
        connection: sqlite3.Connection,
        *,
        chat_id: int,
        user_id: Optional[int],
        kind: str,
        submission_id: Optional[str],
        payload: Dict[str, Any],
        now: str,
        status: str = "queued",
    ) -> str:
        if status not in {"queued", "paused", "scheduled"}:
            raise ValueError("Недопустимое состояние новой фоновой задачи")
        job_id = uuid.uuid4().hex
        connection.execute(
            """INSERT INTO learning_jobs(
                   id, chat_id, user_id, kind, submission_id, payload_json, status, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, chat_id, user_id, kind, submission_id, _json(payload), status, now),
        )
        return job_id

    @staticmethod
    def _get_learning_job(connection: sqlite3.Connection, job_id: str) -> Dict[str, Any]:
        row = connection.execute("SELECT * FROM learning_jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise StateError("Фоновая задача не найдена")
        value = VitalyStore._learning_row(row)
        value["scope"] = "learning"
        return value

    def enqueue_learning_job(
        self,
        *,
        kind: str,
        chat_id: int = 0,
        user_id: Optional[int] = None,
        submission_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            control = connection.execute("SELECT auto_update_paused FROM learning_control WHERE id = 1").fetchone()
            automatic = kind in {"knowledge_generate", "knowledge_verify"}
            status = "paused" if automatic and control is not None and int(control["auto_update_paused"] or 0) else "queued"
            job_id = self._insert_learning_job(
                connection,
                chat_id=chat_id,
                user_id=user_id,
                kind=kind,
                submission_id=submission_id,
                payload=payload or {},
                now=now,
                status=status,
            )
        return self.get_learning_job(job_id)

    def get_learning_job(self, job_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            return self._get_learning_job(connection, job_id)

    def active_learning_jobs(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM learning_jobs WHERE status IN ('queued', 'running') ORDER BY created_at, id"
            ).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            value = self._learning_row(row)
            value["scope"] = "learning"
            result.append(value)
        return result

    def next_queued_work(
        self, *, automatic_run_hour: Optional[int] = None, now: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the next model invocation without letting background work delay a person.

        Once a model run has started it is never cancelled, but after every run
        this selector prefers all ordinary case work before the next learning
        step.  That is the concrete priority boundary promised to users.
        """

        with self._connect() as connection:
            if automatic_run_hour is not None and (
                not 0 <= automatic_run_hour <= 23 or _local_now(now).hour != automatic_run_hour
            ):
                # A generator or verifier may lose the worker race to a
                # person near the end of the 04:00 window.  Keep that next
                # model step for tomorrow rather than starting it in daytime.
                connection.execute(
                    """UPDATE learning_jobs SET status = 'scheduled'
                       WHERE status = 'queued' AND kind IN ('knowledge_generate', 'knowledge_verify')"""
                )
            row = connection.execute(
                "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if row:
                job = self.get_job(str(row["id"]))
                job["scope"] = "case"
                return job
            row = connection.execute(
                """SELECT id FROM learning_jobs
                   WHERE status = 'queued' AND kind IN ('feedback_analyze', 'meeting_analyze')
                   ORDER BY created_at, id LIMIT 1"""
            ).fetchone()
            if row:
                return self._get_learning_job(connection, str(row["id"]))
            row = connection.execute(
                "SELECT id FROM learning_jobs WHERE status = 'queued' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if row:
                return self._get_learning_job(connection, str(row["id"]))
        return None

    def next_queued_job(self) -> Optional[Dict[str, Any]]:
        """Compatibility view of the next ordinary user job."""
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

    def mark_learning_job_running(
        self, job_id: str, *, automatic_run_hour: Optional[int] = None, now: Optional[datetime] = None
    ) -> bool:
        with self._connect() as connection:
            # Take the write lock before reading the clock.  Otherwise a
            # 04:59 check can wait behind another writer and admit a model
            # invocation after 05:00.
            connection.execute("BEGIN IMMEDIATE")
            job = connection.execute("SELECT kind, status FROM learning_jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None or job["status"] != "queued":
                raise StateError("Фоновая задача уже не ожидает запуска")
            if (
                job["kind"] in {"knowledge_generate", "knowledge_verify"}
                and automatic_run_hour is not None
                and (
                    not 0 <= automatic_run_hour <= 23
                    or _local_now(now).hour != automatic_run_hour
                )
            ):
                # Check the schedule again at admission time: a job can be
                # selected at 04:59:59 and reach this write after 05:00.
                connection.execute(
                    "UPDATE learning_jobs SET status = 'scheduled' WHERE id = ? AND status = 'queued'",
                    (job_id,),
                )
                return False
            result = connection.execute(
                "UPDATE learning_jobs SET status = 'running', attempts = attempts + 1, started_at = ? WHERE id = ? AND status = 'queued'",
                (_utc_now(), job_id),
            )
        if result.rowcount != 1:
            raise StateError("Фоновая задача уже не ожидает запуска")
        return True

    def mark_work_running(self, job: Dict[str, Any], *, automatic_run_hour: Optional[int] = None) -> bool:
        if job.get("scope") == "learning":
            return self.mark_learning_job_running(str(job["id"]), automatic_run_hour=automatic_run_hour)
        else:
            self.mark_job_running(str(job["id"]))
            return True

    def finish_job(self, job_id: str, error: Optional[str] = None) -> None:
        status = "failed" if error else "completed"
        with self._connect() as connection:
            connection.execute(
                "UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                (status, error, _utc_now(), job_id),
            )

    def finish_learning_job(self, job_id: str, error: Optional[str] = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE learning_jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                ("failed" if error else "completed", error, _utc_now(), job_id),
            )

    def finish_work(self, job: Dict[str, Any], error: Optional[str] = None) -> None:
        if job.get("scope") == "learning":
            self.finish_learning_job(str(job["id"]), error)
        else:
            self.finish_job(str(job["id"]), error)

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
        if mode == "finished":
            case = self.get_case(case_id)
            self.record_learning_event("grill_finished", "case", case_id, status="finished", chat_id=int(case["chat_id"]))

    def recover_jobs(
        self, *, automatic_run_hour: Optional[int] = None, now: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Recover interrupted work without turning a 04:00 task into daytime work.

        ``automatic_run_hour`` is supplied only by the production worker.  The
        optional argument preserves the generic storage-recovery API for tools
        and migration tests that do not own a scheduler.
        """

        local_now = _local_now(now)
        automatic_in_window = (
            automatic_run_hour is None
            or (0 <= automatic_run_hour <= 23 and local_now.hour == automatic_run_hour)
        )
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
            # A pause is an operator decision, not merely a UI toggle.  A
            # model run already in flight is allowed to finish, but if the
            # process dies before that happens its automatic follow-up must
            # not spring back to life on the next launch.
            control = connection.execute(
                "SELECT auto_update_paused FROM learning_control WHERE id = 1"
            ).fetchone()
            learning_status = (
                "paused"
                if control is not None and int(control["auto_update_paused"] or 0)
                else ("queued" if automatic_in_window else "scheduled")
            )
            connection.execute(
                "UPDATE learning_jobs SET status = ?, started_at = NULL WHERE status = 'running' AND kind IN ('knowledge_generate', 'knowledge_verify')",
                (learning_status,),
            )
            if learning_status == "queued":
                connection.execute(
                    """UPDATE learning_jobs SET status = 'queued', started_at = NULL
                       WHERE status = 'scheduled' AND kind IN ('knowledge_generate', 'knowledge_verify')"""
                )
            elif learning_status == "scheduled":
                connection.execute(
                    """UPDATE learning_jobs SET status = 'scheduled', started_at = NULL
                       WHERE status = 'queued' AND kind IN ('knowledge_generate', 'knowledge_verify')"""
                )
            connection.execute(
                "UPDATE learning_jobs SET status = 'queued', started_at = NULL WHERE status = 'running' AND kind NOT IN ('knowledge_generate', 'knowledge_verify')"
            )
            # Older releases persisted candidate creation and verifier queueing
            # in separate transactions.  Repair an interrupted legacy write
            # before the worker makes any selection, preserving the selected
            # sources instead of allowing them to be consumed forever.  A
            # failed verifier is a terminal result, however: never spend a
            # second model call retrying a recorded permanent failure.
            verification_candidates = connection.execute(
                """SELECT id, source_ids_json, verification_json
                   FROM learning_candidates WHERE status = 'candidate' ORDER BY generated_at, id"""
            ).fetchall()
            verification_jobs = connection.execute(
                """SELECT payload_json, status, error, created_at, finished_at, id FROM learning_jobs
                   WHERE kind = 'knowledge_verify'
                   ORDER BY COALESCE(finished_at, created_at), id"""
            ).fetchall()
            verifications_by_candidate: Dict[str, List[sqlite3.Row]] = {}
            for verification_job in verification_jobs:
                candidate_id = str(_load_json(verification_job["payload_json"], {}).get("candidate_id") or "")
                if candidate_id:
                    verifications_by_candidate.setdefault(candidate_id, []).append(verification_job)
            restored_candidate_ids: List[str] = []
            terminalized_candidates: List[Tuple[str, str, str]] = []
            recovery_now = _utc_now()
            for candidate in verification_candidates:
                candidate_id = str(candidate["id"])
                candidate_jobs = verifications_by_candidate.get(candidate_id, [])
                if any(job["status"] in {"scheduled", "queued", "running", "paused"} for job in candidate_jobs):
                    continue
                if not candidate_jobs:
                    restored_candidate_ids.append(candidate_id)
                    continue
                failed_job = next((job for job in reversed(candidate_jobs) if job["status"] == "failed"), None)
                if failed_job is not None:
                    candidate_status = "failed"
                    error = str(failed_job["error"] or "verification_failed")
                    reason = "Проверка кандидата ранее завершилась с ошибкой; прежняя версия знаний сохранена."
                    source_outcome = "verification_failed"
                else:
                    # A terminal verifier without a saved candidate result is
                    # an interrupted legacy operation.  Do not infer success
                    # or replay it: leave the package unchanged and surface a
                    # recoverable clarification instead.
                    candidate_status = "needs_clarification"
                    error = "verification_result_missing"
                    reason = "Проверка кандидата завершилась без сохранённого решения; автоматическое включение остановлено."
                    source_outcome = "needs_clarification"
                verification = _load_json(candidate["verification_json"], {})
                verification.update({"stage": "verification", "reason": reason, "error_code": error})
                connection.execute(
                    """UPDATE learning_candidates
                       SET status = ?, verification_json = ?, verified_at = ?, error = ?
                       WHERE id = ? AND status = 'candidate'""",
                    (candidate_status, _json(verification), recovery_now, error, candidate_id),
                )
                source_ids = [str(value) for value in _load_json(candidate["source_ids_json"], [])]
                self._annotate_learning_sources_in_connection(
                    connection, source_ids, source_outcome, None, recovery_now
                )
                terminalized_candidates.append((candidate_id, candidate_status, error))
            for candidate_id in restored_candidate_ids:
                self._insert_learning_job(
                    connection,
                    chat_id=0,
                    user_id=None,
                    kind="knowledge_verify",
                    submission_id=None,
                    payload={"candidate_id": candidate_id},
                    now=_utc_now(),
                    status=learning_status,
                )
            rows = connection.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at, id").fetchall()
            learning_rows = connection.execute("SELECT id FROM learning_jobs WHERE status = 'queued' ORDER BY created_at, id").fetchall()
            result = [self.get_job(row["id"]) for row in rows]
            result.extend(self._get_learning_job(connection, row["id"]) for row in learning_rows)
        for candidate_id in restored_candidate_ids:
            self.record_learning_event(
                "candidate_verification_recovered", "candidate", candidate_id, status=learning_status
            )
        for candidate_id, status, error in terminalized_candidates:
            self.record_learning_event(
                "candidate_" + status, "candidate", candidate_id, status=status, error_code=error
            )
        return result

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
            learning_rows = connection.execute("SELECT status, COUNT(*) AS count FROM learning_jobs GROUP BY status").fetchall()
        result = {row["status"]: int(row["count"]) for row in rows}
        result.update({"learning_" + str(row["status"]): int(row["count"]) for row in learning_rows})
        return result

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

    # --- Immutable knowledge packages and permanent learning archive ---

    @staticmethod
    def _learning_row(row: sqlite3.Row) -> Dict[str, Any]:
        value = dict(row)
        for column, fallback in (
            ("payload_json", {}),
            ("result_json", {}),
            ("classification_json", {}),
            ("patch_json", {}),
            ("verification_json", {}),
            ("package_json", {}),
            ("source_ids_json", []),
        ):
            if column in value:
                value[column[:-5]] = _load_json(value.pop(column), fallback)
        return value

    def ensure_legacy_knowledge_package(self, profile_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Seed the current checked-in profile once and pin pre-existing cases to it."""

        # Import locally to keep the storage module usable for low-level SQLite
        # migration tests without a circular dependency at import time.
        from .learning import baseline_package_payload

        package = baseline_package_payload(profile_payload)
        version = str(package["package_version"])
        digest = str(package["package_sha256"])[:12]
        package_id = "legacy-" + digest
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT OR IGNORE INTO knowledge_packages(
                       id, version, parent_id, package_json, summary, verification_json, status, created_at, enabled_at
                   ) VALUES (?, ?, NULL, ?, ?, '{}', 'active', ?, ?)""",
                (package_id, version, _json(package), "Базовый профиль, включённый до накопительного обучения.", now, now),
            )
            control = connection.execute("SELECT active_package_id FROM learning_control WHERE id = 1").fetchone()
            if control is None or not control["active_package_id"]:
                connection.execute(
                    "UPDATE learning_control SET active_package_id = ?, updated_at = ? WHERE id = 1",
                    (package_id, now),
                )
            connection.execute(
                "UPDATE cases SET knowledge_package_id = ? WHERE knowledge_package_id IS NULL",
                (package_id,),
            )
        return self.get_knowledge_package(package_id)

    def get_knowledge_package(self, package_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM knowledge_packages WHERE id = ?", (package_id,)).fetchone()
        if row is None:
            raise StateError("Пакет знаний не найден")
        return self._learning_row(row)

    def active_knowledge_package(self) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT package.* FROM learning_control control
                   JOIN knowledge_packages package ON package.id = control.active_package_id
                   WHERE control.id = 1"""
            ).fetchone()
        if row is None:
            raise StateError("Базовый пакет знаний ещё не инициализирован")
        return self._learning_row(row)

    def knowledge_package_for_case(self, case_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT knowledge_package_id FROM cases WHERE id = ?", (case_id,)).fetchone()
        if row is None or not row["knowledge_package_id"]:
            raise StateError("У кейса нет закреплённого пакета знаний")
        return self.get_knowledge_package(str(row["knowledge_package_id"]))

    def create_knowledge_package(
        self,
        *,
        version: str,
        package: Dict[str, Any],
        parent_id: str,
        summary: str,
        verification: Dict[str, Any],
    ) -> Dict[str, Any]:
        from .learning import verify_immutable_package

        verified_package = verify_immutable_package(package)
        if str(verified_package["package_version"]) != version:
            raise StateError("Версия пакета не совпадает с его содержимым")
        package_id = "kp-" + uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO knowledge_packages(
                       id, version, parent_id, package_json, summary, verification_json, status, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, 'verified', ?)""",
                (package_id, version, parent_id, _json(verified_package), summary, _json(verification), now),
            )
        return self.get_knowledge_package(package_id)

    def publish_knowledge_package(
        self,
        *,
        version: str,
        package: Dict[str, Any],
        parent_id: str,
        summary: str,
        verification: Dict[str, Any],
        candidate_id: Optional[str] = None,
        source_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Persist an approved package and its candidate outcome atomically.

        ``candidate_id`` is used by the automatic learning path.  Including
        it makes package activation, candidate completion and the submitters'
        visible outcome one SQLite transaction: a restart can therefore see
        either the old package or the fully published outcome, never a
        half-published verified candidate.
        """

        from .learning import public_release_summary, verify_immutable_package

        checked = verify_immutable_package(package)
        if str(checked["package_version"]) != version:
            raise StateError("Версия пакета не совпадает с его содержимым")
        package_id = "kp-" + uuid.uuid4().hex
        # Keep the model's detailed rationale inside the immutable patch and
        # verifier archive.  The user-visible release summary is generated
        # from counts only, so it cannot disclose another submitter's facts.
        public_summary = public_release_summary(checked.get("patch", {}))
        now = _utc_now()
        frozen_source_ids = [str(value) for value in (source_ids or [])]
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute("SELECT active_package_id FROM learning_control WHERE id = 1").fetchone()
            if current is None or str(current["active_package_id"] or "") != parent_id:
                raise StateError("Базовый пакет изменился во время проверки; кандидат не включён")
            if candidate_id is not None:
                candidate = connection.execute(
                    "SELECT source_ids_json, status FROM learning_candidates WHERE id = ?", (candidate_id,)
                ).fetchone()
                if candidate is None or candidate["status"] not in {"candidate", "verified"}:
                    raise StateError("Кандидат больше нельзя включить")
                if [str(value) for value in _load_json(candidate["source_ids_json"], [])] != frozen_source_ids:
                    raise StateError("Основания кандидата изменились во время проверки")
            connection.execute(
                """INSERT INTO knowledge_packages(
                       id, version, parent_id, package_json, summary, verification_json, status, created_at, enabled_at
                   ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)""",
                (package_id, version, parent_id, _json(checked), public_summary, _json(verification), now, now),
            )
            connection.execute(
                "UPDATE knowledge_packages SET status = 'superseded' WHERE id = ? AND status = 'active'",
                (parent_id,),
            )
            connection.execute(
                "UPDATE learning_control SET active_package_id = ?, updated_at = ? WHERE id = 1",
                (package_id, now),
            )
            if candidate_id is not None:
                connection.execute(
                    """UPDATE learning_candidates
                       SET status = 'enabled', verification_json = ?, verified_at = ?, package_id = ?, error = NULL
                       WHERE id = ?""",
                    (_json(verification), now, package_id, candidate_id),
                )
                self._annotate_learning_sources_in_connection(
                    connection, frozen_source_ids, "enabled", package_id, now
                )
        self.record_learning_event("knowledge_enabled", "knowledge_package", package_id, status="enabled")
        if candidate_id is not None:
            self.record_learning_event("candidate_verified", "candidate", candidate_id, status="verified")
            self.record_learning_event("candidate_enabled", "candidate", candidate_id, status="enabled")
        return self.get_knowledge_package(package_id)

    def activate_knowledge_package(self, package_id: str, *, actor_user_id: Optional[int] = None) -> Dict[str, Any]:
        from .learning import verify_immutable_package

        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            candidate = connection.execute("SELECT * FROM knowledge_packages WHERE id = ?", (package_id,)).fetchone()
            if candidate is None:
                raise StateError("Пакет знаний не найден")
            if candidate["status"] not in {"verified", "active", "superseded"}:
                raise StateError("Пакет не прошёл проверку и не может быть включён")
            verify_immutable_package(_load_json(candidate["package_json"], {}))
            previous = connection.execute("SELECT active_package_id FROM learning_control WHERE id = 1").fetchone()
            if previous and previous["active_package_id"] and previous["active_package_id"] != package_id:
                connection.execute(
                    "UPDATE knowledge_packages SET status = 'superseded' WHERE id = ? AND status = 'active'",
                    (previous["active_package_id"],),
                )
            connection.execute(
                "UPDATE knowledge_packages SET status = 'active', enabled_at = ? WHERE id = ?",
                (now, package_id),
            )
            connection.execute(
                "UPDATE learning_control SET active_package_id = ?, updated_at = ? WHERE id = 1",
                (package_id, now),
            )
        self.record_learning_event("knowledge_enabled", "knowledge_package", package_id, status="enabled", user_id=actor_user_id)
        return self.get_knowledge_package(package_id)

    def rollback_knowledge_package(self, package_id: str, *, actor_user_id: Optional[int] = None) -> Dict[str, Any]:
        package = self.get_knowledge_package(package_id)
        if package["status"] not in {"superseded", "active"}:
            raise StateError("Этот пакет нельзя включить как версию отката")
        return self.activate_knowledge_package(package_id, actor_user_id=actor_user_id)

    def learning_control(self) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM learning_control WHERE id = 1").fetchone()
        return dict(row) if row else {"auto_update_paused": 0, "active_package_id": None}

    def set_learning_paused(self, paused: bool, *, actor_user_id: Optional[int] = None) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = _utc_now()
            connection.execute(
                "UPDATE learning_control SET auto_update_paused = ?, updated_at = ? WHERE id = 1",
                (1 if paused else 0, now),
            )
            if paused:
                # A running model invocation is never killed, but the next
                # automatic stage must wait for the operator to resume the
                # daily schedule.
                connection.execute(
                    """UPDATE learning_jobs SET status = 'paused', finished_at = NULL
                       WHERE status = 'queued' AND kind IN ('knowledge_generate', 'knowledge_verify')"""
                )
            else:
                connection.execute(
                    """UPDATE learning_jobs SET status = 'scheduled', finished_at = NULL
                       WHERE status = 'paused' AND kind IN ('knowledge_generate', 'knowledge_verify')"""
                )
        self.record_learning_event(
            "learning_paused" if paused else "learning_resumed",
            "learning_control",
            "1",
            status="paused" if paused else "active",
            user_id=actor_user_id,
        )

    def record_learning_event(
        self,
        event_type: str,
        object_type: str,
        object_id: str,
        *,
        status: str,
        chat_id: Optional[int] = None,
        user_id: Optional[int] = None,
        duration_ms: Optional[int] = None,
        error_code: Optional[str] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO learning_events(
                       id, event_type, object_type, object_id, chat_id, user_id, status, duration_ms, error_code, detail_json, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    uuid.uuid4().hex,
                    event_type,
                    object_type,
                    object_id,
                    chat_id,
                    user_id,
                    status,
                    duration_ms,
                    error_code,
                    _json(detail or {}),
                    _utc_now(),
                ),
            )

    def ensure_free_space(self, min_free_bytes: int, target_dir: Optional[Path] = None) -> None:
        """Fail closed on the filesystem that will receive this upload."""

        free = shutil.disk_usage(target_dir or self.knowledge_root).free
        if free < min_free_bytes:
            raise InputError(
                "Недостаточно свободного места для сохранения архива. Освободите место и повторите загрузку."
            )

    # --- Feedback and meeting intake state machines ---

    def start_feedback(self, case_id: str, chat_id: int, user_id: Optional[int], audit_id: Optional[str]) -> Dict[str, Any]:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            case = connection.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()
            latest = connection.execute(
                "SELECT id FROM cases WHERE chat_id = ? ORDER BY created_at DESC LIMIT 1", (chat_id,)
            ).fetchone()
            if case is None or latest is None or latest["id"] != case_id or not case["review_json"]:
                raise StateError("Этот отчёт больше недоступен. Откройте актуальный отчёт.")
            if case["activity"] != "idle" or case["mode"] in {"grilling", "stopping"}:
                raise self._busy_error(case)
            pending = connection.execute(
                """SELECT id, status FROM feedback_submissions WHERE chat_id = ?
                   AND status IN ('awaiting_text', 'received') ORDER BY created_at DESC LIMIT 1""",
                (chat_id,),
            ).fetchone()
            if pending:
                if pending["status"] == "awaiting_text":
                    raise StateError("Обратная связь уже ожидает пояснение. Отправьте текст или /cancel.")
                raise StateError("Обратная связь уже принята и рассматривается; повтор не добавлен. Статус: /learning.")
            submission_id = uuid.uuid4().hex
            connection.execute(
                """INSERT INTO feedback_submissions(
                       id, case_id, audit_id, chat_id, user_id, status, result_json, created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, 'awaiting_text', '{}', ?, ?)""",
                (submission_id, case_id, audit_id, chat_id, user_id, now, now),
            )
        self.record_learning_event("feedback_requested", "feedback", submission_id, status="awaiting_text", chat_id=chat_id, user_id=user_id)
        return self.get_feedback(submission_id)

    def pending_feedback(self, chat_id: int, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM feedback_submissions WHERE chat_id = ? AND status = 'awaiting_text'"
        values: Tuple[Any, ...] = (chat_id,)
        if user_id is not None:
            query += " AND (user_id IS NULL OR user_id = ?)"
            values += (user_id,)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, values).fetchone()
        return self._learning_row(row) if row else None

    def feedback_in_flight(self, chat_id: int, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return a feedback item whose one allowed explanation was consumed.

        The Telegram client can deliver two text updates before it displays the
        acknowledgement for the first one.  ``received`` is therefore still
        owned by feedback intake, even though it is no longer *pending input*.
        Without this distinction, the second update could accidentally become a
        new case artifact.
        """

        query = "SELECT * FROM feedback_submissions WHERE chat_id = ? AND status = 'received'"
        values: Tuple[Any, ...] = (chat_id,)
        if user_id is not None:
            query += " AND (user_id IS NULL OR user_id = ?)"
            values += (user_id,)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, values).fetchone()
        return self._learning_row(row) if row else None

    def get_feedback(self, submission_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM feedback_submissions WHERE id = ?", (submission_id,)).fetchone()
        if row is None:
            raise StateError("Обратная связь не найдена")
        return self._learning_row(row)

    def submit_feedback(self, submission_id: str, text: str) -> Dict[str, Any]:
        clean = text.strip()
        if not clean:
            raise InputError("Нужно коротко пояснить, что именно бот понял неверно или пропустил")
        now = _utc_now()
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM feedback_submissions WHERE id = ?", (submission_id,)).fetchone()
            if row is None or row["status"] != "awaiting_text":
                raise StateError("Это пояснение уже принято; повтор не добавлен")
            job_id = self._insert_learning_job(
                connection,
                chat_id=int(row["chat_id"]),
                user_id=row["user_id"],
                kind="feedback_analyze",
                submission_id=submission_id,
                payload={},
                now=now,
            )
            connection.execute(
                """UPDATE feedback_submissions
                   SET status = 'received', text = ?, text_sha256 = ?, learning_job_id = ?, updated_at = ?
                   WHERE id = ?""",
                (clean, digest, job_id, now, submission_id),
            )
        self.record_learning_event("feedback_received", "feedback", submission_id, status="received", chat_id=int(row["chat_id"]), user_id=row["user_id"])
        return self.get_learning_job(job_id)

    def cancel_feedback(self, chat_id: int, user_id: Optional[int] = None) -> bool:
        pending = self.pending_feedback(chat_id, user_id)
        if not pending:
            return False
        with self._connect() as connection:
            connection.execute(
                "UPDATE feedback_submissions SET status = 'cancelled', updated_at = ? WHERE id = ? AND status = 'awaiting_text'",
                (_utc_now(), pending["id"]),
            )
        self.record_learning_event("feedback_cancelled", "feedback", pending["id"], status="cancelled", chat_id=chat_id, user_id=user_id)
        return True

    def _assert_meeting_allowed(self, connection: sqlite3.Connection, chat_id: int) -> None:
        case = connection.execute(
            "SELECT * FROM cases WHERE chat_id = ? ORDER BY created_at DESC LIMIT 1", (chat_id,)
        ).fetchone()
        if case is not None and (case["activity"] != "idle" or case["mode"] in {"grilling", "stopping"}):
            raise StateError("Нельзя загружать встречу, пока идёт аудит или репетиция. Завершите текущую операцию.")
        pending_feedback = connection.execute(
            "SELECT status FROM feedback_submissions WHERE chat_id = ? AND status IN ('awaiting_text', 'received') "
            "ORDER BY created_at DESC LIMIT 1", (chat_id,)
        ).fetchone()
        if pending_feedback:
            if pending_feedback["status"] == "awaiting_text":
                raise StateError("Сначала отправьте или отмените пояснение к общей обратной связи командой /cancel.")
            raise StateError("Обратная связь уже принята и рассматривается. Дождитесь результата; статус: /learning.")

    def begin_meeting(self, chat_id: int, user_id: Optional[int]) -> Dict[str, Any]:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._assert_meeting_allowed(connection, chat_id)
            existing = connection.execute(
                """SELECT id FROM meeting_submissions
                   WHERE chat_id = ? AND status IN ('awaiting_metadata', 'awaiting_source', 'needs_clarification', 'received')
                   ORDER BY created_at DESC LIMIT 1""",
                (chat_id,),
            ).fetchone()
            if existing:
                raise StateError("Загрузка встречи уже начата или проверяется. Дождитесь результата или используйте /cancel.")
            submission_id = uuid.uuid4().hex
            connection.execute(
                """INSERT INTO meeting_submissions(
                       id, chat_id, user_id, status, step, classification_json, created_at, updated_at
                   ) VALUES (?, ?, ?, 'awaiting_metadata', 'date', '{}', ?, ?)""",
                (submission_id, chat_id, user_id, now, now),
            )
        self.record_learning_event("meeting_started", "meeting", submission_id, status="awaiting_metadata", chat_id=chat_id, user_id=user_id)
        return self.get_meeting(submission_id)

    def pending_meeting(self, chat_id: int, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        query = """SELECT * FROM meeting_submissions WHERE chat_id = ?
                   AND status IN ('awaiting_metadata', 'awaiting_source', 'needs_clarification')"""
        values: Tuple[Any, ...] = (chat_id,)
        if user_id is not None:
            query += " AND (user_id IS NULL OR user_id = ?)"
            values += (user_id,)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, values).fetchone()
        return self._learning_row(row) if row else None

    def meeting_in_flight(self, chat_id: int, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return a transcript that has been accepted but is still classified."""

        query = "SELECT * FROM meeting_submissions WHERE chat_id = ? AND status = 'received'"
        values: Tuple[Any, ...] = (chat_id,)
        if user_id is not None:
            query += " AND (user_id IS NULL OR user_id = ?)"
            values += (user_id,)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._connect() as connection:
            row = connection.execute(query, values).fetchone()
        return self._learning_row(row) if row else None

    def get_meeting(self, submission_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM meeting_submissions WHERE id = ?", (submission_id,)).fetchone()
        if row is None:
            raise StateError("Загрузка встречи не найдена")
        return self._learning_row(row)

    def is_duplicate_meeting_metadata(
        self, submission_id: str, value: str, telegram_message_id: Optional[int] = None
    ) -> bool:
        """Recognise a repeated Telegram update before it becomes the next field.

        Telegram may redeliver an update, and a client can send the same text
        twice before the next-step prompt reaches it.  Persisting a digest makes
        this decision survive restarts without retaining another copy of input.
        """

        clean = " ".join(value.split())
        if not clean:
            return False
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT last_metadata_sha256, last_metadata_message_id FROM meeting_submissions WHERE id = ?",
                (submission_id,),
            ).fetchone()
        if row is None:
            return False
        if telegram_message_id is not None and row["last_metadata_message_id"] == telegram_message_id:
            return True
        return bool(row["last_metadata_sha256"] and row["last_metadata_sha256"] == digest)

    def submit_meeting_metadata(
        self, submission_id: str, value: str, telegram_message_id: Optional[int] = None
    ) -> Dict[str, Any]:
        clean = " ".join(value.split())
        digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM meeting_submissions WHERE id = ?", (submission_id,)).fetchone()
            if row is None or row["status"] != "awaiting_metadata":
                raise StateError("Этот шаг загрузки встречи уже завершён")
            if (
                (telegram_message_id is not None and row["last_metadata_message_id"] == telegram_message_id)
                or (row["last_metadata_sha256"] and row["last_metadata_sha256"] == digest)
            ):
                raise StateError("Это сообщение уже принято; отправьте данные для следующего шага.")
            step = str(row["step"])
            if step == "date":
                try:
                    datetime.strptime(clean, "%Y-%m-%d")
                except ValueError as error:
                    raise InputError("Укажите дату в формате ГГГГ-ММ-ДД, например 2026-09-16") from error
                updates = (clean, "title", step, digest, telegram_message_id, now, submission_id)
                connection.execute(
                    """UPDATE meeting_submissions
                       SET meeting_date = ?, step = ?, last_metadata_step = ?, last_metadata_sha256 = ?,
                           last_metadata_message_id = ?, updated_at = ? WHERE id = ?""",
                    updates,
                )
            elif step == "title":
                if len(clean) < 3:
                    raise InputError("Нужно короткое название встречи")
                connection.execute(
                    """UPDATE meeting_submissions
                       SET title = ?, step = 'vitaly', last_metadata_step = ?, last_metadata_sha256 = ?,
                           last_metadata_message_id = ?, updated_at = ? WHERE id = ?""",
                    (clean[:300], step, digest, telegram_message_id, now, submission_id),
                )
            elif step == "vitaly":
                if len(clean) < 2:
                    raise InputError("Нужно обозначение нужного Виталия среди говорящих")
                connection.execute(
                    """UPDATE meeting_submissions
                       SET vitaly_designation = ?, status = 'awaiting_source', step = 'source',
                           last_metadata_step = ?, last_metadata_sha256 = ?, last_metadata_message_id = ?, updated_at = ?
                       WHERE id = ?""",
                    (clean[:300], step, digest, telegram_message_id, now, submission_id),
                )
            else:
                raise StateError("Сначала отправьте сам транскрипт")
        return self.get_meeting(submission_id)

    def submit_meeting_clarification(self, submission_id: str, value: str) -> Dict[str, Any]:
        clean = " ".join(value.split())
        if len(clean) < 2:
            raise InputError("Нужно уточнение, как распознать нужного Виталия")
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM meeting_submissions WHERE id = ?", (submission_id,)).fetchone()
            if row is None or row["status"] != "needs_clarification" or row["step"] != "clarification":
                raise StateError("Эта встреча не ожидает уточнения")
            job_id = self._insert_learning_job(
                connection,
                chat_id=int(row["chat_id"]),
                user_id=row["user_id"],
                kind="meeting_analyze",
                submission_id=submission_id,
                payload={"clarification": clean},
                now=now,
            )
            designation = str(row["vitaly_designation"] or "")
            connection.execute(
                """UPDATE meeting_submissions SET vitaly_designation = ?, status = 'received', step = 'classifying',
                   learning_job_id = ?, updated_at = ? WHERE id = ?""",
                ((designation + "; уточнение: " + clean).strip("; "), job_id, now, submission_id),
            )
        return self.get_learning_job(job_id)

    def accept_meeting_source(
        self,
        submission_id: str,
        *,
        source_name: str,
        source_path: Path,
        normalized_path: Path,
        source_sha256: str,
        normalized_sha256: str,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """Persist a transcript once; exact copies stop before a model run."""

        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM meeting_submissions WHERE id = ?", (submission_id,)).fetchone()
            if row is None or row["status"] != "awaiting_source" or row["step"] != "source":
                raise StateError("Сначала завершите текущий шаг загрузки встречи")
            duplicate = connection.execute(
                """SELECT id FROM meeting_submissions
                   WHERE id != ? AND source_sha256 = ?
                     AND status IN ('received', 'accepted', 'duplicate', 'continuation', 'other_person', 'needs_clarification')
                   ORDER BY created_at LIMIT 1""",
                (submission_id, source_sha256),
            ).fetchone()
            if duplicate is None:
                # A renamed/re-exported transcript has a different source hash
                # but identical normalized content.  It is still not an
                # independent meeting and must not spend a model run.
                duplicate = connection.execute(
                    """SELECT id FROM meeting_submissions
                       WHERE id != ? AND normalized_sha256 = ?
                         AND status IN ('received', 'accepted', 'duplicate', 'continuation', 'other_person', 'needs_clarification')
                       ORDER BY created_at LIMIT 1""",
                    (submission_id, normalized_sha256),
                ).fetchone()
            if duplicate:
                connection.execute(
                    """UPDATE meeting_submissions
                       SET source_name = ?, source_path = ?, normalized_path = ?, source_sha256 = ?, normalized_sha256 = ?,
                           status = 'duplicate', step = 'done', duplicate_of = ?, updated_at = ? WHERE id = ?""",
                    (source_name, str(source_path), str(normalized_path), source_sha256, normalized_sha256, duplicate["id"], now, submission_id),
                )
                event_status = "duplicate"
                job = None
            else:
                job_id = self._insert_learning_job(
                    connection,
                    chat_id=int(row["chat_id"]),
                    user_id=row["user_id"],
                    kind="meeting_analyze",
                    submission_id=submission_id,
                    payload={},
                    now=now,
                )
                connection.execute(
                    """UPDATE meeting_submissions
                       SET source_name = ?, source_path = ?, normalized_path = ?, source_sha256 = ?, normalized_sha256 = ?,
                           status = 'received', step = 'classifying', learning_job_id = ?, updated_at = ? WHERE id = ?""",
                    (source_name, str(source_path), str(normalized_path), source_sha256, normalized_sha256, job_id, now, submission_id),
                )
                event_status = "received"
                job = self._get_learning_job(connection, job_id)
        meeting = self.get_meeting(submission_id)
        self.record_learning_event("meeting_source_received", "meeting", submission_id, status=event_status, chat_id=int(meeting["chat_id"]), user_id=meeting.get("user_id"))
        return meeting, job

    def cancel_meeting(self, chat_id: int, user_id: Optional[int] = None) -> bool:
        pending = self.pending_meeting(chat_id, user_id)
        if not pending:
            return False
        with self._connect() as connection:
            connection.execute(
                "UPDATE meeting_submissions SET status = 'cancelled', step = 'done', updated_at = ? WHERE id = ?",
                (_utc_now(), pending["id"]),
            )
        self.record_learning_event("meeting_cancelled", "meeting", pending["id"], status="cancelled", chat_id=chat_id, user_id=user_id)
        return True

    def set_meeting_result(self, submission_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        status = str(result.get("status") or "needs_clarification")
        if status not in {"accepted", "duplicate", "continuation", "other_person", "needs_clarification"}:
            raise StateError("Некорректный результат разбора встречи")
        with self._connect() as connection:
            connection.execute(
                """UPDATE meeting_submissions SET status = ?, step = ?, classification_json = ?,
                   duplicate_of = COALESCE(?, duplicate_of), continuation_of = COALESCE(?, continuation_of), updated_at = ? WHERE id = ?""",
                (
                    status,
                    "done" if status != "needs_clarification" else "clarification",
                    _json(result),
                    result.get("duplicate_of"),
                    result.get("continuation_of"),
                    _utc_now(),
                    submission_id,
                ),
            )
        meeting = self.get_meeting(submission_id)
        self.record_learning_event("meeting_classified", "meeting", submission_id, status=status, chat_id=int(meeting["chat_id"]), user_id=meeting.get("user_id"))
        return meeting

    def mark_learning_submission_failed(self, job_id: str, error: str) -> None:
        """Atomically finalize a failed learning job and any linked candidate.

        The original source and detailed model-run failure stay in the archive.
        A user may start a fresh feedback or meeting intake after the explicit
        failure notification, but a duplicate update during a running job is
        always rejected before it reaches a case.
        """

        now = _utc_now()
        failed_candidate: Optional[Tuple[str, List[str]]] = None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            learning_job = connection.execute(
                "SELECT kind, payload_json FROM learning_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            connection.execute(
                """UPDATE learning_jobs SET status = 'failed', error = ?, finished_at = ?
                   WHERE id = ? AND status IN ('queued', 'running')""",
                (error, now, job_id),
            )
            if learning_job is not None and learning_job["kind"] == "knowledge_verify":
                candidate_id = str(_load_json(learning_job["payload_json"], {}).get("candidate_id") or "")
                candidate = connection.execute(
                    "SELECT source_ids_json, verification_json, status FROM learning_candidates WHERE id = ?",
                    (candidate_id,),
                ).fetchone()
                if candidate is not None and candidate["status"] in {"candidate", "verified"}:
                    verification = _load_json(candidate["verification_json"], {})
                    verification.update(
                        {
                            "stage": "verification",
                            "reason": "Проверка кандидата не завершилась; прежняя версия знаний сохранена.",
                            "error_code": error,
                        }
                    )
                    connection.execute(
                        """UPDATE learning_candidates
                           SET status = 'failed', verification_json = ?, verified_at = ?, error = ? WHERE id = ?""",
                        (_json(verification), now, error, candidate_id),
                    )
                    failed_candidate = (candidate_id, [str(value) for value in _load_json(candidate["source_ids_json"], [])])
            feedback = connection.execute(
                "SELECT id FROM feedback_submissions WHERE learning_job_id = ? AND status = 'received'", (job_id,)
            ).fetchone()
            if feedback is not None:
                connection.execute(
                    """UPDATE feedback_submissions
                       SET status = 'failed', result_json = ?, updated_at = ? WHERE id = ?""",
                    (_json({"error": error, "learning_outcome": "failed"}), now, feedback["id"]),
                )
            meeting = connection.execute(
                "SELECT id FROM meeting_submissions WHERE learning_job_id = ? AND status = 'received'", (job_id,)
            ).fetchone()
            if meeting is not None:
                connection.execute(
                    """UPDATE meeting_submissions
                       SET status = 'failed', step = 'done', classification_json = ?, updated_at = ? WHERE id = ?""",
                    (_json({"reason": error, "learning_outcome": "failed"}), now, meeting["id"]),
                )
        if failed_candidate is not None:
            candidate_id, source_ids = failed_candidate
            self.annotate_learning_sources(source_ids, "verification_failed")
            self.record_learning_event(
                "candidate_failed", "candidate", candidate_id, status="failed", error_code=error
            )

    def own_learning_status(self, user_id: Optional[int], chat_id: int) -> Dict[str, List[Dict[str, Any]]]:
        where = "user_id = ?" if user_id is not None else "chat_id = ?"
        value: Any = user_id if user_id is not None else chat_id
        with self._connect() as connection:
            feedback = connection.execute(
                "SELECT * FROM feedback_submissions WHERE " + where + " ORDER BY created_at DESC LIMIT 10", (value,)
            ).fetchall()
            meetings = connection.execute(
                "SELECT * FROM meeting_submissions WHERE " + where + " ORDER BY created_at DESC LIMIT 10", (value,)
            ).fetchall()
        return {"feedback": [self._learning_row(row) for row in feedback], "meetings": [self._learning_row(row) for row in meetings]}

    def set_feedback_result(self, submission_id: str, status: str, result: Dict[str, Any]) -> Dict[str, Any]:
        if status not in {"case_only", "example", "candidate", "insufficient", "rejected", "analyzed"}:
            raise StateError("Некорректный результат обратной связи")
        with self._connect() as connection:
            connection.execute(
                "UPDATE feedback_submissions SET status = ?, result_json = ?, updated_at = ? WHERE id = ?",
                (status, _json(result), _utc_now(), submission_id),
            )
        value = self.get_feedback(submission_id)
        self.record_learning_event("feedback_analyzed", "feedback", submission_id, status=status, chat_id=int(value["chat_id"]), user_id=value.get("user_id"))
        return value

    def capture_point_correction(
        self,
        case_id: str,
        job_id: str,
        result: Dict[str, Any],
        correction: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Turn a resolved point correction into a traceable verification example.

        A correction that leaves the point in force is useful telemetry but not a
        training signal.  Removed/corrected points become a narrowly scoped
        candidate source; the later generator and verifier still decide whether
        it belongs in a shared rule.
        """

        decision = str(result.get("decision") or result.get("status") or "")
        if decision not in {"removed", "corrected"}:
            return None
        now = _utc_now()
        with self._connect() as connection:
            case = connection.execute("SELECT chat_id FROM cases WHERE id = ?", (case_id,)).fetchone()
            if case is None:
                return None
            existing = connection.execute(
                "SELECT id FROM feedback_submissions WHERE learning_job_id = ?", (job_id,)
            ).fetchone()
            if existing:
                return self.get_feedback(str(existing["id"]))
            submission_id = uuid.uuid4().hex
            correction = correction or {}
            item_id = str(correction.get("item_id") or result.get("item_id") or "")
            if not item_id:
                # A correction without its selected point cannot become a
                # reusable evidence example.  Preserve the original audit
                # history but do not manufacture a vague learning source.
                return None
            before = correction.get("before") if isinstance(correction.get("before"), dict) else {}
            applied = correction.get("result") if isinstance(correction.get("result"), dict) else result
            explanation = str(correction.get("explanation") or result.get("explanation") or "")
            payload = {
                "source": "point_correction",
                "decision": decision,
                "case_id": case_id,
                "item_id": item_id,
                "source_document_id": "audit:" + case_id + ":" + item_id,
                "fragment": explanation[:2000],
                "explanation": explanation,
                "before": before,
                "after": applied,
            }
            connection.execute(
                """INSERT INTO feedback_submissions(
                       id, case_id, audit_id, chat_id, status, text, text_sha256, result_json, learning_job_id, created_at, updated_at
                   ) VALUES (?, ?, NULL, ?, 'example', ?, ?, ?, ?, ?, ?)""",
                (
                    submission_id,
                    case_id,
                    int(case["chat_id"]),
                    "Точечное исправление " + item_id,
                    hashlib.sha256(_json(payload).encode("utf-8")).hexdigest(),
                    _json(payload),
                    job_id,
                    now,
                    now,
                ),
            )
        self.record_learning_event("point_correction_example", "feedback", submission_id, status="example", chat_id=int(case["chat_id"]))
        return self.get_feedback(submission_id)

    def learning_sources(self) -> List[Dict[str, Any]]:
        """Return only sources eligible for an automatic candidate, never raw logs."""

        with self._connect() as connection:
            feedback = connection.execute(
                "SELECT * FROM feedback_submissions WHERE status IN ('example', 'candidate') ORDER BY created_at"
            ).fetchall()
            meetings = connection.execute(
                "SELECT * FROM meeting_submissions WHERE status = 'accepted' ORDER BY created_at"
            ).fetchall()
        values: List[Dict[str, Any]] = []
        for row in feedback:
            value = self._learning_row(row)
            value["source_kind"] = "feedback"
            values.append(value)
        for row in meetings:
            value = self._learning_row(row)
            value["source_kind"] = "meeting"
            values.append(value)
        return values

    def learning_sources_by_ids(self, source_ids: Sequence[str]) -> List[Dict[str, Any]]:
        """Resolve a candidate's frozen source set with its original kind."""

        requested = [str(item) for item in source_ids]
        if not requested or len(set(requested)) != len(requested):
            raise StateError("Кандидат содержит некорректный набор источников")
        available = {str(item["id"]): item for item in self.learning_sources()}
        missing = [item for item in requested if item not in available]
        if missing:
            raise StateError("Основания кандидата больше недоступны: " + ", ".join(missing[:5]))
        return [available[item] for item in requested]

    def eligible_learning_sources(self) -> List[Dict[str, Any]]:
        """Sources not already sent through a candidate generator cycle."""

        with self._connect() as connection:
            candidate_rows = connection.execute("SELECT source_ids_json FROM learning_candidates").fetchall()
        consumed: set[str] = set()
        for row in candidate_rows:
            consumed.update(str(value) for value in _load_json(row["source_ids_json"], []))
        return [item for item in self.learning_sources() if str(item["id"]) not in consumed]

    def meeting_similarity_candidates(self, submission_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return metadata and paths for model-side duplicate/continuation checks."""

        current = self.get_meeting(submission_id)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM meeting_submissions
                   WHERE id != ? AND status IN ('accepted', 'duplicate', 'continuation', 'other_person')
                   ORDER BY CASE WHEN meeting_date = ? THEN 0 ELSE 1 END, created_at DESC LIMIT ?""",
                (submission_id, current.get("meeting_date"), limit),
            ).fetchall()
        return [self._learning_row(row) for row in rows]

    def create_learning_candidate(
        self,
        source_ids: Sequence[str],
        patch: Dict[str, Any],
        scope: str,
        expected_improvement: str,
        *,
        baseline_package_id: str,
    ) -> Dict[str, Any]:
        if scope not in {"analysis", "behaviour", "mixed"}:
            raise StateError("Некорректная область кандидата")
        candidate_id = "lc-" + uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO learning_candidates(
                       id, source_ids_json, patch_json, baseline_package_id, status, scope, expected_improvement, generated_at
                   ) VALUES (?, ?, ?, ?, 'candidate', ?, ?, ?)""",
                (
                    candidate_id,
                    _json(list(source_ids)),
                    _json(patch),
                    baseline_package_id,
                    scope,
                    expected_improvement[:1000],
                    _utc_now(),
                ),
            )
        self.record_learning_event("candidate_generated", "candidate", candidate_id, status="candidate", detail={"source_count": len(source_ids), "scope": scope})
        return self.get_learning_candidate(candidate_id)

    def create_learning_candidate_with_verification(
        self,
        source_ids: Sequence[str],
        patch: Dict[str, Any],
        scope: str,
        expected_improvement: str,
        *,
        baseline_package_id: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Atomically persist a candidate and its independent verifier job."""

        if scope not in {"analysis", "behaviour", "mixed"}:
            raise StateError("Некорректная область кандидата")
        candidate_id = "lc-" + uuid.uuid4().hex
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """INSERT INTO learning_candidates(
                       id, source_ids_json, patch_json, baseline_package_id, status, scope, expected_improvement, generated_at
                   ) VALUES (?, ?, ?, ?, 'candidate', ?, ?, ?)""",
                (
                    candidate_id,
                    _json(list(source_ids)),
                    _json(patch),
                    baseline_package_id,
                    scope,
                    expected_improvement[:1000],
                    now,
                ),
            )
            control = connection.execute("SELECT auto_update_paused FROM learning_control WHERE id = 1").fetchone()
            status = "paused" if control is not None and int(control["auto_update_paused"] or 0) else "queued"
            verification_job_id = self._insert_learning_job(
                connection,
                chat_id=0,
                user_id=None,
                kind="knowledge_verify",
                submission_id=None,
                payload={"candidate_id": candidate_id},
                now=now,
                status=status,
            )
        self.record_learning_event(
            "candidate_generated", "candidate", candidate_id, status="candidate",
            detail={"source_count": len(source_ids), "scope": scope},
        )
        return self.get_learning_candidate(candidate_id), self.get_learning_job(verification_job_id)

    def get_learning_candidate(self, candidate_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM learning_candidates WHERE id = ?", (candidate_id,)).fetchone()
        if row is None:
            raise StateError("Кандидат обновления не найден")
        return self._learning_row(row)

    def set_candidate_result(
        self,
        candidate_id: str,
        *,
        status: str,
        verification: Dict[str, Any],
        package_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        if status not in {"verified", "enabled", "rejected", "needs_clarification", "no_change", "failed"}:
            raise StateError("Некорректный статус кандидата")
        with self._connect() as connection:
            connection.execute(
                """UPDATE learning_candidates SET status = ?, verification_json = ?, verified_at = ?, package_id = ?, error = ?
                   WHERE id = ?""",
                (status, _json(verification), _utc_now(), package_id, error, candidate_id),
            )
        self.record_learning_event("candidate_" + status, "candidate", candidate_id, status=status, error_code=error)
        return self.get_learning_candidate(candidate_id)

    @staticmethod
    def _annotate_learning_sources_in_connection(
        connection: sqlite3.Connection,
        source_ids: Sequence[str],
        outcome: str,
        package_id: Optional[str],
        now: str,
    ) -> None:
        """Write per-submitter outcomes inside a caller-owned transaction."""

        for source_id in source_ids:
            feedback = connection.execute("SELECT result_json FROM feedback_submissions WHERE id = ?", (source_id,)).fetchone()
            if feedback is not None:
                result = _load_json(feedback["result_json"], {})
                result["learning_outcome"] = outcome
                if package_id:
                    result["knowledge_package_id"] = package_id
                connection.execute(
                    "UPDATE feedback_submissions SET result_json = ?, updated_at = ? WHERE id = ?",
                    (_json(result), now, source_id),
                )
                continue
            meeting = connection.execute("SELECT classification_json FROM meeting_submissions WHERE id = ?", (source_id,)).fetchone()
            if meeting is not None:
                result = _load_json(meeting["classification_json"], {})
                result["learning_outcome"] = outcome
                if package_id:
                    result["knowledge_package_id"] = package_id
                connection.execute(
                    "UPDATE meeting_submissions SET classification_json = ?, updated_at = ? WHERE id = ?",
                    (_json(result), now, source_id),
                )

    def annotate_learning_sources(self, source_ids: Sequence[str], outcome: str, package_id: Optional[str] = None) -> None:
        """Expose a package decision to its own submitters without exposing peers."""

        now = _utc_now()
        with self._connect() as connection:
            self._annotate_learning_sources_in_connection(connection, source_ids, outcome, package_id, now)

    def add_regression_run(
        self, candidate_id: str, baseline_package_id: str, result: Dict[str, Any], passing: bool
    ) -> Dict[str, Any]:
        run_id = "lr-" + uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO learning_regression_runs(id, candidate_id, baseline_package_id, result_json, passing, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, candidate_id, baseline_package_id, _json(result), 1 if passing else 0, _utc_now()),
            )
        return {"id": run_id, "candidate_id": candidate_id, "baseline_package_id": baseline_package_id, "result": result, "passing": passing}

    def due_learning_cycle(self, run_hour: int, *, now: Optional[datetime] = None) -> bool:
        """Whether the low-priority daily cycle may start in this local hour.

        The calendar hour is intentionally evaluated in the Mac's current
        timezone.  This prevents an elapsed 24-hour interval from slowly
        moving the background model work into a user's daytime session.  A
        busy queue does not cause a daytime catch-up run: the next opportunity
        is tomorrow's configured hour.
        """

        if not 0 <= run_hour <= 23:
            return False
        local_now = _local_now(now)
        if local_now.hour != run_hour:
            return False
        control = self.learning_control()
        if int(control.get("auto_update_paused") or 0):
            return False
        last = self._as_utc_datetime(str(control.get("last_cycle_at") or ""))
        if last is not None and last.astimezone(local_now.tzinfo).date() >= local_now.date():
            return False
        with self._connect() as connection:
            user_work = connection.execute("SELECT 1 FROM jobs WHERE status IN ('queued', 'running') LIMIT 1").fetchone()
            user_learning = connection.execute(
                """SELECT 1 FROM learning_jobs
                   WHERE status IN ('queued', 'running') AND kind IN ('feedback_analyze', 'meeting_analyze')
                   LIMIT 1"""
            ).fetchone()
            running_learning = connection.execute("SELECT 1 FROM learning_jobs WHERE status = 'running' LIMIT 1").fetchone()
            automatic_learning = connection.execute(
                """SELECT 1 FROM learning_jobs
                   WHERE status IN ('scheduled', 'queued', 'running')
                     AND kind IN ('knowledge_generate', 'knowledge_verify') LIMIT 1"""
            ).fetchone()
        return (
            user_work is None
            and user_learning is None
            and running_learning is None
            and automatic_learning is None
            and bool(self.learning_sources())
        )

    def queue_learning_cycle(self, run_hour: int, *, now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Atomically queue the daily cycle without allowing off-hour catch-up."""

        if not 0 <= run_hour <= 23:
            return None
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            # Read the clock only after the lock is acquired: a queued writer
            # can otherwise cross the 04:59 boundary while waiting here.
            local_now = _local_now(now)
            if local_now.hour != run_hour:
                return None
            now_text = local_now.astimezone(timezone.utc).isoformat()
            control = connection.execute("SELECT * FROM learning_control WHERE id = 1").fetchone()
            if control is None or int(control["auto_update_paused"] or 0):
                return None
            last = self._as_utc_datetime(str(control["last_cycle_at"] or ""))
            if last is not None and last.astimezone(local_now.tzinfo).date() >= local_now.date():
                return None
            existing = connection.execute(
                """SELECT id FROM learning_jobs
                   WHERE kind IN ('knowledge_generate', 'knowledge_verify')
                     AND status IN ('scheduled', 'queued', 'running') LIMIT 1"""
            ).fetchone()
            if existing:
                return self._get_learning_job(connection, str(existing["id"]))
            user_work = connection.execute("SELECT 1 FROM jobs WHERE status IN ('queued', 'running') LIMIT 1").fetchone()
            user_learning = connection.execute(
                """SELECT 1 FROM learning_jobs
                   WHERE status IN ('queued', 'running') AND kind IN ('feedback_analyze', 'meeting_analyze')
                   LIMIT 1"""
            ).fetchone()
            if user_work or user_learning:
                return None
            job_id = self._insert_learning_job(
                connection, chat_id=0, user_id=None, kind="knowledge_generate", submission_id=None, payload={}, now=now_text
            )
            connection.execute("UPDATE learning_control SET last_cycle_at = ?, updated_at = ? WHERE id = 1", (now_text, now_text))
            return self._get_learning_job(connection, job_id)

    def resume_scheduled_learning(
        self, run_hour: int, *, now: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Make deferred automatic work runnable only in its next daily window."""

        if not 0 <= run_hour <= 23 or _local_now(now).hour != run_hour:
            return []
        with self._connect() as connection:
            control = connection.execute("SELECT auto_update_paused FROM learning_control WHERE id = 1").fetchone()
            if control is None or int(control["auto_update_paused"] or 0):
                return []
            rows = connection.execute(
                """SELECT id FROM learning_jobs
                   WHERE status = 'scheduled' AND kind IN ('knowledge_generate', 'knowledge_verify')
                   ORDER BY created_at, id"""
            ).fetchall()
            if not rows:
                return []
            connection.execute(
                """UPDATE learning_jobs SET status = 'queued'
                   WHERE status = 'scheduled' AND kind IN ('knowledge_generate', 'knowledge_verify')"""
            )
            return [self._get_learning_job(connection, str(row["id"])) for row in rows]

    def create_learning_model_run(self, job_id: str, kind: str, model: str, reasoning_effort: str) -> str:
        run_id = uuid.uuid4().hex
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO learning_model_runs(id, learning_job_id, kind, model, reasoning_effort, status, started_at)
                   VALUES (?, ?, ?, ?, ?, 'running', ?)""",
                (run_id, job_id, kind, model, reasoning_effort, _utc_now()),
            )
        return run_id

    def finish_learning_model_run(self, run_id: str, error: Optional[str] = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE learning_model_runs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                ("failed" if error else "completed", error, _utc_now(), run_id),
            )

    def cleanup_expired_files(self, days: int) -> List[str]:
        """Retain evidence indefinitely.

        Kept as a compatibility API for launchd and older scripts.  Deleting an
        old case would break reproducibility and silently erase learning
        evidence, so expiry now only affects the active conversational context.
        """

        del days
        return []
