from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pytest

from tof_reliability.vitaly_bot.documents import DocumentProcessor, transcript_title
from tof_reliability.vitaly_bot.evidence import build_evidence_pack, extract_hints
from tof_reliability.vitaly_bot.models import ArtifactInput, BotConfig, InputError, ModelError, ModelResult, StateError
from tof_reliability.vitaly_bot.profile import load_default_profile, score_holdout_coverage
from tof_reliability.vitaly_bot.service import FifoWorker, VitalyService
from tof_reliability.vitaly_bot.storage import VitalyStore


def _config(tmp_path: Path) -> BotConfig:
    return BotConfig(token="test-token", state_dir=tmp_path / "state", codex_bin=Path("/usr/bin/codex"))


def _question(topic: str, text: str = "Где подтверждение?") -> Dict[str, Any]:
    return {
        "text": text,
        "topic": topic,
        "likelihood": 5,
        "severity": 5,
        "trigger": "Нет источника цифры",
        "material_reference": "Файл: input.txt",
        "expected_answer": "Назвать источник, периметр, формулу и владельца проверки.",
        "expected_answer_evidence_level": "material",
        "expected_answer_references": ["Файл: input.txt"],
        "followups": ["Кто проверит и когда?"],
        "profile_evidence": ["2026-06-08"],
        "confidence": "high",
    }


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.feedback_followup = ""

    async def run(
        self,
        prompt: str,
        schema: Dict[str, Any],
        image_paths: Optional[Sequence[Any]] = None,
    ) -> ModelResult:
        del image_paths
        self.calls.append(prompt)
        properties = schema["properties"]
        if "readiness_score" in properties:
            value = {
                "readiness_score": 47,
                "verdict": "Материал показывает идею, но не доказывает эффект.",
                "critical_gaps": [{"title": "Нет базы", "detail": "Не указан источник", "material_reference": "input.txt"}],
                "contradictions": [],
                "questions": [_question("evidence_scope"), _question("ownership_timing", "Кто сделает и к какой дате?")],
                "prep_checklist": ["Указать источник."],
            }
        elif "scores" in properties:
            value = {
                "verdict": "Ответ слишком общий.",
                "scores": {"evidence": 1, "specificity": 2, "numbers": 0, "ownership": 1},
                "challenge": "Назовите источник и срок.",
                "followup_question": self.feedback_followup,
                "answer_status": "incorrect",
                "correct_answer": "Источник SAP, периметр ТОФ, проверка Ивановым завтра.",
                "evidence_level": "ledger",
                "evidence_references": ["work_order_events: order_id=123456"],
                "missing_evidence": [],
            }
        elif "overall_score" in properties:
            value = {
                "overall_score": 40,
                "verdict": "К встрече нужна доработка.",
                "strengths": ["Понятна общая логика."],
                "weak_answers": ["Нет проверяемых цифр."],
                "before_meeting": ["Подготовить расчёт."],
            }
        else:
            value = {"summary": "Краткая карта", "claims": [], "numbers": [], "gaps": []}
        return ModelResult(value=value, stdout="", stderr="")


class FakeEvidenceProvider:
    async def build(self, material_text: str, artifact_hashes: Sequence[str]) -> Dict[str, Any]:
        assert material_text
        assert artifact_hashes
        return {
            "evidence_version": 1,
            "generation": {"run_id": "generation-test", "ledger_path": "/srv/tof/current/ledger", "promoted_at": ""},
            "candidate_ids": [],
            "capabilities": [],
            "records": [],
            "limitations": ["fixture"],
        }


def _add_text_artifact(store: VitalyStore, case_id: str, tmp_path: Path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("Заявлен эффект 20%, но источник и владелец не указаны.", encoding="utf-8")
    DocumentProcessor(store).ingest(
        case_id,
        ArtifactInput("telegram-file-1", source.name, source, source.stat().st_size),
    )


def test_meeting_title_uses_transcript_header_and_filename_fallback():
    assert transcript_title("Meeting Title:  План ремонта на октябрь\nDate: Sep 25", "upload.docx") == "План ремонта на октябрь"
    assert transcript_title("Them: Виталий, какой срок?", "план_ремонтов.md") == "план_ремонтов"


def test_profile_has_exactly_confirmed_core_and_excludes_other_vitaly():
    profile = load_default_profile()

    assert len(profile.core_meetings) == 12
    assert "2026-05-20-muimapov" not in {meeting["id"] for meeting in profile.core_meetings}
    assert {"2026-07-31", "2026-08-18"} == set(profile.holdout["meeting_ids"])
    assert score_holdout_coverage(profile, profile.holdout["expected_topic_ids"])["passing"] is True
    assert score_holdout_coverage(profile, ["evidence_scope"])["passing"] is False


def test_updated_profile_has_traceable_september_patterns_and_preserves_baseline():
    from tof_reliability.vitaly_bot.profile import VitalyProfile, default_profile_path
    from tof_reliability.vitaly_bot.models import InputError
    import copy
    import json

    path = default_profile_path()
    value = json.loads(path.read_text())
    profile = VitalyProfile.from_dict(value)
    baseline = VitalyProfile.load(path.with_name("vitaly_profile.v1.json"))
    assert baseline.version == "v1.0.0"
    assert len(baseline.core_meetings) == 10
    assert profile.version == "v1.1.0"
    assert {m["id"] for m in baseline.core_meetings} <= {m["id"] for m in profile.core_meetings}
    assert profile.holdout["meeting_ids"] == baseline.holdout["meeting_ids"]
    root = Path(__file__).resolve().parents[1]
    for pattern in profile.question_patterns:
        assert pattern["followup_chain"]
        for ref in pattern["evidence_references"]:
            assert ref["meeting_id"] in pattern["evidence_meetings"]
            assert (root / ref["source"]).is_file()
            assert ref["time_range"]
    assert all("evidence_references" not in pattern for pattern in profile.prompt_context()["question_patterns"])
    duplicate = copy.deepcopy(value)
    duplicate["meeting_index"].append(copy.deepcopy(duplicate["meeting_index"][-1]))
    with pytest.raises(InputError, match="повторяются"):
        VitalyProfile.from_dict(duplicate)
    wrong_person = copy.deepcopy(value)
    wrong_person["meeting_index"][-1]["person"] = "vitaly_muimapov"
    with pytest.raises(InputError, match="другого Виталия"):
        VitalyProfile.from_dict(wrong_person)


def test_local_config_uses_explicit_env_and_never_needs_a_repository_token(tmp_path: Path):
    from tof_reliability.vitaly_bot.config import load_config

    config = load_config(
        {
            "VITALY_BOT_TOKEN": "only-in-test",
            "VITALY_BOT_STATE_DIR": str(tmp_path / "state"),
            "VITALY_BOT_ENV_FILE": str(tmp_path / "missing.env"),
            "VITALY_BOT_CODEX_BIN": "/tmp/codex",
        }
    )

    assert config.token == "only-in-test"
    assert config.state_dir == tmp_path / "state"
    assert config.codex_bin == Path("/tmp/codex")
    assert config.model_timeout_seconds == 480
    assert config.context_ttl_days == 4
    assert config.knowledge_dir == tmp_path / "state" / "knowledge"
    assert config.cleanup_days == 0
    assert config.learning_run_hour == 4
    assert config.operator_user_ids == ()


@pytest.mark.parametrize("value", ["-1", "24", "four"])
def test_local_config_rejects_an_invalid_learning_run_hour(tmp_path: Path, value: str):
    from tof_reliability.vitaly_bot.config import load_config

    with pytest.raises(InputError, match="VITALY_BOT_LEARNING_RUN_HOUR"):
        load_config(
            {
                "VITALY_BOT_TOKEN": "only-in-test",
                "VITALY_BOT_STATE_DIR": str(tmp_path / "state"),
                "VITALY_BOT_ENV_FILE": str(tmp_path / "missing.env"),
                "VITALY_BOT_CODEX_BIN": "/tmp/codex",
                "VITALY_BOT_LEARNING_RUN_HOUR": value,
            }
        )


def test_learning_status_displays_the_configured_local_hour(tmp_path: Path):
    pytest.importorskip("aiogram")
    from tof_reliability.vitaly_bot.telegram import format_learning_status

    config = BotConfig(
        token="test-token", state_dir=tmp_path / "state", codex_bin=Path("/usr/bin/codex"), learning_run_hour=3
    )
    store = VitalyStore(config.state_dir)
    VitalyService(store, load_default_profile(), config)

    text = format_learning_status(SimpleNamespace(store=store, config=config), chat_id=1, user_id=1)
    assert "03:00 по времени Mac" in text


def test_model_timeout_message_confirms_the_process_was_stopped():
    pytest.importorskip("aiogram")
    from tof_reliability.vitaly_bot.telegram import user_error

    assert "запуск остановлен" in user_error(ModelError("model_timeout")).casefold()


def test_launchd_agent_exports_a_path_for_server_evidence_tools(tmp_path: Path):
    from scripts.install_vitaly_bot_launch_agent import build_plist

    payload = build_plist(tmp_path, tmp_path / "config.env", tmp_path / "logs")
    assert "PATH" in payload["EnvironmentVariables"]
    assert "/usr/bin" in payload["EnvironmentVariables"]["PATH"]


def test_launchd_reload_replaces_an_already_loaded_job(tmp_path: Path):
    from scripts.install_vitaly_bot_launch_agent import reload_launch_agent

    calls: list[list[str]] = []

    def runner(command: list[str], check: bool) -> SimpleNamespace:
        assert check is False
        calls.append(command)
        return SimpleNamespace(returncode=0)

    target = tmp_path / "bot.plist"
    assert reload_launch_agent(target, runner=runner) == 0
    assert calls[0][:2] == ["launchctl", "bootout"]
    assert calls[1][:2] == ["launchctl", "bootstrap"]
    assert str(target) == calls[0][-1] == calls[1][-1]


def test_launchd_reload_uses_legacy_fallback_if_bootstrap_rejects_the_job(tmp_path: Path):
    from scripts.install_vitaly_bot_launch_agent import reload_launch_agent

    statuses = iter([0, 1, 0])
    calls: list[list[str]] = []

    def runner(command: list[str], check: bool) -> SimpleNamespace:
        assert check is False
        calls.append(command)
        return SimpleNamespace(returncode=next(statuses))

    assert reload_launch_agent(tmp_path / "bot.plist", runner=runner) == 0
    assert calls[-1][:3] == ["launchctl", "load", "-w"]


def test_document_ingestion_normalizes_text_and_invalidates_old_review(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    case = store.create_case(101, "v1")
    source = tmp_path / "input.txt"
    source.write_text("Проверить периметр расчёта", encoding="utf-8")
    artifact = DocumentProcessor(store).ingest(
        case["id"], ArtifactInput("file-1", "входные данные.txt", source, source.stat().st_size)
    )

    assert artifact["suffix"] == ".txt"
    assert "Проверить периметр расчёта" in Path(artifact["normalized_text_path"]).read_text(encoding="utf-8")
    store.set_review(case["id"], {"ready": True})
    second = tmp_path / "second.txt"
    second.write_text("Добавился второй материал", encoding="utf-8")
    DocumentProcessor(store).ingest(case["id"], ArtifactInput("file-2", second.name, second, second.stat().st_size))

    refreshed = store.get_case(case["id"])
    assert refreshed["review"] == {}
    assert refreshed["mode"] == "collecting"


def test_document_limits_are_checked_before_copying(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    case = store.create_case(101, "v1")
    source = tmp_path / "input.txt"
    source.write_text("x", encoding="utf-8")

    try:
        DocumentProcessor(store).validate_input(
            case["id"], ArtifactInput("file", source.name, source, 50 * 1024 * 1024 + 1)
        )
    except Exception as error:
        assert "50 МБ" in str(error)
    else:
        raise AssertionError("oversized upload must be rejected")


def test_store_preserves_fifo_order_and_recovers_running_jobs(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    case = store.create_case(101, "v1")
    first = store.enqueue_job(case["id"], 101, "review")
    second = store.enqueue_job(case["id"], 101, "finish")

    assert store.next_queued_job()["id"] == first["id"]
    assert store.queue_position(second["id"]) == 2
    store.mark_job_running(first["id"])
    recovered = store.recover_jobs()

    assert [job["id"] for job in recovered] == [first["id"], second["id"]]
    assert store.get_job(first["id"])["status"] == "queued"
    assert store.get_case(case["id"])["activity"] == "queued"


def test_case_operation_lock_rejects_parallel_upload_and_duplicate_job(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    case = store.create_case(101, "v1")

    store.begin_upload(case["id"])
    try:
        store.require_idle(case["id"])
    except StateError as error:
        assert "Бот занят" in str(error)
    else:
        raise AssertionError("busy upload must be visible before file validation")
    try:
        store.begin_job(case["id"], 101, "review")
    except StateError as error:
        assert "Бот занят" in str(error)
    else:
        raise AssertionError("job must not start while upload is active")
    store.finish_upload(case["id"])

    first = store.begin_job(case["id"], 101, "review")
    try:
        store.begin_job(case["id"], 101, "review")
    except StateError as error:
        assert "Бот занят" in str(error)
    else:
        raise AssertionError("duplicate command must not create a second job")
    assert store.queue_position(first["id"]) == 1


def test_answer_is_saved_once_and_stop_queues_only_one_summary(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    case = store.create_case(101, "v1")
    question = _question("evidence_scope")
    question.update({"planned_index": 0, "planned_total": 1, "is_followup": False})
    store.set_question(case["id"], question, 0, "grilling")

    answer_job = store.accept_answer_and_enqueue(case["id"], 101, "Ответ с источником.")
    try:
        store.accept_answer_and_enqueue(case["id"], 101, "Второй ответ.")
    except StateError as error:
        assert "Бот занят" in str(error)
    else:
        raise AssertionError("second answer must not be accepted")
    assert len(store.turns(case["id"])) == 1

    assert store.request_stop(case["id"], 101) is None
    store.finish_job(answer_job["id"])
    finish_job = store.queue_finish(case["id"], 101)
    assert finish_job is not None
    try:
        store.request_stop(case["id"], 101)
    except StateError as error:
        assert "Остановка уже запрошена" in str(error)
    else:
        raise AssertionError("duplicate stop must not enqueue another summary")


def test_stop_without_answer_stops_without_a_model_summary(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    case = store.create_case(101, "v1")
    question = _question("evidence_scope")
    store.set_question(case["id"], question, 0, "grilling")

    assert store.request_stop(case["id"], 101) is None
    stopped = store.get_case(case["id"])
    assert stopped["mode"] == "stopped"
    assert stopped["activity"] == "idle"
    assert stopped["closed_by_stop"] == 1


def test_inactive_non_stopped_case_is_replaced_and_old_material_is_not_reused(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    case = store.create_case(101, "v1")
    _add_text_artifact(store, case["id"], tmp_path)
    artifact = store.artifacts(case["id"])[0]
    store._update_case(
        case["id"], last_user_activity_at=(datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    )

    fresh, expired_case_id = store.activate_case_for_user(101, "v1", 4)

    assert expired_case_id == case["id"]
    assert fresh["id"] != case["id"]
    assert store.get_case(case["id"])["mode"] == "expired"
    assert store.artifact_count(fresh["id"]) == 0
    assert Path(artifact["local_path"]).exists()


def test_explicitly_stopped_case_is_not_auto_reset_after_four_days(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    case = store.create_case(101, "v1")
    question = _question("evidence_scope")
    store.set_question(case["id"], question, 0, "grilling")
    store.request_stop(case["id"], 101)
    store._update_case(
        case["id"], last_user_activity_at=(datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    )

    current, expired_case_id = store.activate_case_for_user(101, "v1", 4)

    assert current["id"] == case["id"]
    assert expired_case_id is None


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def test_evidence_hints_ignore_document_paths_dates_and_prose_words():
    material = (
        "Источник: /cases/a9489883745b_20260930_GenAI.pdf\n"
        "Оборудования для пилота достаточно. ЕО категории 3-4.\n"
        "929395 Гайка M20, 660678 Болт с гайкой. Заказ № 12345678.\n"
    )

    assert set(extract_hints(material)["candidate_ids"]) == {"929395", "660678", "12345678"}


def test_evidence_lookup_avoids_loading_entire_ledger(tmp_path: Path, monkeypatch):
    from tof_reliability import data

    ledger = tmp_path / "ledger"
    ledger.mkdir()
    _write_csv(
        ledger / "source_documents.csv",
        ["source_document_id", "relative_path", "sha256", "document_type", "parser_status"],
        [
            {"source_document_id": "doc-good", "relative_path": "good.csv", "sha256": "a", "document_type": "xlsx", "parser_status": "published"},
            {"source_document_id": "doc-review", "relative_path": "review.csv", "sha256": "b", "document_type": "xlsx", "parser_status": "needs_review"},
        ],
    )
    _write_csv(
        ledger / "work_order_events.csv",
        ["event_id", "source_document_id", "source_file", "order_id", "equipment_id", "work_type_raw", "event_date", "actual_cost"],
        [
            {"event_id": "event-good", "source_document_id": "doc-good", "source_file": "good.csv", "order_id": "123456", "equipment_id": "EO-1", "work_type_raw": "ТР", "event_date": "2026-01-01", "actual_cost": "100"},
            {"event_id": "event-review", "source_document_id": "doc-review", "source_file": "review.csv", "order_id": "123456", "equipment_id": "EO-2", "work_type_raw": "ТР", "event_date": "2026-01-02", "actual_cost": "200"},
        ],
    )
    state = tmp_path / "generation.json"
    state.write_text('{"run_id":"generation-1","ledger_path":"' + str(ledger) + '"}', encoding="utf-8")
    original_load_table = data.load_table

    def reject_full_table(name: str, **kwargs: Any):
        if name != "source_documents":
            raise AssertionError("full ledger table loaded: " + name)
        return original_load_table(name, **kwargs)

    monkeypatch.setattr(data, "load_table", reject_full_table)
    pack = build_evidence_pack(
        {"version": 1, "candidate_ids": ["123456"], "artifact_hashes": ["hash"]},
        generation_state_path=state,
        root=Path(__file__).resolve().parents[1],
    )

    assert [(row["record"]["event_id"], row["source_document_ids"]) for row in pack["records"]] == [
        ("event-good", ["doc-good"])
    ]


def test_evidence_pack_uses_current_generation_and_excludes_needs_review(tmp_path: Path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    _write_csv(
        ledger / "source_documents.csv",
        ["source_document_id", "relative_path", "sha256", "document_type", "parser_status"],
        [{"source_document_id": "doc-1", "relative_path": "source.csv", "sha256": "a", "document_type": "xlsx", "parser_status": "published"}],
    )
    _write_csv(
        ledger / "source_records.csv",
        ["event_id", "source_document_id", "record_kind"],
        [{"event_id": "event-1", "source_document_id": "doc-1", "record_kind": "row"}],
    )
    _write_csv(
        ledger / "work_order_events.csv",
        ["event_id", "source_file", "order_id", "equipment_id", "work_type_raw", "event_date", "actual_cost"],
        [{"event_id": "event-1", "source_file": "source.csv", "order_id": "123456", "equipment_id": "EO-1", "work_type_raw": "ТР", "event_date": "2026-01-01", "actual_cost": "100"}],
    )
    state = tmp_path / "generation.json"
    state.write_text('{"run_id":"generation-1","ledger_path":"' + str(ledger) + '"}', encoding="utf-8")
    pack = build_evidence_pack(
        {"version": 1, "candidate_ids": ["123456"], "artifact_hashes": ["hash"]},
        generation_state_path=state,
        root=Path(__file__).resolve().parents[1],
    )

    assert pack["generation"]["run_id"] == "generation-1"
    assert pack["records"][0]["source_document_ids"] == ["doc-1"]
    assert pack["records"][0]["ledger_table"] == "work_order_events"

    _write_csv(
        ledger / "source_documents.csv",
        ["source_document_id", "relative_path", "sha256", "document_type", "parser_status"],
        [{"source_document_id": "doc-1", "relative_path": "source.csv", "sha256": "a", "document_type": "xlsx", "parser_status": "needs_review"}],
    )
    excluded = build_evidence_pack(
        {"version": 1, "candidate_ids": ["123456"], "artifact_hashes": ["hash"]},
        generation_state_path=state,
        root=Path(__file__).resolve().parents[1],
    )
    assert excluded["records"] == []


class FakeTelegramBot:
    def __init__(self) -> None:
        self.id = 1
        self.messages: list[str] = []
        self.edits: list[str] = []
        self.actions = 0

    async def send_message(self, _chat_id: int, text: str) -> SimpleNamespace:
        self.messages.append(text)
        return SimpleNamespace(message_id=len(self.messages))

    async def edit_message_text(self, text: str, **_kwargs: Any) -> SimpleNamespace:
        self.edits.append(text)
        return SimpleNamespace(message_id=1)

    async def send_chat_action(self, **_kwargs: Any) -> bool:
        self.actions += 1
        return True


def test_progress_status_uses_typing_and_correction_output(tmp_path: Path):
    import pytest

    pytest.importorskip("aiogram")

    async def scenario() -> None:
        from tof_reliability.vitaly_bot.telegram import TelegramRuntime

        config = _config(tmp_path)
        bot = FakeTelegramBot()
        runtime = TelegramRuntime(bot, config)  # type: ignore[arg-type]
        case = runtime.store.create_case(101, "v1")
        runtime.store.begin_upload(case["id"])
        await runtime._ensure_progress(case["id"], 101)
        await asyncio.sleep(0.02)
        runtime.store.finish_upload(case["id"])
        await runtime._end_progress(case["id"], 101, "✅ Файл обработан")
        await runtime.stop()

        assert bot.messages
        assert bot.actions >= 1
        assert bot.edits[-1] == "✅ Файл обработан"

    asyncio.run(scenario())
    from tof_reliability.vitaly_bot.telegram import format_feedback, format_question

    text = format_feedback(
        {
            "feedback": {
                "verdict": "Нет источника.",
                "scores": {"evidence": 0, "specificity": 1, "numbers": 0, "ownership": 0},
                "challenge": "Где SAP?",
                "answer_status": "incorrect",
                "correct_answer": "Назвать заказ 123456 и источник SAP.",
                "evidence_level": "ledger",
                "evidence_references": ["work_order_events: order_id=123456"],
                "missing_evidence": [],
            },
            "next_question": _question("ownership_timing"),
        }
    )
    assert "Как лучше ответить" in text
    assert "work_order_events" in text
    assert "Основной вопрос" in format_question({"text": "Кто?", "planned_index": 2, "planned_total": 8})


def test_telegram_notifies_that_an_inactive_context_was_reset(tmp_path: Path):
    import pytest

    pytest.importorskip("aiogram")

    async def scenario() -> None:
        from tof_reliability.vitaly_bot.telegram import TelegramRuntime

        runtime = TelegramRuntime(FakeTelegramBot(), _config(tmp_path))  # type: ignore[arg-type]
        old_case = runtime.store.create_case(101, "v1")
        _add_text_artifact(runtime.store, old_case["id"], tmp_path)
        runtime.store._update_case(
            old_case["id"], last_user_activity_at=(datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        )

        fresh = await runtime.case_for_user(101)
        await runtime.stop()

        assert fresh["id"] != old_case["id"]
        assert any("Контекст обнулён" in text for text in runtime.bot.messages)
        assert any("не будут использованы" in text for text in runtime.bot.messages)

    asyncio.run(scenario())


def test_learning_scheduler_recovers_after_a_transient_tick_error(tmp_path: Path, monkeypatch):
    pytest.importorskip("aiogram")

    async def scenario() -> None:
        from tof_reliability.vitaly_bot import telegram
        from tof_reliability.vitaly_bot.telegram import TelegramRuntime

        runtime = TelegramRuntime(FakeTelegramBot(), _config(tmp_path))  # type: ignore[arg-type]
        attempts = 0

        def flaky_resume(_hour: int) -> list[Dict[str, Any]]:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise StateError("temporary SQLite error")
            return []

        sleeps = 0

        async def stop_after_two_ticks(_seconds: float) -> None:
            nonlocal sleeps
            sleeps += 1
            if sleeps == 2:
                raise asyncio.CancelledError

        monkeypatch.setattr(runtime.store, "resume_scheduled_learning", flaky_resume)
        monkeypatch.setattr(runtime.store, "due_learning_cycle", lambda _hour: False)
        monkeypatch.setattr(telegram.asyncio, "sleep", stop_after_two_ticks)

        with pytest.raises(asyncio.CancelledError):
            await runtime._learning_scheduler()

        assert attempts == 2
        with runtime.store._connect() as connection:
            event = connection.execute(
                "SELECT error_code FROM learning_events WHERE event_type = 'learning_scheduler_failed'"
            ).fetchone()
        assert event["error_code"] == "scheduler_tick_failed"

    asyncio.run(scenario())


def test_review_grill_followup_and_finish_are_explicitly_persisted(tmp_path: Path):
    async def scenario() -> None:
        config = _config(tmp_path)
        store = VitalyStore(config.state_dir)
        profile = load_default_profile()
        runner = FakeRunner()
        service = VitalyService(store, profile, config, runner=runner, evidence_provider=FakeEvidenceProvider())
        case = service.new_case(101)
        _add_text_artifact(store, case["id"], tmp_path)

        review = await service.review_case(case["id"], "job-review")
        assert review["readiness_score"] == 47
        start = await service.start_grill(case["id"], "job-start")
        assert start.payload["question"]["planned_index"] == 0

        runner.feedback_followup = "А кто лично проверит цифру?"
        turn = service.store_answer(case["id"], "Проверим позже.")
        first_feedback = await service.evaluate_turn(case["id"], "job-answer-1", turn["id"])
        assert first_feedback.payload["next_question"]["is_followup"] is True

        runner.feedback_followup = "Не должно использоваться после follow-up."
        turn = service.store_answer(case["id"], "Иванов, срок завтра, источник SAP.")
        second_feedback = await service.evaluate_turn(case["id"], "job-answer-2", turn["id"])
        assert second_feedback.payload["next_question"]["planned_index"] == 1
        assert second_feedback.payload["next_question"]["text"] == "Кто сделает и к какой дате?"

        result = await service.finish_grill(case["id"], "job-finish")
        assert result.payload["overall_score"] == 40
        assert store.get_case(case["id"])["mode"] == "finished"

    asyncio.run(scenario())


def test_stop_during_answer_evaluation_suppresses_the_next_question(tmp_path: Path):
    async def scenario() -> None:
        config = _config(tmp_path)
        store = VitalyStore(config.state_dir)
        service = VitalyService(
            store, load_default_profile(), config, runner=FakeRunner(), evidence_provider=FakeEvidenceProvider()
        )
        case = service.new_case(101)
        _add_text_artifact(store, case["id"], tmp_path)
        await service.start_grill(case["id"], "job-start")
        turn = service.store_answer(case["id"], "Пока не проверили.")
        store.set_stop_requested(case["id"], True)
        outcome = await service.evaluate_turn(case["id"], "job-answer", turn["id"])

        assert outcome.payload["stopped"] is True
        assert outcome.payload["next_question"] is None
        assert store.get_case(case["id"])["mode"] == "stopping"

    asyncio.run(scenario())


def test_fifo_worker_executes_a_review_and_persists_completion(tmp_path: Path):
    async def scenario() -> None:
        config = _config(tmp_path)
        store = VitalyStore(config.state_dir)
        service = VitalyService(
            store, load_default_profile(), config, runner=FakeRunner(), evidence_provider=FakeEvidenceProvider()
        )
        case = service.new_case(101)
        _add_text_artifact(store, case["id"], tmp_path)
        completed: list[JobOutcome] = []
        failures: list[Exception] = []

        async def on_completed(_job: Dict[str, Any], outcome: JobOutcome) -> None:
            completed.append(outcome)

        async def on_failed(_job: Dict[str, Any], error: Exception) -> None:
            failures.append(error)

        worker = FifoWorker(store, service, on_completed, on_failed)
        await worker.start()
        job = store.enqueue_job(case["id"], 101, "review")
        worker.notify()
        for _ in range(20):
            if store.get_job(job["id"])["status"] == "completed":
                break
            await asyncio.sleep(0.01)
        await worker.stop()

        assert not failures
        assert completed[0].kind == "review"
        assert store.get_job(job["id"])["status"] == "completed"

    asyncio.run(scenario())


def test_codex_command_is_isolated_and_schema_constrained(tmp_path: Path):
    from tof_reliability.vitaly_bot.codex import CodexRunner

    config = _config(tmp_path)
    command = CodexRunner(config).command(tmp_path, tmp_path / "schema.json", tmp_path / "output.json", [])

    assert "--ephemeral" in command
    assert "--output-schema" in command
    assert "read-only" in command
    assert "gpt-6-astra" in command


def test_codex_timeout_stops_the_entire_cli_process_group(tmp_path: Path):
    import os

    from tof_reliability.vitaly_bot.codex import CodexRunner

    async def scenario() -> None:
        child_pid_path = tmp_path / "child.pid"
        hanging_cli = tmp_path / "hanging-codex.sh"
        hanging_cli.write_text(
            "#!/bin/sh\nsleep 60 &\necho $! > " + str(child_pid_path) + "\nwait\n", encoding="utf-8"
        )
        hanging_cli.chmod(0o755)
        config = BotConfig(
            token="test-token",
            state_dir=tmp_path / "state",
            codex_bin=hanging_cli,
            model_timeout_seconds=2,
        )
        try:
            await CodexRunner(config).run("test", {"type": "object", "properties": {}, "required": []})
        except ModelError as error:
            assert error.code == "model_timeout"
            assert error.detail == "limit_seconds=2"
        else:
            raise AssertionError("hanging Codex CLI must time out")

        child_pid = int(child_pid_path.read_text(encoding="utf-8").strip())
        for _ in range(20):
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                return
            await asyncio.sleep(0.05)
        raise AssertionError("the Codex child process survived its timeout")

    asyncio.run(scenario())
