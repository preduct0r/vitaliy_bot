"""Telegram adapter for the Vitaly simulator's persistent application service."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, Message
from aiogram.utils.chat_action import ChatActionSender

from .config import load_config
from .documents import MAX_FILE_BYTES
from .models import ArtifactInput, BotConfig, EvidenceError, InputError, ModelError, StateError, VitalyBotError
from .profile import load_default_profile
from .service import FifoWorker, JobOutcome, VitalyService
from .storage import VitalyStore
from . import correction_ui


HELP_TEXT = (
    "Это симулятор жёсткой проверки перед встречей, а не настоящий Виталий.\n\n"
    "1. Отправьте до 10 файлов или текст и дождитесь сообщения «Файл обработан».\n"
    "2. /review — получить аудит материалов.\n"
    "3. /grill — пройти конечную репетицию вопрос за вопросом.\n"
    "4. /stop — прекратить вопросы и получить итоговый разбор.\n"
    "5. /status — увидеть этап, очередь и ошибку.\n\n"
    "Ошибочный пункт аудита можно перепроверить кнопкой «Исправить пункт» под отчётом.\n\n"
    "Пока бот занят, новые файлы, текст и команды не принимаются. Поддерживаются PPTX, PDF, DOCX, XLSX, TXT, MD, "
    "CSV, PNG и JPEG; до 50 МБ на файл. Новый кейс: /new."
)


def split_message(text: str, limit: int = 3900) -> List[str]:
    """Keep Telegram output readable without splitting a word where possible."""

    if len(text) <= limit:
        return [text]
    chunks: List[str] = []
    remaining = text
    while len(remaining) > limit:
        boundary = remaining.rfind("\n", 0, limit)
        if boundary < limit // 2:
            boundary = remaining.rfind(" ", 0, limit)
        if boundary < limit // 2:
            boundary = limit
        chunks.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


async def send_with_retry(bot: Bot, chat_id: int, text: str, **kwargs: Any) -> List[int]:
    """Retry only Telegram networking failures, never duplicate logic errors."""

    message_ids = []
    for chunk in split_message(text):
        last_error: Optional[TelegramNetworkError] = None
        for attempt in range(3):
            try:
                message = await bot.send_message(chat_id, chunk, **kwargs)
                message_ids.append(message.message_id)
                last_error = None
                break
            except TelegramNetworkError as error:
                last_error = error
                await asyncio.sleep(2 ** attempt)
        if last_error is not None:
            raise last_error
    return message_ids


def _bullet(items: Iterable[str]) -> str:
    values = list(items)
    return "\n".join("• " + value for value in values) if values else "• Нет"


def format_review(review: Dict[str, Any]) -> str:
    lines = [
        "Аудит материалов",
        "Готовность: " + str(review["readiness_score"]) + "/100",
        review["verdict"],
        "\nКритические пробелы:",
        _bullet("G" + str(i) + ". " + item["title"] + " — " + item["detail"] for i, item in enumerate(review["critical_gaps"], 1)),
        "\nПротиворечия:",
        _bullet("C" + str(i) + ". " + item["title"] + " — " + item["detail"] for i, item in enumerate(review["contradictions"], 1)),
        "\nВероятные вопросы:",
    ]
    for number, question in enumerate(review["questions"], start=1):
        lines.extend(
            [
                "Q" + str(number) + ". " + question["text"],
                "   Риск " + str(question["severity"]) + "/5, вероятность " + str(question["likelihood"]) + "/5.",
                "   Почему: " + question["trigger"],
                "   Короткий ответ: " + question["expected_answer"],
            ]
        )
        references = question.get("expected_answer_references", [])
        if references:
            lines.append("   Основание: " + "; ".join(references))
        if question["followups"]:
            lines.append("   Дальше спросит: " + "; ".join(question["followups"]))
    lines.extend(["\nПеред встречей:", _bullet(review["prep_checklist"])])
    return "\n".join(lines)


def format_question(question: Dict[str, Any]) -> str:
    number = int(question.get("planned_index", 0)) + 1
    total = int(question.get("planned_total", number))
    title = "Уточнение к вопросу " if question.get("is_followup") else "Основной вопрос "
    return (
        "Репетиция. "
        + title
        + str(number)
        + " из "
        + str(total)
        + ":\n"
        + question["text"]
        + "\n\nОтвечайте одним сообщением. После ответа бот даст разбор; /stop завершит репетицию."
    )


def format_feedback(payload: Dict[str, Any]) -> str:
    feedback = payload["feedback"]
    scores = feedback["scores"]
    text = (
        "Разбор ответа: " + feedback["verdict"] + "\n"
        + "Оценки: доказательность " + str(scores["evidence"]) + "/5; конкретность " + str(scores["specificity"])
        + "/5; цифры " + str(scores["numbers"]) + "/5; владелец и срок " + str(scores["ownership"]) + "/5.\n"
        + "Уточнение: " + feedback["challenge"]
    )
    if feedback.get("answer_status") != "correct":
        text += "\n\nКак лучше ответить: " + feedback.get("correct_answer", "Подтверждённого ответа нет.")
        references = feedback.get("evidence_references", [])
        if references:
            text += "\nОснование: " + "; ".join(references)
        missing = feedback.get("missing_evidence", [])
        if missing:
            text += "\nНужно подтвердить: " + "; ".join(missing)
    question = payload.get("next_question")
    if question:
        return text + "\n\n" + format_question(question)
    if payload.get("stopped"):
        return text + "\n\nНовых вопросов не будет; готовлю итог по завершённым ответам."
    return text + "\n\nВопросы закончились; готовлю итог репетиции."


def format_finish(result: Dict[str, Any]) -> str:
    return "\n".join(
        [
            "Итог репетиции: " + str(result["overall_score"]) + "/100",
            result["verdict"],
            "\nСильные стороны:",
            _bullet(result["strengths"]),
            "\nСлабые ответы:",
            _bullet(result["weak_answers"]),
            "\nДо встречи:",
            _bullet(result["before_meeting"]),
        ]
    )


def user_error(error: Exception) -> str:
    if isinstance(error, EvidenceError):
        if error.code == "tof_unavailable":
            return "Текущие данные ТОФ недоступны; проверка не выполнена."
        return "Не удалось собрать проверяемые факты ТОФ: " + error.code
    if isinstance(error, ModelError):
        messages = {
            "codex_not_logged_in": "Codex на этом Mac не авторизован в ChatGPT. Выполните `codex login` локально.",
            "subscription_limit_reached": "Лимит ChatGPT-подписки для Codex исчерпан. Попробуйте позже.",
            "model_timeout": "Модель не ответила вовремя; запуск остановлен. Повторите действие позже.",
            "model_output_invalid_json": "Модель вернула неполный структурированный ответ. Повторите команду.",
            "model_output_invalid_schema": "Модель вернула ответ не по контракту. Повторите команду.",
        }
        return messages.get(error.code, "Модельный запуск не завершился: " + error.code)
    if isinstance(error, VitalyBotError):
        return str(error)
    return "Внутренняя ошибка. Деталь сохранена в /status."


def _elapsed(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        started = datetime.fromisoformat(value)
    except ValueError:
        return ""
    seconds = max(0, int((datetime.now(timezone.utc) - started).total_seconds()))
    return str(seconds // 60) + " мин " + str(seconds % 60) + " сек"


class TelegramRuntime:
    """Composition root shared by handlers and the background FIFO worker."""

    def __init__(self, bot: Bot, config: BotConfig) -> None:
        self.bot = bot
        self.config = config
        self.store = VitalyStore(config.state_dir)
        self.profile = load_default_profile()
        self.service = VitalyService(self.store, self.profile, config)
        self.worker = FifoWorker(self.store, self.service, self._completed, self._failed)
        self._progress_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._progress_text: Dict[str, str] = {}

    async def start(self) -> None:
        self.store.cleanup_expired_files(self.config.cleanup_days)
        await self.worker.start()
        for case in self.store.active_cases():
            await self._ensure_progress(case["id"], int(case["chat_id"]))

    async def case_for_user(self, chat_id: int) -> Dict[str, Any]:
        case, expired_case_id = self.service.case_for_user(chat_id)
        if expired_case_id is not None:
            await send_with_retry(
                self.bot,
                chat_id,
                "Контекст обнулён: предыдущий кейс не закрыли командой /stop и он не использовался более "
                + str(self.config.context_ttl_days)
                + " суток. "
                "Ранее загруженные файлы не будут использованы. Отправьте материалы заново.",
            )
        return case

    async def stop(self) -> None:
        for task in list(self._progress_tasks.values()):
            task.cancel()
        for task in list(self._progress_tasks.values()):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self.worker.stop()

    def _progress_message(self, case: Dict[str, Any]) -> str:
        activity = str(case.get("activity") or "idle")
        detail = str(case.get("activity_detail") or "")
        if activity in {"uploading", "queued"}:
            prefix = "⏳"
        elif activity in {"thinking", "processing_materials", "checking_ledger", "evaluating_answer", "summarizing"}:
            prefix = "🧠"
        elif activity == "stopping":
            prefix = "⏹"
        else:
            prefix = "⏳"
        text = prefix + " " + (detail or "Бот занят")
        active_job_id = case.get("active_job_id")
        if active_job_id:
            try:
                job = self.store.get_job(str(active_job_id))
            except StateError:
                job = None
            if job and job["status"] == "queued":
                text += ". Очередь: " + str(self.store.queue_position(job["id"]))
        elapsed = _elapsed(case.get("activity_started_at"))
        if elapsed:
            text += ". Длится: " + elapsed
        text += ". Пока доступен /status" + (" и /stop" if case.get("mode") in {"grilling", "stopping"} else "") + "."
        return text

    async def _render_progress(self, case_id: str, chat_id: int, override: Optional[str] = None) -> None:
        case = self.store.get_case(case_id)
        text = override or self._progress_message(case)
        if self._progress_text.get(case_id) == text:
            return
        message_id = case.get("progress_message_id")
        try:
            if message_id:
                await self.bot.edit_message_text(text, chat_id=chat_id, message_id=int(message_id))
            else:
                message = await self.bot.send_message(chat_id, text)
                self.store.set_progress_message(case_id, message.message_id)
        except (TelegramBadRequest, TelegramNetworkError):
            try:
                message = await self.bot.send_message(chat_id, text)
                self.store.set_progress_message(case_id, message.message_id)
            except TelegramNetworkError:
                return
        self._progress_text[case_id] = text

    async def _progress_loop(self, case_id: str, chat_id: int) -> None:
        try:
            async with ChatActionSender.typing(bot=self.bot, chat_id=chat_id, initial_sleep=0):
                while True:
                    case = self.store.get_case(case_id)
                    await self._render_progress(case_id, chat_id)
                    if case.get("activity") in {"idle", "awaiting_answer"}:
                        return
                    await asyncio.sleep(2)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A cosmetic Telegram action must never stop a document/model job.
            return

    async def _ensure_progress(self, case_id: str, chat_id: int) -> None:
        await self._render_progress(case_id, chat_id)
        current = self._progress_tasks.get(case_id)
        if current is None or current.done():
            self._progress_tasks[case_id] = asyncio.create_task(
                self._progress_loop(case_id, chat_id), name="vitaly-progress-" + case_id[:8]
            )

    async def _end_progress(self, case_id: str, chat_id: int, text: str) -> None:
        task = self._progress_tasks.pop(case_id, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._render_progress(case_id, chat_id, text)

    async def queue(self, case_id: str, chat_id: int, kind: str, payload: Optional[Dict[str, Any]] = None) -> int:
        job = self.store.begin_job(case_id, chat_id, kind, payload)
        self.worker.notify()
        await self._ensure_progress(case_id, chat_id)
        return self.store.queue_position(job["id"])

    async def _queue_final(self, case_id: str, chat_id: int, stopped: bool) -> None:
        job = self.store.queue_finish(case_id, chat_id)
        if job is None:
            await self._end_progress(case_id, chat_id, "⏹ Репетиция остановлена: завершённых ответов нет.")
            return
        self.worker.notify()
        await self._ensure_progress(case_id, chat_id)
        if not stopped:
            await self._render_progress(case_id, chat_id, "🧠 Вопросы закончились; готовлю итог репетиции.")

    async def _complete_stopped_grill(self, case_id: str, chat_id: int) -> None:
        case = self.store.get_case(case_id)
        await self._queue_final(case_id, chat_id, stopped=bool(case.get("stop_requested")))

    async def _completed(self, job: Dict[str, Any], outcome: JobOutcome) -> None:
        chat_id = int(job["chat_id"])
        case_id = str(job["case_id"])
        case = self.store.get_case(case_id)
        if outcome.kind == "review":
            self.store.settle_job(case_id, mode="ready")
            await self._end_progress(case_id, chat_id, "✅ Аудит материалов готов.")
            await correction_ui.show_report(self, case_id, chat_id)
            return
        if outcome.kind == "review_correction":
            self.store.settle_job(case_id, mode="ready")
            await self._end_progress(case_id, chat_id, "✅ Пункт перепроверен.")
            await correction_ui.show_result(self, job, outcome.payload)
            return
        if outcome.kind == "grill_stopped":
            await self._complete_stopped_grill(case_id, chat_id)
            return
        if outcome.kind == "grill_question":
            if case.get("stop_requested"):
                await self._complete_stopped_grill(case_id, chat_id)
                return
            await self._end_progress(case_id, chat_id, "✅ Вопрос репетиции готов.")
            await send_with_retry(self.bot, chat_id, format_question(outcome.payload["question"]))
            return
        if outcome.kind == "grill_feedback":
            await self._end_progress(case_id, chat_id, "✅ Ответ разобран.")
            await send_with_retry(self.bot, chat_id, format_feedback(outcome.payload))
            if outcome.payload.get("next_question") is None:
                await self._complete_stopped_grill(case_id, chat_id)
            return
        if outcome.kind == "finish":
            self.store.settle_job(case_id, mode="finished")
            await self._end_progress(case_id, chat_id, "✅ Итог репетиции готов.")
            await send_with_retry(self.bot, chat_id, format_finish(outcome.payload))
            return
        self.store.settle_job(case_id)
        await self._end_progress(case_id, chat_id, "✅ Задача завершена.")

    async def _failed(self, job: Dict[str, Any], error: Exception) -> None:
        chat_id = int(job["chat_id"])
        case_id = str(job["case_id"])
        case = self.store.get_case(case_id)
        if case.get("stop_requested"):
            await send_with_retry(self.bot, chat_id, "Текущий этап не завершился, но новых вопросов не будет.")
            await self._complete_stopped_grill(case_id, chat_id)
            return
        if job["kind"] == "grill_answer":
            self.store.settle_job(case_id, activity="awaiting_answer", detail="", mode="grilling")
        elif job["kind"] == "finish":
            self.store.set_stop_requested(case_id, False)
            self.store.settle_job(case_id, mode="ready")
        else:
            self.store.settle_job(case_id, mode="ready" if self.store.artifact_count(case_id) else "collecting")
        await self._end_progress(case_id, chat_id, "❌ " + user_error(error))
        await send_with_retry(self.bot, chat_id, "Не удалось выполнить запрос: " + user_error(error))


def _case_status(runtime: TelegramRuntime, case: Dict[str, Any]) -> str:
    queue = runtime.store.queue_summary()
    text = (
        "Кейс " + case["id"][:8] + ": " + str(case["mode"]) + ". "
        + "Файлов: " + str(runtime.store.artifact_count(case["id"])) + ". "
        + "Этап: " + str(case.get("activity") or "idle")
    )
    if case.get("activity_detail"):
        text += " (" + str(case["activity_detail"]) + ")"
    active_job_id = case.get("active_job_id")
    if active_job_id:
        try:
            job = runtime.store.get_job(str(active_job_id))
        except StateError:
            job = None
        if job and job["status"] == "queued":
            text += ". Позиция: " + str(runtime.store.queue_position(job["id"]))
    elapsed = _elapsed(case.get("activity_started_at"))
    if elapsed:
        text += ". Длится: " + elapsed
    question = case.get("current_question") or {}
    if question:
        text += ". Вопрос: " + str(int(question.get("planned_index", 0)) + 1) + " из " + str(question.get("planned_total", "?"))
    text += ". Общая очередь: " + str(queue.get("queued", 0)) + ", выполняется: " + str(queue.get("running", 0)) + "."
    if case.get("last_error"):
        text += " Последняя ошибка: " + str(case["last_error"])
        detail = " ".join(str(case.get("last_error_detail") or "").split())[:350]
        if detail:
            text += " (" + detail + ")"
    return text


def build_router(runtime: TelegramRuntime) -> Router:
    router = Router(name="vitaly_bot")
    correction_ui.register(router, runtime)

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        if message.chat:
            await runtime.case_for_user(message.chat.id)
            await send_with_retry(runtime.bot, message.chat.id, HELP_TEXT)

    @router.message(Command("help"))
    async def help_command(message: Message) -> None:
        await runtime.case_for_user(message.chat.id)
        await send_with_retry(runtime.bot, message.chat.id, HELP_TEXT)

    @router.message(Command("new"))
    async def new_case(message: Message) -> None:
        try:
            case = runtime.store.new_case_if_idle(message.chat.id, runtime.profile.version)
            await send_with_retry(runtime.bot, message.chat.id, "Новый кейс создан: " + case["id"][:8] + ". Отправьте материалы.")
        except Exception as error:
            await send_with_retry(runtime.bot, message.chat.id, user_error(error))

    @router.message(Command("review"))
    async def review(message: Message) -> None:
        try:
            case = await runtime.case_for_user(message.chat.id)
            runtime.store.require_idle(case["id"])
            runtime.service.corrections.cancel(case["id"])
            if not runtime.store.artifacts(case["id"]):
                raise StateError("Сначала отправьте хотя бы один файл")
            await runtime.queue(case["id"], message.chat.id, "review")
        except Exception as error:
            await send_with_retry(runtime.bot, message.chat.id, user_error(error))

    @router.message(Command("grill"))
    async def grill(message: Message) -> None:
        try:
            case = await runtime.case_for_user(message.chat.id)
            runtime.store.require_idle(case["id"])
            runtime.service.corrections.cancel(case["id"])
            if not runtime.store.artifacts(case["id"]):
                raise StateError("Сначала отправьте хотя бы один файл")
            await runtime.queue(case["id"], message.chat.id, "grill_start")
        except Exception as error:
            await send_with_retry(runtime.bot, message.chat.id, user_error(error))

    @router.message(Command("finish"))
    async def finish_alias(message: Message) -> None:
        await runtime.case_for_user(message.chat.id)
        await send_with_retry(runtime.bot, message.chat.id, "Команда переименована в /stop")

    @router.message(Command("stop"))
    async def stop_grill(message: Message) -> None:
        try:
            case = await runtime.case_for_user(message.chat.id)
            job = runtime.store.request_stop(case["id"], message.chat.id)
            if job is not None:
                runtime.worker.notify()
                await runtime._ensure_progress(case["id"], message.chat.id)
            elif runtime.store.get_case(case["id"])["mode"] == "stopped":
                await runtime._end_progress(case["id"], message.chat.id, "⏹ Репетиция остановлена: завершённых ответов нет.")
            else:
                await runtime._ensure_progress(case["id"], message.chat.id)
        except Exception as error:
            await send_with_retry(runtime.bot, message.chat.id, user_error(error))

    @router.message(Command("status"))
    async def status(message: Message) -> None:
        case = await runtime.case_for_user(message.chat.id)
        await send_with_retry(runtime.bot, message.chat.id, _case_status(runtime, case))

    async def save_upload(message: Message, file_id: str, file_name: str, size_bytes: int) -> None:
        case: Optional[Dict[str, Any]] = None
        try:
            if size_bytes > MAX_FILE_BYTES:
                raise InputError("Один файл не должен быть больше 50 МБ")
            case = await runtime.case_for_user(message.chat.id)
            runtime.store.begin_upload(case["id"])
            runtime.service.corrections.cancel(case["id"])
            await runtime._ensure_progress(case["id"], message.chat.id)
            incoming = runtime.store.case_dir(case["id"]) / "incoming"
            incoming.mkdir(parents=True, exist_ok=True)
            path = incoming / (file_id[-24:].replace("/", "_") + "_" + Path(file_name).name)
            await runtime.bot.download(file_id, destination=path)
            artifact = await asyncio.to_thread(
                runtime.service.documents.ingest,
                case["id"],
                ArtifactInput(file_id, file_name, path, size_bytes),
            )
            runtime.store.finish_upload(case["id"])
            await runtime._end_progress(
                case["id"],
                message.chat.id,
                "✅ Файл обработан: " + str(artifact["original_name"]) + ". Всего файлов: "
                + str(runtime.store.artifact_count(case["id"])) + ".",
            )
        except Exception as error:
            if case is not None:
                runtime.store.finish_upload(case["id"], getattr(error, "code", type(error).__name__))
                await runtime._end_progress(case["id"], message.chat.id, "❌ " + user_error(error))
            else:
                await send_with_retry(runtime.bot, message.chat.id, user_error(error))

    @router.message(F.document)
    async def document(message: Message) -> None:
        document = message.document
        if document:
            await save_upload(message, document.file_id, document.file_name or "document", document.file_size or 0)

    @router.message(F.photo)
    async def photo(message: Message) -> None:
        if message.photo:
            photo = message.photo[-1]
            await save_upload(message, photo.file_id, "photo_" + photo.file_unique_id + ".jpg", photo.file_size or 0)

    @router.message(F.text)
    async def answer(message: Message) -> None:
        if not message.text or message.text.startswith("/"):
            return
        case: Optional[Dict[str, Any]] = None
        try:
            case = await runtime.case_for_user(message.chat.id)
            pending = runtime.service.corrections.pending(case["id"])
            if pending:
                runtime.service.corrections.submit(pending["id"], message.chat.id, message.text)
                runtime.worker.notify()
                await runtime._ensure_progress(case["id"], message.chat.id)
                return
            if case["mode"] == "grilling":
                runtime.store.accept_answer_and_enqueue(case["id"], message.chat.id, message.text)
                runtime.worker.notify()
                await runtime._ensure_progress(case["id"], message.chat.id)
                return
            runtime.store.begin_upload(case["id"])
            await runtime._ensure_progress(case["id"], message.chat.id)
            incoming = runtime.store.case_dir(case["id"]) / "incoming"
            incoming.mkdir(parents=True, exist_ok=True)
            path = incoming / ("message_" + str(message.message_id) + ".txt")
            await asyncio.to_thread(path.write_text, message.text, encoding="utf-8")
            artifact = await asyncio.to_thread(
                runtime.service.documents.ingest,
                case["id"],
                ArtifactInput("message:" + str(message.message_id), path.name, path, path.stat().st_size),
            )
            runtime.store.finish_upload(case["id"])
            await runtime._end_progress(case["id"], message.chat.id, "✅ Текст обработан: " + str(artifact["original_name"]) + ".")
        except Exception as error:
            if case is not None and runtime.store.get_case(case["id"]).get("activity") == "uploading":
                runtime.store.finish_upload(case["id"], getattr(error, "code", type(error).__name__))
                await runtime._end_progress(case["id"], message.chat.id, "❌ " + user_error(error))
            else:
                await send_with_retry(runtime.bot, message.chat.id, user_error(error))

    return router


async def run_bot(config: Optional[BotConfig] = None) -> None:
    """Run exactly one poller; launchd is responsible for restarting it."""

    active_config = config or load_config()
    session = AiohttpSession(timeout=20.0)
    bot = Bot(token=active_config.token, session=session)
    runtime = TelegramRuntime(bot, active_config)
    dispatcher = Dispatcher()
    dispatcher.include_router(build_router(runtime))
    async def drain_handlers(**kwargs: Any) -> None:
        # aiogram 3.30 stops polling before shutdown but does not await handlers.
        pending = list(dispatcher._handle_update_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    dispatcher.shutdown.register(drain_handlers)
    await runtime.start()
    try:
        with contextlib.suppress(TelegramNetworkError):
            await bot.set_my_commands(
                [
                    BotCommand(command="new", description="новый кейс"),
                    BotCommand(command="review", description="аудит материалов"),
                    BotCommand(command="grill", description="начать репетицию"),
                    BotCommand(command="stop", description="остановить и получить итог"),
                    BotCommand(command="status", description="состояние и очередь"),
                    BotCommand(command="help", description="инструкция"),
                ]
            )
        await dispatcher.start_polling(bot, polling_timeout=active_config.poll_timeout, close_bot_session=False)
    finally:
        await runtime.stop()
        await bot.session.close()
