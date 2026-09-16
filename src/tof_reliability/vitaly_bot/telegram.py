"""Telegram adapter for the Vitaly simulator's persistent application service."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Dict, Iterable, List, Optional

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatType
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, CallbackQuery, Message
from aiogram.utils.chat_action import ChatActionSender

from .config import load_config
from .documents import MAX_FILE_BYTES
from .learning import public_release_summary
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
    "Ошибочный пункт аудита можно перепроверить кнопкой «Исправить пункт»; общую проблему — кнопкой «Общая обратная связь».\n"
    "/meeting — добавить транскрипт новой встречи в отдельную базу профиля.\n"
    "/learning — увидеть версию контекста и статус своих отправок.\n\n"
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


def format_learning_status(runtime: "TelegramRuntime", chat_id: int, user_id: Optional[int]) -> str:
    package = runtime.store.active_knowledge_package()
    own = runtime.store.own_learning_status(user_id, chat_id)
    control = runtime.store.learning_control()
    lines = [
        "Накопительный контекст",
        "Активная версия: " + str(package["version"]) + " (" + str(package["id"])[:12] + ").",
        "Последнее изменение: " + public_release_summary(package.get("package", {}).get("patch", {})),
        "Автообновление: " + ("приостановлено" if int(control.get("auto_update_paused") or 0) else "включено"),
        "\nВаши отправки:",
    ]
    feedback = own["feedback"]
    meetings = own["meetings"]
    if not feedback and not meetings:
        lines.append("• Пока нет")
        return "\n".join(lines)
    for item in feedback:
        result = item.get("result", {})
        outcome = result.get("learning_outcome")
        label = {
            "case_only": "учтено только в этом кейсе",
            "example": "добавлено в проверочные примеры",
            "candidate": "ожидает проверки для общих правил",
            "insufficient": "недостаточно оснований",
            "received": "ожидает рассмотрения",
            "failed": "не рассмотрено из-за ошибки; можно отправить новую обратную связь",
        }.get(str(item.get("status")), str(item.get("status")))
        if outcome == "enabled":
            label = "включено в общие правила"
        elif outcome == "rejected":
            label = "не включено: проверка кандидата не пройдена"
        elif outcome == "needs_clarification":
            label = "нужно уточнение перед общими правилами"
        elif outcome == "no_change":
            label = "проверено: общее правило не требуется"
        elif outcome == "verification_failed":
            label = "не включено: технический сбой проверки, прежние правила сохранены"
        reason = str(result.get("reason") or "")[:180]
        lines.append("• Обратная связь: " + label + (" — " + reason if reason else ""))
    for item in meetings:
        classification = item.get("classification", {})
        reason = str(classification.get("reason") or "")[:180]
        title = str(item.get("title") or "встреча")
        outcome = classification.get("learning_outcome")
        label = {
            "accepted": "принята как источник профиля",
            "duplicate": "дубль",
            "continuation": "продолжение известной встречи",
            "other_person": "другой Виталий",
            "needs_clarification": "нужно уточнение атрибуции",
            "failed": "не рассмотрена из-за ошибки",
        }.get(str(item.get("status")), str(item.get("status")))
        if outcome == "enabled":
            label = "включена в обновление профиля"
        elif outcome == "rejected":
            label = "не включена: проверка кандидата не пройдена"
        elif outcome == "needs_clarification":
            label = "нужно уточнение перед обновлением профиля"
        elif outcome == "no_change":
            label = "проверена: изменение профиля не требуется"
        elif outcome == "verification_failed":
            label = "не включена: технический сбой проверки, прежний профиль сохранён"
        lines.append("• " + title + ": " + label + (" — " + reason if reason else ""))
    return "\n".join(lines)


def _next_meeting_prompt(meeting: Dict[str, Any]) -> str:
    step = str(meeting.get("step"))
    if step == "date":
        return "Укажите дату встречи в формате ГГГГ-ММ-ДД."
    if step == "title":
        return "Укажите короткое название встречи."
    if step == "vitaly":
        return "Как среди говорящих обозначен нужный Виталий? Не угадывайте: укажите имя, роль или метку из транскрипта."
    if step == "source":
        return "Отправьте транскрипт: текст сообщением или TXT, MD, DOCX либо текстовый PDF. Аудио в этот режим не принимается."
    if step == "clarification":
        return "Нужно уточнить, как в транскрипте распознать нужного Виталия. Ответьте одним сообщением или /cancel."
    return "Встреча принята в обработку."


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
        self.store = VitalyStore(config.state_dir, config.knowledge_dir)
        self.profile = load_default_profile()
        self.service = VitalyService(self.store, self.profile, config)
        self.worker = FifoWorker(self.store, self.service, self._completed, self._failed)
        self._progress_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._progress_text: Dict[str, str] = {}
        self._learning_typing_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._learning_scheduler_task: Optional[asyncio.Task[Any]] = None

    async def start(self) -> None:
        await self.worker.start()
        for case in self.store.active_cases():
            await self._ensure_progress(case["id"], int(case["chat_id"]))
        for job in self.store.active_learning_jobs():
            self._ensure_learning_typing(job)
        self._learning_scheduler_task = asyncio.create_task(
            self._learning_scheduler(), name="vitaly-learning-scheduler"
        )

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
        if self._learning_scheduler_task is not None:
            self._learning_scheduler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._learning_scheduler_task
        for task in list(self._learning_typing_tasks.values()):
            task.cancel()
        for task in list(self._progress_tasks.values()):
            task.cancel()
        for task in list(self._learning_typing_tasks.values()):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for task in list(self._progress_tasks.values()):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self.worker.stop()

    async def _learning_scheduler(self) -> None:
        """Schedule at most one low-priority cycle each configured interval."""

        try:
            while True:
                if self.store.due_learning_cycle(self.config.learning_cycle_seconds):
                    job = self.store.queue_learning_cycle()
                    if job is not None:
                        self.worker.notify()
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A scheduler failure must never stop polling or the user queue.
            return

    async def _learning_typing_loop(self, job_id: str, chat_id: int) -> None:
        if not chat_id:
            return
        try:
            async with ChatActionSender.typing(bot=self.bot, chat_id=chat_id, initial_sleep=0):
                while True:
                    try:
                        job = self.store.get_learning_job(job_id)
                    except StateError:
                        return
                    if job.get("status") not in {"queued", "running"}:
                        return
                    await asyncio.sleep(2)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    def _ensure_learning_typing(self, job: Dict[str, Any]) -> None:
        if not int(job.get("chat_id") or 0):
            return
        job_id = str(job["id"])
        current = self._learning_typing_tasks.get(job_id)
        if current is None or current.done():
            self._learning_typing_tasks[job_id] = asyncio.create_task(
                self._learning_typing_loop(job_id, int(job["chat_id"])), name="vitaly-learning-typing-" + job_id[:8]
            )

    async def _with_typing(self, chat_id: int, operation: Awaitable[Any]) -> Any:
        """Run a local preparation while a cosmetic typing indicator is best-effort."""

        task = asyncio.ensure_future(operation)
        try:
            async with ChatActionSender.typing(bot=self.bot, chat_id=chat_id, initial_sleep=0):
                return await task
        except asyncio.CancelledError:
            task.cancel()
            raise
        except (TelegramBadRequest, TelegramNetworkError):
            return await task
        except Exception:
            # If the indicator itself or the operation fails, await the latter
            # so its real user-facing error is preserved.
            return await task

    async def _end_learning_typing(self, job_id: str) -> None:
        task = self._learning_typing_tasks.pop(job_id, None)
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

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
        if job.get("scope") == "learning":
            await self._completed_learning(job, outcome)
            return
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

    async def _completed_learning(self, job: Dict[str, Any], outcome: JobOutcome) -> None:
        await self._end_learning_typing(str(job["id"]))
        chat_id = int(job.get("chat_id") or 0)
        if outcome.kind == "feedback_analyzed" and chat_id:
            assessment = outcome.payload["assessment"]
            label = {
                "case_only": "Учтено только в этом кейсе.",
                "example": "Добавлено в проверочные примеры.",
                "candidate": "Принято как кандидат в общие правила; он пройдёт отдельную проверку.",
                "insufficient": "Недостаточно оснований, чтобы менять общие правила.",
            }[assessment["result"]]
            await send_with_retry(self.bot, chat_id, "Общая обратная связь рассмотрена. " + label + "\n" + assessment["reason"])
            return
        if outcome.kind == "meeting_analyzed" and chat_id:
            result = outcome.payload["classification"]
            messages = {
                "accepted": "Встреча принята как источник для следующего контролируемого обновления профиля.",
                "duplicate": "Встреча определена как дубль и не будет добавлена повторно.",
                "continuation": "Встреча определена как продолжение уже известной записи и не будет учтена как независимый паттерн.",
                "other_person": "Запись относится к другому Виталию и не будет использоваться в профиле.",
                "needs_clarification": "Недостаточно надёжно определён нужный Виталий; запись не будет использоваться без уточнения.",
            }
            await send_with_retry(self.bot, chat_id, messages[result["status"]] + "\n" + result["reason"])
            return
        # A package release is global and may derive from several users.  It is
        # intentionally visible through each author's /learning only, not pushed
        # into unrelated chats.


    async def _failed(self, job: Dict[str, Any], error: Exception) -> None:
        if job.get("scope") == "learning":
            await self._end_learning_typing(str(job["id"]))
            chat_id = int(job.get("chat_id") or 0)
            if chat_id:
                subject = "обратную связь" if job.get("kind") == "feedback_analyze" else "транскрипт встречи"
                await send_with_retry(self.bot, chat_id, "Не удалось рассмотреть " + subject + ": " + user_error(error))
            return
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

    # Cases are deliberately isolated by private chat.  Accepting updates in a
    # group would otherwise let different participants share one case and its
    # uploaded materials, which contradicts the learning-domain boundary.
    @router.message(F.chat.type != ChatType.PRIVATE)
    async def reject_non_private_message(message: Message) -> None:
        await send_with_retry(
            runtime.bot,
            message.chat.id,
            "Этот бот работает только в личном чате: так материалы и обратная связь не смешиваются между пользователями.",
        )

    @router.callback_query(F.message.chat.type != ChatType.PRIVATE)
    async def reject_non_private_callback(callback: CallbackQuery) -> None:
        await callback.answer("Откройте бота в личном чате.", show_alert=True)

    correction_ui.register(router, runtime)

    def user_id(message: Message) -> Optional[int]:
        return message.from_user.id if message.from_user else None

    def assert_not_collecting_learning_input(message: Message) -> None:
        if runtime.store.pending_feedback(message.chat.id, user_id(message)):
            raise StateError("Сначала отправьте или отмените пояснение к общей обратной связи командой /cancel.")
        if runtime.store.pending_meeting(message.chat.id, user_id(message)):
            raise StateError("Сначала завершите или отмените загрузку встречи командой /cancel.")

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
            assert_not_collecting_learning_input(message)
            case = runtime.service.new_case(message.chat.id)
            await send_with_retry(runtime.bot, message.chat.id, "Новый кейс создан: " + case["id"][:8] + ". Отправьте материалы.")
        except Exception as error:
            await send_with_retry(runtime.bot, message.chat.id, user_error(error))

    @router.message(Command("review"))
    async def review(message: Message) -> None:
        try:
            assert_not_collecting_learning_input(message)
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
            assert_not_collecting_learning_input(message)
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

    @router.message(Command("meeting"))
    async def meeting(message: Message) -> None:
        try:
            intake = runtime.store.begin_meeting(message.chat.id, user_id(message))
            await send_with_retry(runtime.bot, message.chat.id, "Новый транскрипт будет храниться отдельно от текущего кейса. " + _next_meeting_prompt(intake))
        except Exception as error:
            await send_with_retry(runtime.bot, message.chat.id, user_error(error))

    @router.message(Command("learning"))
    async def learning(message: Message) -> None:
        try:
            await send_with_retry(runtime.bot, message.chat.id, format_learning_status(runtime, message.chat.id, user_id(message)))
        except Exception as error:
            await send_with_retry(runtime.bot, message.chat.id, user_error(error))

    @router.message(Command("cancel"))
    async def cancel(message: Message) -> None:
        try:
            cancelled = runtime.store.cancel_feedback(message.chat.id, user_id(message))
            cancelled = runtime.store.cancel_meeting(message.chat.id, user_id(message)) or cancelled
            if cancelled:
                await send_with_retry(runtime.bot, message.chat.id, "Ввод отменён. Кейс и аудит не изменены.")
            else:
                await send_with_retry(runtime.bot, message.chat.id, "Нет ожидающего пояснения или загрузки встречи.")
        except Exception as error:
            await send_with_retry(runtime.bot, message.chat.id, user_error(error))

    def assert_operator(message: Message) -> int:
        current = user_id(message)
        if current is None or current not in runtime.config.operator_user_ids:
            raise StateError("Эта служебная команда недоступна")
        return current

    @router.message(Command("learning_pause"))
    async def learning_pause(message: Message) -> None:
        try:
            runtime.store.set_learning_paused(True, actor_user_id=assert_operator(message))
            await send_with_retry(runtime.bot, message.chat.id, "Автоматическое обновление пакетов знаний приостановлено.")
        except Exception as error:
            await send_with_retry(runtime.bot, message.chat.id, user_error(error))

    @router.message(Command("learning_resume"))
    async def learning_resume(message: Message) -> None:
        try:
            runtime.store.set_learning_paused(False, actor_user_id=assert_operator(message))
            runtime.worker.notify()
            await send_with_retry(runtime.bot, message.chat.id, "Автоматическое обновление пакетов знаний включено.")
        except Exception as error:
            await send_with_retry(runtime.bot, message.chat.id, user_error(error))

    @router.message(Command("learning_rollback"))
    async def learning_rollback(message: Message) -> None:
        try:
            actor = assert_operator(message)
            parts = (message.text or "").split(maxsplit=1)
            if len(parts) != 2:
                raise InputError("Укажите идентификатор пакета: /learning_rollback kp-...")
            package = runtime.store.rollback_knowledge_package(parts[1].strip(), actor_user_id=actor)
            await send_with_retry(runtime.bot, message.chat.id, "Включён пакет отката " + str(package["version"]) + ". Новые кейсы получат его; старые не изменены.")
        except Exception as error:
            await send_with_retry(runtime.bot, message.chat.id, user_error(error))

    async def reject_feedback_file_if_pending(message: Message) -> bool:
        if runtime.store.pending_feedback(message.chat.id, user_id(message)) is not None:
            await send_with_retry(
                runtime.bot,
                message.chat.id,
                "Сейчас ожидается текст пояснения к общей обратной связи. Файл не добавлен к материалам; отправьте пояснение или /cancel.",
            )
            return True
        if runtime.store.feedback_in_flight(message.chat.id, user_id(message)) is not None:
            await send_with_retry(
                runtime.bot,
                message.chat.id,
                "Обратная связь уже принята и рассматривается. Повторный файл не добавлен к материалам. Статус: /learning.",
            )
            return True
        return False

    async def save_upload(message: Message, file_id: str, file_name: str, size_bytes: int) -> None:
        case: Optional[Dict[str, Any]] = None
        try:
            if await reject_feedback_file_if_pending(message):
                return
            if size_bytes > MAX_FILE_BYTES:
                raise InputError("Один файл не должен быть больше 50 МБ")
            case = await runtime.case_for_user(message.chat.id)
            runtime.store.ensure_free_space(runtime.config.min_free_bytes, runtime.store.case_root)
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

    async def save_meeting_upload(message: Message, file_id: str, file_name: str, size_bytes: int) -> bool:
        """Consume an upload only when this chat has a transcript intake open."""

        if await reject_feedback_file_if_pending(message):
            return True
        intake = runtime.store.pending_meeting(message.chat.id, user_id(message))
        if intake is None:
            if runtime.store.meeting_in_flight(message.chat.id, user_id(message)) is not None:
                await send_with_retry(
                    runtime.bot,
                    message.chat.id,
                    "Транскрипт уже принят и проверяется. Повторный файл не добавлен ни к встрече, ни к текущему кейсу. Статус: /learning.",
                )
                return True
            return False
        try:
            if intake.get("step") != "source":
                raise StateError(_next_meeting_prompt(intake))
            if size_bytes > MAX_FILE_BYTES:
                raise InputError("Один файл не должен быть больше 50 МБ")
            runtime.store.ensure_free_space(runtime.config.min_free_bytes)
            await send_with_retry(runtime.bot, message.chat.id, "⏳ Загружаю и разбираю транскрипт встречи…")
            incoming = runtime.store.knowledge_root / "meetings" / str(intake["id"]) / "incoming"
            incoming.mkdir(parents=True, exist_ok=True)
            path = incoming / (file_id[-24:].replace("/", "_") + "_" + Path(file_name).name)
            await runtime.bot.download(file_id, destination=path)
            prepared = await runtime._with_typing(
                message.chat.id,
                asyncio.to_thread(
                    runtime.service.documents.prepare_meeting_source,
                    str(intake["id"]),
                    ArtifactInput(file_id, file_name, path, size_bytes),
                    min_free_bytes=runtime.config.min_free_bytes,
                ),
            )
            meeting, job = runtime.store.accept_meeting_source(
                str(intake["id"]),
                source_name=str(prepared["source_name"]),
                source_path=Path(prepared["source_path"]),
                normalized_path=Path(prepared["normalized_path"]),
                source_sha256=str(prepared["source_sha256"]),
                normalized_sha256=str(prepared["normalized_sha256"]),
            )
            if job is None:
                await send_with_retry(runtime.bot, message.chat.id, "✅ Транскрипт сохранён, но он является дублем уже загруженной встречи. В профиль повторно не попадёт.")
            else:
                runtime.worker.notify()
                runtime._ensure_learning_typing(job)
                await send_with_retry(runtime.bot, message.chat.id, "⏳ Транскрипт сохранён. Проверяю атрибуцию Виталия и возможные дубли…")
            return True
        except Exception as error:
            await send_with_retry(runtime.bot, message.chat.id, "❌ " + user_error(error))
            return True

    @router.message(F.document)
    async def document(message: Message) -> None:
        document = message.document
        if document:
            if not await save_meeting_upload(message, document.file_id, document.file_name or "document", document.file_size or 0):
                await save_upload(message, document.file_id, document.file_name or "document", document.file_size or 0)

    @router.message(F.photo)
    async def photo(message: Message) -> None:
        if message.photo:
            if await reject_feedback_file_if_pending(message):
                return
            photo = message.photo[-1]
            intake = runtime.store.pending_meeting(message.chat.id, user_id(message))
            if intake is not None:
                await send_with_retry(runtime.bot, message.chat.id, "В режиме /meeting фото не принимаются. Нужен TXT, MD, DOCX или текстовый PDF.")
                return
            if runtime.store.meeting_in_flight(message.chat.id, user_id(message)) is not None:
                await send_with_retry(runtime.bot, message.chat.id, "Транскрипт уже принят и проверяется. Повторное фото не добавлено к материалам. Статус: /learning.")
                return
            await save_upload(message, photo.file_id, "photo_" + photo.file_unique_id + ".jpg", photo.file_size or 0)

    @router.message(F.text)
    async def answer(message: Message) -> None:
        if not message.text or message.text.startswith("/"):
            return
        case: Optional[Dict[str, Any]] = None
        try:
            pending_feedback = runtime.store.pending_feedback(message.chat.id, user_id(message))
            if pending_feedback:
                job = runtime.store.submit_feedback(str(pending_feedback["id"]), message.text)
                runtime.worker.notify()
                runtime._ensure_learning_typing(job)
                await send_with_retry(runtime.bot, message.chat.id, "⏳ Обратная связь принята. Сверяю её с аудитом и основаниями…")
                return
            pending_meeting = runtime.store.pending_meeting(message.chat.id, user_id(message))
            if pending_meeting:
                if runtime.store.is_duplicate_meeting_metadata(
                    str(pending_meeting["id"]), message.text, message.message_id
                ):
                    await send_with_retry(
                        runtime.bot,
                        message.chat.id,
                        "Это сообщение уже принято. " + _next_meeting_prompt(pending_meeting),
                    )
                    return
                step = str(pending_meeting.get("step"))
                if step in {"date", "title", "vitaly"}:
                    updated = runtime.store.submit_meeting_metadata(
                        str(pending_meeting["id"]), message.text, message.message_id
                    )
                    await send_with_retry(runtime.bot, message.chat.id, _next_meeting_prompt(updated))
                    return
                if step == "clarification":
                    job = runtime.store.submit_meeting_clarification(str(pending_meeting["id"]), message.text)
                    runtime.worker.notify()
                    runtime._ensure_learning_typing(job)
                    await send_with_retry(runtime.bot, message.chat.id, "⏳ Уточнение принято. Повторно проверяю атрибуцию встречи…")
                    return
                if step == "source":
                    runtime.store.ensure_free_space(runtime.config.min_free_bytes, runtime.store.knowledge_root)
                    incoming = runtime.store.knowledge_root / "meetings" / str(pending_meeting["id"]) / "incoming"
                    incoming.mkdir(parents=True, exist_ok=True)
                    path = incoming / ("message_" + str(message.message_id) + ".txt")
                    await asyncio.to_thread(path.write_text, message.text, encoding="utf-8")
                    await send_with_retry(runtime.bot, message.chat.id, "⏳ Сохраняю и разбираю текст транскрипта…")
                    prepared = await runtime._with_typing(
                        message.chat.id,
                        asyncio.to_thread(
                            runtime.service.documents.prepare_meeting_source,
                            str(pending_meeting["id"]),
                            ArtifactInput("message:" + str(message.message_id), path.name, path, path.stat().st_size),
                            min_free_bytes=runtime.config.min_free_bytes,
                        ),
                    )
                    _meeting, job = runtime.store.accept_meeting_source(
                        str(pending_meeting["id"]),
                        source_name=str(prepared["source_name"]),
                        source_path=Path(prepared["source_path"]),
                        normalized_path=Path(prepared["normalized_path"]),
                        source_sha256=str(prepared["source_sha256"]),
                        normalized_sha256=str(prepared["normalized_sha256"]),
                    )
                    if job is None:
                        await send_with_retry(runtime.bot, message.chat.id, "✅ Текст встречи сохранён, но он является дублем уже известной записи.")
                    else:
                        runtime.worker.notify()
                        runtime._ensure_learning_typing(job)
                        await send_with_retry(runtime.bot, message.chat.id, "⏳ Текст встречи сохранён. Проверяю атрибуцию Виталия и дубли…")
                    return
                raise StateError(_next_meeting_prompt(pending_meeting))
            if runtime.store.feedback_in_flight(message.chat.id, user_id(message)) is not None:
                await send_with_retry(
                    runtime.bot,
                    message.chat.id,
                    "Обратная связь уже принята и рассматривается. Повторное сообщение не добавлено к материалам. Статус: /learning.",
                )
                return
            if runtime.store.meeting_in_flight(message.chat.id, user_id(message)) is not None:
                await send_with_retry(
                    runtime.bot,
                    message.chat.id,
                    "Транскрипт уже принят и проверяется. Повторное сообщение не добавлено к материалам. Статус: /learning.",
                )
                return
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
            runtime.store.ensure_free_space(runtime.config.min_free_bytes, runtime.store.case_root)
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
                    BotCommand(command="meeting", description="добавить транскрипт встречи"),
                    BotCommand(command="learning", description="версия контекста и мои отправки"),
                    BotCommand(command="cancel", description="отменить ввод"),
                    BotCommand(command="help", description="инструкция"),
                ]
            )
        await dispatcher.start_polling(bot, polling_timeout=active_config.poll_timeout, close_bot_session=False)
    finally:
        await runtime.stop()
        await bot.session.close()
