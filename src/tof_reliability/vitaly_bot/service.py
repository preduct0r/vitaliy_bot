"""Application service, structured prompts, and one-at-a-time persistent queue."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from .codex import TRANSIENT_CODES, CodexRunner
from .documents import DocumentProcessor
from .evidence import EvidenceProvider, RemoteLedgerEvidenceProvider
from .models import BotConfig, EvidenceError, ModelError, ModelResult, StateError
from .profile import VitalyProfile
from .storage import VitalyStore
from .corrections import Corrections, correct_point


class StructuredRunner(Protocol):
    async def run(self, prompt: str, schema: Dict[str, Any], image_paths: Optional[Sequence[Any]] = None) -> ModelResult:
        ...


def _object(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": required}


QUESTION_PROPERTIES = {
    "text": {"type": "string"},
    "topic": {"type": "string"},
    "likelihood": {"type": "integer", "minimum": 1, "maximum": 5},
    "severity": {"type": "integer", "minimum": 1, "maximum": 5},
    "trigger": {"type": "string"},
    "material_reference": {"type": "string"},
    "expected_answer": {"type": "string"},
    "expected_answer_evidence_level": {"type": "string", "enum": ["ledger", "material", "none"]},
    "expected_answer_references": {"type": "array", "items": {"type": "string"}},
    "followups": {"type": "array", "items": {"type": "string"}},
    "profile_evidence": {"type": "array", "items": {"type": "string"}},
    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
}


def review_schema(profile: VitalyProfile) -> Dict[str, Any]:
    question = _object(dict(QUESTION_PROPERTIES, topic={"type": "string", "enum": profile.topic_ids}), list(QUESTION_PROPERTIES))
    gap = _object(
        {"title": {"type": "string"}, "detail": {"type": "string"}, "material_reference": {"type": "string"}},
        ["title", "detail", "material_reference"],
    )
    return _object(
        {
            "readiness_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "verdict": {"type": "string"},
            "critical_gaps": {"type": "array", "items": gap},
            "contradictions": {"type": "array", "items": gap},
            "questions": {"type": "array", "minItems": 1, "maxItems": 12, "items": question},
            "prep_checklist": {"type": "array", "items": {"type": "string"}},
        },
        ["readiness_score", "verdict", "critical_gaps", "contradictions", "questions", "prep_checklist"],
    )


MATERIAL_MAP_SCHEMA = _object(
    {
        "summary": {"type": "string"},
        "claims": {"type": "array", "items": {"type": "string"}},
        "numbers": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
    },
    ["summary", "claims", "numbers", "gaps"],
)

FEEDBACK_SCHEMA = _object(
    {
        "verdict": {"type": "string"},
        "scores": _object(
            {
                "evidence": {"type": "integer", "minimum": 0, "maximum": 5},
                "specificity": {"type": "integer", "minimum": 0, "maximum": 5},
                "numbers": {"type": "integer", "minimum": 0, "maximum": 5},
                "ownership": {"type": "integer", "minimum": 0, "maximum": 5},
            },
            ["evidence", "specificity", "numbers", "ownership"],
        ),
        "challenge": {"type": "string"},
        "followup_question": {"type": "string"},
        "answer_status": {"type": "string", "enum": ["correct", "partial", "incorrect", "not_confirmed"]},
        "correct_answer": {"type": "string"},
        "evidence_level": {"type": "string", "enum": ["ledger", "material", "none"]},
        "evidence_references": {"type": "array", "items": {"type": "string"}},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
    },
    [
        "verdict",
        "scores",
        "challenge",
        "followup_question",
        "answer_status",
        "correct_answer",
        "evidence_level",
        "evidence_references",
        "missing_evidence",
    ],
)

FINISH_SCHEMA = _object(
    {
        "overall_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "verdict": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weak_answers": {"type": "array", "items": {"type": "string"}},
        "before_meeting": {"type": "array", "items": {"type": "string"}},
    },
    ["overall_score", "verdict", "strengths", "weak_answers", "before_meeting"],
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _question_with_position(question: Dict[str, Any], index: int) -> Dict[str, Any]:
    value = dict(question)
    value["planned_index"] = index
    value["is_followup"] = False
    return value


def _validate_review(value: Dict[str, Any], profile: VitalyProfile) -> Dict[str, Any]:
    required = {"readiness_score", "verdict", "critical_gaps", "contradictions", "questions", "prep_checklist"}
    if required - set(value) or not isinstance(value.get("questions"), list):
        raise ModelError("model_output_invalid_schema", "Не хватает полей отчёта")
    if not 0 <= value["readiness_score"] <= 100 or not value["questions"]:
        raise ModelError("model_output_invalid_schema", "Неверная оценка готовности или пустые вопросы")
    for question in value["questions"]:
        if set(QUESTION_PROPERTIES) - set(question) or question["topic"] not in profile.topic_ids:
            raise ModelError("model_output_invalid_schema", "Вопрос не соответствует профилю")
    return value


@dataclass(frozen=True)
class JobOutcome:
    """A completed queue item to be rendered by the Telegram adapter."""

    kind: str
    case_id: str
    payload: Dict[str, Any]


class VitalyService:
    """The non-Telegram part of reviews, drills, and final scorecards."""

    def __init__(
        self,
        store: VitalyStore,
        profile: VitalyProfile,
        config: BotConfig,
        runner: Optional[StructuredRunner] = None,
        evidence_provider: Optional[EvidenceProvider] = None,
    ) -> None:
        self.store = store
        self.profile = profile
        self.corrections = Corrections(store)
        self.config = config
        self.documents = DocumentProcessor(store)
        self.runner: StructuredRunner = runner or CodexRunner(config)
        self.evidence_provider: EvidenceProvider = evidence_provider or RemoteLedgerEvidenceProvider(config)

    def new_case(self, chat_id: int) -> Dict[str, Any]:
        return self.store.create_case(chat_id, self.profile.version)

    def active_case(self, chat_id: int) -> Dict[str, Any]:
        return self.store.active_case(chat_id, self.profile.version)

    def case_for_user(self, chat_id: int) -> Tuple[Dict[str, Any], Optional[str]]:
        """Activate a user context and report whether the previous one expired."""

        return self.store.activate_case_for_user(chat_id, self.profile.version, self.config.context_ttl_days)

    async def _model(
        self,
        case_id: str,
        job_id: str,
        kind: str,
        prompt: str,
        schema: Dict[str, Any],
        images: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        stage = {
            "material_map": "Обрабатываю материалы",
            "review": "Модель готовит аудит",
            "grill_feedback": "Модель проверяет ответ",
            "finish": "Модель готовит итог репетиции",
            "review_correction": "Перепроверяю выбранный пункт",
        }.get(kind, "Модель размышляет")
        self.store.set_activity(case_id, "thinking", stage)
        last_error: Optional[ModelError] = None
        for attempt in range(2):
            run_id = self.store.create_model_run(
                case_id, job_id, kind, self.config.model, self.config.reasoning_effort
            )
            try:
                result = await self.runner.run(prompt, schema, images)
            except ModelError as error:
                self.store.finish_model_run(run_id, error.code + (":" + error.detail[-1000:] if error.detail else ""))
                last_error = error
                if error.code in TRANSIENT_CODES and attempt == 0:
                    continue
                raise
            self.store.finish_model_run(run_id)
            return result.value
        raise last_error or ModelError("codex_transient")

    def _evidence_fingerprint(self, case_id: str) -> str:
        digest = hashlib.sha256()
        for artifact in self.store.artifacts(case_id):
            digest.update(str(artifact["sha256"]).encode("ascii", errors="ignore"))
            digest.update(b"\0")
        return digest.hexdigest()

    async def _evidence_pack(self, case_id: str) -> Dict[str, Any]:
        case = self.store.get_case(case_id)
        fingerprint = self._evidence_fingerprint(case_id)
        if case["evidence_pack"] and case.get("evidence_fingerprint") == fingerprint:
            return case["evidence_pack"]
        prepared = self.documents.prepared(case_id)
        if not prepared:
            raise StateError("Сначала отправьте текст или файл")
        self.store.set_activity(case_id, "checking_ledger", "Проверяю опубликованный ledger ТОФ")
        material_text = "\n\n".join(item.text_reference + "\n" + item.text for item in prepared)
        try:
            pack = await self.evidence_provider.build(material_text, [str(item["sha256"]) for item in self.store.artifacts(case_id)])
        except EvidenceError:
            raise
        except Exception as error:
            raise EvidenceError("tof_unavailable", str(error)) from error
        self.store.set_evidence_pack(case_id, pack, fingerprint)
        return pack

    def _base_prompt(self) -> str:
        return (
            "Ты — " + self.profile.simulator_label + ". Это симулятор, а не реальный человек. "
            "Не имитируй дословные реплики и не приписывай ему мнения вне профиля.\n"
            "Профиль поведения:\n" + _json(self.profile.prompt_context())
        )

    async def _material_map(self, case_id: str, job_id: str) -> Dict[str, Any]:
        case = self.store.get_case(case_id)
        if case["material_map"]:
            return case["material_map"]
        chunks = self.documents.material_chunks(case_id)
        if not chunks:
            raise StateError("Сначала отправьте текст или файл")
        if len(chunks) == 1:
            material_map = {"mode": "raw", "content": chunks[0]}
            self.store.set_case_material_map(case_id, material_map)
            return material_map
        maps = []
        for number, chunk in enumerate(chunks, start=1):
            prompt = (
                self._base_prompt()
                + "\nСоставь нейтральную карту этого фрагмента для последующей проверки: утверждения, числа, пробелы. "
                + "Не придумывай фактов.\nФрагмент " + str(number) + ":\n" + chunk
            )
            maps.append(await self._model(case_id, job_id, "material_map", prompt, MATERIAL_MAP_SCHEMA))
        material_map = {"mode": "summaries", "fragments": maps}
        self.store.set_case_material_map(case_id, material_map)
        return material_map

    async def review_case(self, case_id: str, job_id: str) -> Dict[str, Any]:
        case = self.store.get_case(case_id)
        if case["review"]:
            return self.corrections.effective(case_id)
        evidence_pack = await self._evidence_pack(case_id)
        self.store.set_activity(case_id, "processing_materials", "Собираю карту утверждений и чисел")
        material_map = await self._material_map(case_id, job_id)
        prompt = (
            self._base_prompt()
            + "\nПроверь подготовленные материалы перед встречей. Сначала назови только проверяемые пробелы и противоречия. "
            + "Затем сформируй до 12 наиболее вероятных неудобных вопросов. Для topic выбери один из topic_id профиля. "
            + "В profile_evidence указывай только идентификаторы встреч из профиля. "
            + "Для expected_answer используй только evidence-пакет с provenance=resolved или конкретный фрагмент материалов. "
            + "Если факта нет, expected_answer должен честно сказать, что подтверждённого ответа нет, а expected_answer_evidence_level выбрать none. "
            + "Ответ должен быть на русском и строго соответствовать схеме.\nМатериалы:\n"
            + _json(material_map)
            + "\nEvidence-пакет текущего опубликованного поколения:\n"
            + _json(evidence_pack)
        )
        review = _validate_review(
            await self._model(
                case_id, job_id, "review", prompt, review_schema(self.profile), self.documents.selected_images(case_id)
            ),
            self.profile,
        )
        self.store.set_review(case_id, review)
        return review

    def _next_planned_question(self, review: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        questions = review.get("questions", [])
        if index >= len(questions):
            return None
        value = _question_with_position(questions[index], index)
        value["planned_total"] = len(questions)
        return value

    async def start_grill(self, case_id: str, job_id: str) -> JobOutcome:
        case = self.store.get_case(case_id)
        if case.get("stop_requested"):
            return JobOutcome("grill_stopped", case_id, {})
        review = await self.review_case(case_id, job_id)
        review = self.corrections.effective(case_id, rehearsal=True)
        case = self.store.get_case(case_id)
        if case.get("stop_requested"):
            return JobOutcome("grill_stopped", case_id, {})
        question = self._next_planned_question(review, 0)
        if question is None:
            raise StateError("После исправлений не осталось вопросов для репетиции. Откройте актуальный отчёт.")
        self.store.clear_turns(case_id)
        self.store.set_question(case_id, question, 0, "grilling")
        return JobOutcome("grill_question", case_id, {"question": question, "review": review})

    def store_answer(self, case_id: str, answer_text: str) -> Dict[str, Any]:
        case = self.store.get_case(case_id)
        question = case["current_question"]
        if case["mode"] != "grilling" or not question:
            raise StateError("Сначала запустите репетицию командой /grill")
        if not answer_text.strip():
            raise StateError("Нужен текстовый ответ на вопрос")
        return self.store.add_turn(case_id, question, answer_text.strip())

    async def evaluate_turn(self, case_id: str, job_id: str, turn_id: str) -> JobOutcome:
        case = self.store.get_case(case_id)
        turn = self.store.get_turn(turn_id)
        evidence_pack = await self._evidence_pack(case_id)
        material_map = await self._material_map(case_id, job_id)
        prompt = (
            self._base_prompt()
            + "\nОцени ответ сотрудника на вопрос перед встречей. Будь прямым и коротким. "
            + "Если ответ не закрывает главный риск, сформулируй ровно один короткий followup_question; иначе оставь его пустой строкой. "
            + "При partial, incorrect или not_confirmed обязательно сформулируй correct_answer. Он может опираться только на evidence-пакет "
            + "с provenance=resolved либо на явный фрагмент материала. Если такого основания нет, напиши: «Подтверждённого ответа в материалах и текущем ledger нет» "
            + "и перечисли недостающие данные в missing_evidence.\n"
            + "Материалы:\n" + _json(material_map)
            + "\nEvidence-пакет:\n" + _json(evidence_pack)
            + "\nВопрос:\n" + _json(turn["question"])
            + "\nОтвет сотрудника:\n" + turn["answer_text"]
            + "\nПринятые исправления аудита (не повторяй снятые претензии):\n" + self.corrections.context(case_id)
        )
        feedback = await self._model(case_id, job_id, "grill_feedback", prompt, FEEDBACK_SCHEMA)
        self.store.set_turn_feedback(turn_id, feedback)
        case = self.store.get_case(case_id)
        current = turn["question"]
        review = self.corrections.effective(case_id, rehearsal=True)
        planned_index = int(current.get("planned_index", case["planned_question_index"]))
        if case.get("stop_requested"):
            self.store.set_question(case_id, None, planned_index, "stopping")
            return JobOutcome("grill_feedback", case_id, {"feedback": feedback, "next_question": None, "stopped": True})
        followup = feedback["followup_question"].strip()
        if followup and not current.get("is_followup"):
            next_question = dict(current)
            next_question.update({"text": followup, "is_followup": True, "planned_index": planned_index})
            next_index = planned_index
        else:
            next_index = planned_index + 1
            next_question = self._next_planned_question(review, next_index)
        self.store.set_question(case_id, next_question, next_index, "grilling" if next_question else "ready")
        return JobOutcome("grill_feedback", case_id, {"feedback": feedback, "next_question": next_question})

    async def finish_grill(self, case_id: str, job_id: str) -> JobOutcome:
        case = self.store.get_case(case_id)
        turns = self.store.turns(case_id)
        if not turns:
            raise StateError("Для итогового разбора нужен хотя бы один ответ в репетиции")
        prompt = (
            self._base_prompt()
            + "\nСделай итоговый разбор репетиции. Оцени доказательность, конкретность, владение цифрами, "
            + "причинно-следственную связь и готовность назвать владельца и срок. Не добавляй фактов.\n"
            + "Актуальный аудит:\n" + _json(self.corrections.effective(case_id))
            + "\nПринятые исправления (не повторяй снятые претензии):\n" + self.corrections.context(case_id)
            + "\nХоды репетиции:\n" + _json(turns)
        )
        result = await self._model(case_id, job_id, "finish", prompt, FINISH_SCHEMA)
        self.store.set_question(case_id, None, int(case["planned_question_index"]), "finished")
        self.store.set_stop_requested(case_id, False)
        return JobOutcome("finish", case_id, result)

    async def execute(self, job: Dict[str, Any]) -> JobOutcome:
        kind = job["kind"]
        if kind == "review_correction":
            result = await correct_point(self, job)
            return JobOutcome("review_correction", job["case_id"], result)
        if kind == "review":
            review = await self.review_case(job["case_id"], job["id"])
            return JobOutcome("review", job["case_id"], review)
        if kind == "grill_start":
            return await self.start_grill(job["case_id"], job["id"])
        if kind == "grill_answer":
            return await self.evaluate_turn(job["case_id"], job["id"], str(job["payload"]["turn_id"]))
        if kind == "finish":
            return await self.finish_grill(job["case_id"], job["id"])
        raise StateError("Неизвестный тип задачи")


CompletionHandler = Callable[[Dict[str, Any], JobOutcome], Awaitable[None]]
FailureHandler = Callable[[Dict[str, Any], Exception], Awaitable[None]]


class FifoWorker:
    """One persistent FIFO consumer.  Queue state lives in SQLite, not RAM."""

    def __init__(self, store: VitalyStore, service: VitalyService, completed: CompletionHandler, failed: FailureHandler) -> None:
        self.store = store
        self.service = service
        self.completed = completed
        self.failed = failed
        self._wake = asyncio.Event()
        self._task: Optional[asyncio.Task[Any]] = None
        self._stopping = False

    async def start(self) -> None:
        self.store.recover_jobs()
        self._task = asyncio.create_task(self._loop(), name="vitaly-bot-model-worker")
        self._wake.set()

    def notify(self) -> None:
        self._wake.set()

    async def stop(self) -> None:
        self._stopping = True
        self._wake.set()
        if self._task:
            await self._task

    async def _loop(self) -> None:
        while not self._stopping:
            job = self.store.next_queued_job()
            if job is None:
                self._wake.clear()
                await self._wake.wait()
                continue
            try:
                self.store.mark_job_running(job["id"])
                outcome = await self.service.execute(job)
            except Exception as error:
                code = error.code if isinstance(error, (ModelError, EvidenceError)) else str(error)
                self.store.finish_job(job["id"], code)
                detail = error.detail if isinstance(error, (ModelError, EvidenceError)) else str(error)
                self.store.set_error(job["case_id"], code, detail)
                await self.failed(job, error)
            else:
                self.store.finish_job(job["id"])
                try:
                    await self.completed(job, outcome)
                except Exception as error:
                    self.store.set_error(job["case_id"], "telegram_delivery_failed")
                    try:
                        await self.failed(job, error)
                    except Exception:
                        pass
                else:
                    self.store.set_error(job["case_id"], None)
