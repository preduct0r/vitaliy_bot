"""Model-backed, low-priority learning workflow for the Vitaly simulator.

This is deliberately separate from review/grill code.  It can read a submitted
transcript or feedback item and propose a bounded knowledge patch, but it never
edits a user case, the ledger, or the checked-in profile.  Only an immutable,
validated package can become active for *future* cases.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from .codex import TRANSIENT_CODES
from .learning import (
    TRANSFERABLE_ANALYSIS_RULES,
    fixed_regression_fixtures,
    immutable_package_payload,
    regression_fixture_digest,
    regression_gate,
)
from .models import BotConfig, InputError, ModelError, ModelResult, StateError
from .profile import VitalyProfile, score_holdout_coverage
from .storage import VitalyStore


class LearningRunner(Protocol):
    async def run(
        self, prompt: str, schema: Dict[str, Any], image_paths: Optional[Sequence[Any]] = None
    ) -> ModelResult:
        ...


def _object(properties: Dict[str, Any], required: Iterable[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(required),
    }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


REFERENCE_SCHEMA = _object(
    {
        "source_id": {"type": "string"},
        "source_document_id": {"type": "string"},
        "reference": {"type": "string"},
        "fragment": {"type": "string"},
    },
    ["source_id", "source_document_id", "reference", "fragment"],
)

MEETING_REFERENCE_SCHEMA = _object(
    {
        "meeting_id": {"type": "string"},
        "source": {"type": "string"},
        "time_range": {"type": "string"},
        "text_span": _object(
            {"excerpt": {"type": "string"}},
            ["excerpt"],
        ),
    },
    ["meeting_id", "source"],
)

ANALYSIS_RULE_SCHEMA = _object(
    {
        "id": {"type": "string"},
        "transfer_key": {"type": "string", "enum": sorted(TRANSFERABLE_ANALYSIS_RULES)},
        "scope": {"type": "string", "enum": ["narrow", "general"]},
        "trigger": {"type": "string"},
        "guidance": {"type": "string"},
        "expected_improvement": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
        "evidence_references": {"type": "array", "items": REFERENCE_SCHEMA},
    },
    ["id", "transfer_key", "scope", "trigger", "guidance", "expected_improvement", "source_ids", "evidence_references"],
)

PATTERN_SCHEMA = _object(
    {
        "operation": {"type": "string", "enum": ["add", "replace"]},
        "topic_id": {"type": "string"},
        "name": {"type": "string"},
        "behaviour": {"type": "string"},
        "style_mode": {"type": "string", "enum": ["direct", "cross_examine", "commitment"]},
        "followup_chain": {"type": "array", "items": {"type": "string"}},
        "evidence_meetings": {"type": "array", "items": {"type": "string"}},
        "evidence_references": {"type": "array", "items": MEETING_REFERENCE_SCHEMA},
    },
    ["operation", "topic_id", "name", "behaviour", "style_mode", "followup_chain", "evidence_meetings", "evidence_references"],
)

MEETING_PATCH_SCHEMA = _object(
    {
        "id": {"type": "string"},
        "date": {"type": "string"},
        "title": {"type": "string"},
        "source": {"type": "string"},
        "person": {"type": "string", "enum": ["vitaly_hard_negotiator"]},
        "use_for_simulation": {"type": "boolean"},
        "dedupe_key": {"type": "string"},
        "identity_basis": {"type": "string"},
        "source_quality": {"type": "string"},
        "date_basis": {"type": "string"},
    },
    [
        "id",
        "date",
        "title",
        "source",
        "person",
        "use_for_simulation",
        "dedupe_key",
        "identity_basis",
        "source_quality",
        "date_basis",
    ],
)

PATCH_SCHEMA = _object(
    {
        "schema_version": {"type": "integer", "enum": [1]},
        "summary": {"type": "string"},
        "analysis_rules": {"type": "array", "items": ANALYSIS_RULE_SCHEMA},
        "question_patterns": {"type": "array", "items": PATTERN_SCHEMA},
        "meetings": {"type": "array", "items": MEETING_PATCH_SCHEMA},
    },
    ["schema_version", "summary", "analysis_rules", "question_patterns", "meetings"],
)

MEETING_CLASSIFICATION_SCHEMA = _object(
    {
        "status": {
            "type": "string",
            "enum": ["accepted", "duplicate", "continuation", "other_person", "needs_clarification"],
        },
        "reason": {"type": "string"},
        "identity_basis": {"type": "string", "enum": ["speaker_labels", "user_designation", "role_inferred", "ambiguous"]},
        "identity_confident": {"type": "boolean"},
        "duplicate_of": {"type": "string"},
        "continuation_of": {"type": "string"},
        "source_fragments": {"type": "array", "items": {"type": "string"}},
        "observed_topics": {"type": "array", "items": {"type": "string"}},
    },
    [
        "status",
        "reason",
        "identity_basis",
        "identity_confident",
        "duplicate_of",
        "continuation_of",
        "source_fragments",
        "observed_topics",
    ],
)

FEEDBACK_ASSESSMENT_SCHEMA = _object(
    {
        "result": {"type": "string", "enum": ["case_only", "example", "candidate", "insufficient"]},
        "reason": {"type": "string"},
        "source_document_id": {"type": "string"},
        "fragment": {"type": "string"},
        "expected_improvement": {"type": "string"},
    },
    ["result", "reason", "source_document_id", "fragment", "expected_improvement"],
)

GENERATOR_SCHEMA = _object(
    {
        "decision": {"type": "string", "enum": ["candidate", "none", "needs_clarification"]},
        "reason": {"type": "string"},
        "source_ids": {"type": "array", "items": {"type": "string"}},
        "scope": {"type": "string", "enum": ["analysis", "behaviour", "mixed"]},
        "expected_improvement": {"type": "string"},
        "patch": PATCH_SCHEMA,
    },
    ["decision", "reason", "source_ids", "scope", "expected_improvement", "patch"],
)

REGRESSION_ROW_SCHEMA = _object(
    {
        "scenario_id": {"type": "string"},
        "passed": {"type": "boolean"},
        "target_score": {"type": "number", "minimum": 0, "maximum": 1},
        "critical_error_ids": {"type": "array", "items": {"type": "string"}},
        "style_coverage": {"type": "number", "minimum": 0, "maximum": 1},
        "first_ten_topics": {"type": "array", "items": {"type": "string"}},
    },
    ["scenario_id", "passed", "target_score", "critical_error_ids", "style_coverage", "first_ten_topics"],
)

VERIFIER_SCHEMA = _object(
    {
        "verdict": {"type": "string", "enum": ["pass", "fail", "needs_clarification"]},
        "reason": {"type": "string"},
        "baseline_results": {"type": "array", "items": REGRESSION_ROW_SCHEMA},
        "candidate_results": {"type": "array", "items": REGRESSION_ROW_SCHEMA},
    },
    ["verdict", "reason", "baseline_results", "candidate_results"],
)


@dataclass(frozen=True)
class LearningOutcome:
    kind: str
    payload: Dict[str, Any]


class LearningManager:
    """Owns learning model calls and immutable-package promotion."""

    def __init__(self, store: VitalyStore, profile: VitalyProfile, config: BotConfig, runner: LearningRunner) -> None:
        self.store = store
        self.baseline_profile = profile
        self.config = config
        self.runner = runner
        self.store.ensure_legacy_knowledge_package(self._profile_dict(profile))

    @staticmethod
    def _profile_dict(profile: VitalyProfile) -> Dict[str, Any]:
        return {
            "version": profile.version,
            "simulator_label": profile.simulator_label,
            "meeting_index": profile.meeting_index,
            "question_patterns": profile.question_patterns,
            "holdout": profile.holdout,
        }

    @staticmethod
    def _profile_generation_context(profile: VitalyProfile) -> Dict[str, Any]:
        """Give a background generator the transferable profile projection only."""

        return profile.prompt_context()

    def active_package(self) -> Dict[str, Any]:
        return self.store.active_knowledge_package()

    def profile_for_case(self, case_id: str) -> VitalyProfile:
        package = self.store.knowledge_package_for_case(case_id)["package"]
        return VitalyProfile.from_dict(dict(package["profile"]))

    def analysis_rules_for_case(self, case_id: str) -> List[Dict[str, Any]]:
        """Return generic rule guidance only, never another user's evidence.

        Full source IDs, document references and fragments remain in the
        immutable archive package.  A future case only needs the narrow
        decision rule; serialising provenance into its model prompt would leak
        facts from the case that taught the rule.
        """

        package = self.store.knowledge_package_for_case(case_id)["package"]
        raw = package.get("analysis_rules") if isinstance(package, dict) else []
        if raw is None:
            raw = package.get("patch", {}).get("analysis_rules", [])
        result: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for rule in raw if isinstance(raw, list) else []:
            if not isinstance(rule, dict):
                continue
            transfer_key = str(rule.get("transfer_key") or "")
            template = TRANSFERABLE_ANALYSIS_RULES.get(transfer_key)
            if transfer_key in seen:
                continue
            if template is not None:
                result.append(dict(template))
                seen.add(transfer_key)
        return result

    def profile_version_for_new_case(self) -> Tuple[str, str]:
        package = self.active_package()
        payload = package["package"]
        return str(payload["profile"]["version"]), str(package["id"])

    async def _model(
        self,
        job: Dict[str, Any],
        kind: str,
        prompt: str,
        schema: Dict[str, Any],
        image_paths: Optional[Sequence[Path]] = None,
    ) -> Dict[str, Any]:
        last_error: Optional[ModelError] = None
        for attempt in range(2):
            run_id = self.store.create_learning_model_run(
                str(job["id"]), kind, self.config.model, self.config.reasoning_effort
            )
            try:
                result = await self.runner.run(prompt, schema, image_paths)
            except ModelError as error:
                self.store.finish_learning_model_run(run_id, error.code + (":" + error.detail[-1000:] if error.detail else ""))
                last_error = error
                if error.code in TRANSIENT_CODES and attempt == 0:
                    continue
                raise
            self.store.finish_learning_model_run(run_id)
            return result.value
        raise last_error or ModelError("codex_transient")

    @staticmethod
    def _read_text(path_value: Any, maximum: Optional[int] = None) -> str:
        if not path_value:
            return ""
        path = Path(str(path_value))
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
        return text if maximum is None else text[:maximum]

    def _feedback_images(self, case_id: str) -> List[Path]:
        """Attach a bounded visual sample when assessing a slide-related claim."""

        images: List[Path] = []
        for artifact in self.store.artifacts(case_id):
            images.extend(Path(value) for value in artifact.get("rendered_pages", []) if Path(value).is_file())
        if len(images) <= 12:
            return images
        step = (len(images) - 1) / 11.0
        return [images[round(index * step)] for index in range(12)]

    async def _analyze_feedback(self, job: Dict[str, Any]) -> LearningOutcome:
        feedback = self.store.get_feedback(str(job["submission_id"]))
        # A model may have completed and persisted its assessment just before a
        # process crash.  ``recover_jobs`` requeues the durable job, so replay
        # the stored result instead of calling the model a second time or
        # overwriting a concluded assessment.
        if feedback.get("status") in {"case_only", "example", "candidate", "insufficient"}:
            return LearningOutcome(
                "feedback_analyzed",
                {"feedback": feedback, "assessment": dict(feedback.get("result") or {})},
            )
        if feedback.get("status") != "received":
            return LearningOutcome("learning_replayed", {"feedback": feedback})
        case = self.store.get_case(str(feedback["case_id"]))
        available_source_document_ids = sorted(self._feedback_reference_ids(feedback))
        prompt = (
            "Ты проверяешь обратную связь к симулятору Виталия. Жалоба сама по себе не доказывает ошибку. "
            "Сопоставь её с исходным аудитом, материалами и evidence-пакетом, если он есть. "
            "Выбери: case_only — полезно лишь в этом кейсе; example — подтверждённый проверочный пример; "
            "candidate — достаточно оснований для узкого будущего правила; insufficient — оснований нет. "
            "Не меняй текущий отчёт и не выдумывай факты.\n"
            + _json(
                {
                    "feedback": feedback.get("text", ""),
                    "audit": case.get("review", {}),
                    "evidence": case.get("evidence_pack", {}),
                    "material_map": case.get("material_map", {}),
                    "visual_pages_attached": bool(self._feedback_images(str(case["id"]))),
                    "available_source_document_ids": available_source_document_ids,
                }
            )
        )
        result = await self._model(
            job,
            "feedback_analyze",
            prompt,
            FEEDBACK_ASSESSMENT_SCHEMA,
            self._feedback_images(str(case["id"])),
        )
        if str(result.get("source_document_id") or "") not in available_source_document_ids:
            result = dict(result)
            result["result"] = "insufficient"
            result["reason"] = "Не удалось связать замечание с исходным аудитом или загруженным материалом."
        status = str(result["result"])
        saved = self.store.set_feedback_result(str(feedback["id"]), status, result)
        return LearningOutcome("feedback_analyzed", {"feedback": saved, "assessment": result})

    async def _analyze_meeting(self, job: Dict[str, Any]) -> LearningOutcome:
        from .documents import transcript_title

        meeting = self.store.get_meeting(str(job["submission_id"]))
        # The same restart boundary applies to classified meetings.  In
        # particular, a ``needs_clarification`` result is not a license to
        # silently run attribution again without a new user clarification.
        if meeting.get("status") in {
            "accepted", "duplicate", "continuation", "other_person", "needs_clarification",
        }:
            return LearningOutcome(
                "meeting_analyzed",
                {"meeting": meeting, "classification": dict(meeting.get("classification") or {})},
            )
        if meeting.get("status") != "received":
            return LearningOutcome("learning_replayed", {"meeting": meeting})
        transcript = self._read_text(meeting.get("normalized_path"))
        if not transcript:
            raise StateError("Текст транскрипта недоступен; загрузите встречу заново")
        inferred_title = transcript_title(transcript, str(meeting.get("source_name") or "Встреча"))
        if inferred_title != str(meeting.get("title") or ""):
            meeting = self.store.set_meeting_title(str(meeting["id"]), inferred_title)
        candidates: List[Dict[str, Any]] = []
        for other in self.store.meeting_similarity_candidates(str(meeting["id"])):
            candidates.append(
                {
                    "id": other["id"],
                    "date": other.get("meeting_date"),
                    "title": other.get("title"),
                    "designation": other.get("vitaly_designation"),
                    "text": self._read_text(other.get("normalized_path"), 12_000),
                }
            )
        active = self.active_package()
        active_profile = VitalyProfile.from_dict(dict(active["package"]["profile"]))
        prompt = (
            "Ты классифицируешь новый транскрипт для профиля переговорного поведения. "
            "Для Granola учитывай, что весь разговор может быть помечен Them или одним участником. "
            "Искомый Виталий — тот, кто ведёт переговоры: задаёт жёсткие уточняющие вопросы, перебивает и возвращает "
            "собеседников к цифрам, срокам и основаниям. Восстанавливай роль по последовательности вопросов и ответов "
            "и сходству с подтверждёнными паттернами. Упоминание или обращение по имени само по себе не доказывает роль: "
            "не путай адресата с тем, кто задаёт вопросы. Это допустимая role_inferred атрибуция. "
            "По принятому правилу запись принимается целиком, даже если отдельные реплики нельзя надёжно разделить. "
            "Не выбирай needs_clarification только из-за отсутствия speaker labels или неоднозначности отдельных реплик. "
            "other_person ставь только при явном свидетельстве в записи, что речь о другом Виталии. "
            "Проверь точные/содержательные дубли и продолжение одной встречи. "
            "accepted означает самостоятельную встречу нужного Виталия; не требуется назвать его метку говорящего. "
            "Не извлекай бизнес-факты и не копируй реплики дословно.\n"
            + _json(
                {
                    "submission": {
                        "id": meeting["id"],
                        "date": meeting.get("meeting_date"),
                        "title": meeting.get("title"),
                        "target_vitaly_designation": meeting.get("vitaly_designation"),
                    },
                    "transcript": transcript,
                    "confirmed_vitaly_profile": self._profile_generation_context(active_profile),
                    "possible_duplicates_or_continuations": candidates,
                }
            )
        )
        result = await self._model(job, "meeting_analyze", prompt, MEETING_CLASSIFICATION_SCHEMA)
        result = dict(result)
        if result["status"] == "needs_clarification":
            result["status"] = "accepted"
            result["identity_basis"] = "role_inferred"
            result["identity_confident"] = False
            result["reason"] = str(result.get("reason") or "") + " Принята целиком по пользовательскому правилу; отдельные роли остаются спорными."
        elif result["status"] == "accepted" and result["identity_basis"] == "ambiguous":
            result = dict(result)
            result["identity_basis"] = "role_inferred"
            result["identity_confident"] = False
        # The classifier compares this transcript to archive records that may
        # belong to another person.  Its free-form rationale is retained only
        # as local diagnostic evidence; all Telegram-visible paths use a
        # fixed, source-free explanation.
        result = dict(result)
        result["internal_reason"] = str(result.get("reason") or "")
        public_reason = {
            "accepted": "Запись атрибутирована нужному Виталию и прошла проверку на дубли.",
            "duplicate": "Совпадение с уже известной записью подтверждено проверкой дубликатов.",
            "continuation": "Запись относится к продолжению уже известной встречи.",
            "other_person": "Атрибуция указывает на другого участника с этим именем.",
            "needs_clarification": "Недостаточно данных, чтобы надёжно атрибутировать нужного Виталия.",
        }[str(result["status"])]
        if result["status"] == "accepted" and result.get("identity_basis") == "role_inferred" and not result.get("identity_confident"):
            public_reason = "Запись принята целиком по последовательности обращений, вопросов и уточнений; распределение отдельных реплик остаётся спорным."
        result["reason"] = public_reason
        saved = self.store.set_meeting_result(str(meeting["id"]), result)
        return LearningOutcome("meeting_analyzed", {"meeting": saved, "classification": result})

    def _source_context(self, source: Dict[str, Any], transcript_limit: Optional[int] = None) -> Dict[str, Any]:
        if source["source_kind"] == "feedback":
            return {
                "source_id": source["id"],
                "kind": "feedback",
                "assessment": source.get("result", {}),
                "text": str(source.get("text", ""))[:16_000],
            }
        return {
            "source_id": source["id"],
            "kind": "meeting",
            "date": source.get("meeting_date"),
            "title": source.get("title"),
            "designation": source.get("vitaly_designation"),
            "classification": source.get("classification", {}),
            "source": source.get("normalized_path"),
            "transcript": self._read_text(source.get("normalized_path"), transcript_limit),
        }

    def _bounded_learning_batch(
        self, sources: Sequence[Dict[str, Any]], *, max_sources: int = 8, max_chars: int = 300_000
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Select a deterministic, bounded source batch for one daily cycle.

        Unselected sources retain their eligible state and are considered in a
        later cycle.  This prevents permanent retention from turning one model
        invocation into an unbounded transcript dump.
        """

        selected: List[Dict[str, Any]] = []
        contexts: List[Dict[str, Any]] = []
        remaining = max_chars
        for source in sources:
            if len(selected) >= max_sources or remaining < 1:
                break
            # Transcripts are never sliced: a profile update must be able to
            # cite material from the end of a long meeting as well as its start.
            # Feedback bodies retain their independent, small per-item bound.
            context = self._source_context(source, transcript_limit=16_000 if source.get("source_kind") == "feedback" else None)
            size = len(_json(context))
            if size > remaining and selected:
                continue
            if size > remaining:
                if "transcript" in context:
                    raise ModelError(
                        "learning_transcript_too_large",
                        "Транскрипт превышает безопасный размер одного запуска; он сохранён целиком, но не был обрезан или обработан частично.",
                    )
                context["text"] = str(context.get("text") or "")[:max(0, remaining)]
                size = len(_json(context))
            selected.append(source)
            contexts.append(context)
            remaining -= size
        return selected, contexts

    @staticmethod
    def _meeting_locator(meeting_id: str) -> str:
        return "meeting:" + meeting_id

    def _feedback_reference_ids(self, feedback: Dict[str, Any]) -> set[str]:
        """Return actual, case-local document locators available to feedback."""

        allowed = {"audit"}
        case_id = str(feedback.get("case_id") or "")
        if case_id:
            allowed.update(str(artifact["id"]) for artifact in self.store.artifacts(case_id))
        return allowed

    @staticmethod
    def _feedback_locator(source: Dict[str, Any]) -> str:
        result = source.get("result", {})
        return str(result.get("source_document_id") or "") if isinstance(result, dict) else ""

    def _validate_candidate_source_references(self, patch: Dict[str, Any], sources: Sequence[Dict[str, Any]]) -> None:
        """Bind model references to archive objects that really exist."""

        by_id = {str(source["id"]): source for source in sources}
        for rule in patch.get("analysis_rules", []):
            if not isinstance(rule, dict):
                continue
            for reference in rule.get("evidence_references", []):
                source_id = str(reference.get("source_id") or "") if isinstance(reference, dict) else ""
                source = by_id.get(source_id)
                locator = self._feedback_locator(source or {})
                if source is None or source.get("source_kind") != "feedback" or not locator:
                    raise ModelError("model_output_invalid_schema", "У правила нет доступного источника обратной связи")
                if str(reference.get("source_document_id") or "") != locator:
                    raise ModelError("model_output_invalid_schema", "Ссылка правила не совпадает с исходным документом")
                fragment = str(reference.get("fragment") or "")
                if not fragment or len(fragment) > 2_000:
                    raise ModelError("model_output_invalid_schema", "Фрагмент основания правила имеет неверную длину")
        for pattern in patch.get("question_patterns", []):
            if not isinstance(pattern, dict):
                continue
            for reference in pattern.get("evidence_references", []):
                meeting_id = str(reference.get("meeting_id") or "") if isinstance(reference, dict) else ""
                source = by_id.get(meeting_id)
                if source is None or source.get("source_kind") != "meeting":
                    raise ModelError("model_output_invalid_schema", "Ссылка паттерна не указывает на принятую встречу")
                if str(reference.get("source") or "") != self._meeting_locator(meeting_id):
                    raise ModelError("model_output_invalid_schema", "Ссылка паттерна не совпадает с архивом встречи")
                time_range = str(reference.get("time_range") or "")
                if time_range:
                    if not re.fullmatch(r"\d{1,2}:\d{2}(?:\s*[-–]\s*\d{1,2}:\d{2})?", time_range):
                        raise ModelError("model_output_invalid_schema", "У ссылки встречи неверный диапазон времени")
                    continue
                span = reference.get("text_span")
                excerpt = str(span.get("excerpt") or "").strip() if isinstance(span, dict) else ""
                if len(excerpt) < 12 or len(excerpt) > 500:
                    raise ModelError("model_output_invalid_schema", "У ссылки встречи неверный текстовый фрагмент")
                transcript = self._read_text(source.get("normalized_path"))
                if not transcript or transcript.count(excerpt) != 1:
                    raise ModelError("model_output_invalid_schema", "Текстовая ссылка встречи должна один раз точно встречаться в сохранённом транскрипте")
        for meeting in patch.get("meetings", []):
            if not isinstance(meeting, dict):
                continue
            meeting_id = str(meeting.get("id") or "")
            source = by_id.get(meeting_id)
            if source is None or source.get("source_kind") != "meeting":
                raise ModelError("model_output_invalid_schema", "В профиль добавлена недоступная встреча")
            if str(meeting.get("source") or "") != self._meeting_locator(meeting_id):
                raise ModelError("model_output_invalid_schema", "Запись встречи не связана с архивным источником")

    @staticmethod
    def _package_rules(package: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw = package.get("analysis_rules")
        if raw is None:
            raw = package.get("patch", {}).get("analysis_rules", [])
        return [dict(item) for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []

    @staticmethod
    def _source_ids(generated: Dict[str, Any], sources: Sequence[Dict[str, Any]], *, require: bool) -> List[str]:
        valid_ids = {str(item["id"]) for item in sources}
        selected = [str(value) for value in generated.get("source_ids", [])]
        if len(set(selected)) != len(selected) or any(value not in valid_ids for value in selected):
            raise ModelError("model_output_invalid_schema", "Кандидат ссылается на отсутствующий или повторённый источник")
        if require and not selected:
            raise ModelError("model_output_invalid_schema", "У кандидата нет оснований")
        return selected or [str(item["id"]) for item in sources]

    @staticmethod
    def _validate_candidate_provenance(patch: Dict[str, Any], sources: Sequence[Dict[str, Any]], scope: str) -> None:
        """Keep feedback rules and meeting behaviour in separate evidence lanes."""

        if not isinstance(patch, dict):
            raise ModelError("model_output_invalid_schema", "Патч кандидата имеет неверную структуру")
        feedback_ids = {str(item["id"]) for item in sources if item.get("source_kind") == "feedback"}
        meeting_ids = {str(item["id"]) for item in sources if item.get("source_kind") == "meeting"}
        rules = patch.get("analysis_rules", [])
        patterns = patch.get("question_patterns", [])
        meetings = patch.get("meetings", [])
        if not isinstance(rules, list) or not isinstance(patterns, list) or not isinstance(meetings, list):
            raise ModelError("model_output_invalid_schema", "Патч кандидата имеет неверные списки")
        if scope == "analysis" and (patterns or meetings):
            raise ModelError("model_output_invalid_schema", "Кандидат анализа не может менять профиль поведения")
        if scope == "behaviour" and rules:
            raise ModelError("model_output_invalid_schema", "Поведенческий кандидат не может добавлять правила анализа")
        for rule in rules:
            referenced = {str(value) for value in rule.get("source_ids", [])} if isinstance(rule, dict) else set()
            if not referenced or not referenced.issubset(feedback_ids):
                raise ModelError("model_output_invalid_schema", "Правило анализа должно опираться только на выбранную обратную связь")
        for pattern in patterns:
            referenced = {str(value) for value in pattern.get("evidence_meetings", [])} if isinstance(pattern, dict) else set()
            if not referenced or not referenced.issubset(meeting_ids):
                raise ModelError("model_output_invalid_schema", "Паттерн вопросов должен опираться только на выбранные встречи")
        included_meetings = {
            str(meeting.get("id") or "") for meeting in meetings if isinstance(meeting, dict)
        }
        cited_meetings = {
            str(value)
            for pattern in patterns if isinstance(pattern, dict)
            for value in pattern.get("evidence_meetings", [])
        }
        missing_meetings = sorted(cited_meetings - included_meetings)
        if missing_meetings:
            raise ModelError(
                "model_output_invalid_schema",
                "Паттерн ссылается на встречу, отсутствующую в индексе профиля: " + ", ".join(missing_meetings),
            )
        for meeting in meetings:
            meeting_id = str(meeting.get("id") or "") if isinstance(meeting, dict) else ""
            if meeting_id not in meeting_ids:
                raise ModelError("model_output_invalid_schema", "В профиль можно добавить только выбранную принятую встречу")

    def _candidate_package(
        self,
        active: Dict[str, Any],
        patch: Dict[str, Any],
        sources: Sequence[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        active_payload = dict(active["package"])
        profile = VitalyProfile.from_dict(dict(active_payload["profile"]))
        feedback_ids = [str(item["id"]) for item in sources if item.get("source_kind") == "feedback"]
        meeting_ids = [str(item["id"]) for item in sources if item.get("source_kind") == "meeting"]
        version = "knowledge." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        package = immutable_package_payload(
            profile,
            patch,
            package_version=version,
            accepted_source_ids=feedback_ids,
            accepted_meeting_ids=meeting_ids,
            prior_analysis_rules=self._package_rules(active_payload),
        )
        return version, package

    def _finish_no_change(
        self,
        generated: Dict[str, Any],
        sources: Sequence[Dict[str, Any]],
        active: Dict[str, Any],
    ) -> LearningOutcome:
        source_ids = self._source_ids(
            generated, sources, require=generated["decision"] == "needs_clarification"
        )
        status = "needs_clarification" if generated["decision"] == "needs_clarification" else "no_change"
        candidate = self.store.create_learning_candidate(
            source_ids,
            dict(generated["patch"]),
            str(generated["scope"]),
            str(generated["expected_improvement"]),
            baseline_package_id=str(active["id"]),
        )
        verification = {"stage": "generation", "reason": generated["reason"], "decision": generated["decision"]}
        saved = self.store.set_candidate_result(str(candidate["id"]), status=status, verification=verification)
        self.store.annotate_learning_sources(source_ids, status)
        return LearningOutcome(
            "knowledge_needs_clarification" if status == "needs_clarification" else "knowledge_noop",
            {"candidate": saved, "reason": generated["reason"], "decision": generated["decision"]},
        )

    async def _generate_knowledge(self, job: Dict[str, Any]) -> LearningOutcome:
        eligible_sources = self.store.eligible_learning_sources()
        if not eligible_sources:
            return LearningOutcome("knowledge_noop", {"reason": "Нет новых проверенных источников для пакета знаний."})
        sources, source_context = self._bounded_learning_batch(eligible_sources)
        if not sources:
            return LearningOutcome("knowledge_noop", {"reason": "Новые источники слишком велики для безопасного пакета."})
        active = self.active_package()
        profile = VitalyProfile.from_dict(dict(active["package"]["profile"]))
        prompt = (
            "Ты предлагаешь ОДИН ограниченный кандидат обновления контекста симулятора Виталия. "
            "Это не обучение весов, не изменение кода и не изменение ledger. "
            "Правила анализа берутся только из подтверждённых исправлений; один пример оправдывает лишь narrow. "
            "Поведенческий паттерн обязан иметь минимум две разные принятые встречи. "
            "Правило анализа обязано выбрать transfer_key из схемы. Поведенческий паттерн может только replace "
            "существующую тему и обязан выбрать style_mode из схемы. Для каждой новой встречи и её ссылки "
            "используй source строго в форме meeting:<source_id>. Если у транскрипта нет таймкодов, "
            "укажи text_span.excerpt — короткий точный фрагмент из доступного транскрипта; он будет проверен по архиву. "
            "Не копируй реплики дословно, не делай новые бизнес-факты и не используй сомнительную атрибуцию. "
            "Если обоснованного изменения нет, decision=none и верни пустой patch с обязательной структурой.\n"
            + _json(
                {
                    "current_profile": self._profile_generation_context(profile),
                    "replaceable_topic_ids": profile.topic_ids,
                    "current_analysis_rules": self.analysis_rules_for_case_placeholder(active["package"]),
                    "available_sources": source_context,
                }
            )
        )
        generated = await self._model(job, "knowledge_generate", prompt, GENERATOR_SCHEMA)
        if generated["decision"] != "candidate":
            return self._finish_no_change(generated, sources, active)
        source_ids = self._source_ids(generated, sources, require=True)
        selected_sources = self.store.learning_sources_by_ids(source_ids)
        self._validate_candidate_provenance(dict(generated["patch"]), selected_sources, str(generated["scope"]))
        self._validate_candidate_source_references(dict(generated["patch"]), selected_sources)
        # Validate now, before persisting a candidate; the same frozen inputs
        # are rebuilt by the independent verifier in its own FIFO step.
        self._candidate_package(active, dict(generated["patch"]), selected_sources)
        candidate, verify_job = self.store.create_learning_candidate_with_verification(
            source_ids,
            dict(generated["patch"]),
            str(generated["scope"]),
            str(generated["expected_improvement"]),
            baseline_package_id=str(active["id"]),
        )
        return LearningOutcome("knowledge_candidate", {"candidate": candidate, "verification_job": verify_job})

    async def _verify_candidate(
        self,
        job: Dict[str, Any],
        candidate: Dict[str, Any],
        active: Dict[str, Any],
        package_payload: Dict[str, Any],
        source_context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Separate model run; it sees sources/criteria but no generator rationale."""

        # The verifier needs the candidate's new evidence, but not every raw
        # reference retained by older packages.  Keep the baseline comparison
        # transferable and expose source-specific detail only for this
        # candidate's explicitly selected sources.
        active_profile = VitalyProfile.from_dict(dict(active["package"]["profile"]))
        candidate_profile = VitalyProfile.from_dict(dict(package_payload["profile"]))
        current_context = {
            "package_version": active["package"].get("package_version"),
            "profile": active_profile.prompt_context(),
            "analysis_rules": self.analysis_rules_for_case_placeholder(active["package"]),
        }
        candidate_context = {
            "package_version": package_payload.get("package_version"),
            "profile": candidate_profile.prompt_context(),
            # This incremental patch is the evidence under review; it is not
            # a dump of all prior user sources.
            "patch": package_payload.get("patch", {}),
            "analysis_rules": self.analysis_rules_for_case_placeholder(package_payload),
        }

        prompt = (
            "Ты независимый проверяющий кандидата обновления для симулятора Виталия. "
            "Проверь структурные основания, отсутствие неподтверждённых обобщений и одинаковый набор регрессий "
            "для текущей и кандидатной версии. Не доверяй положительному решению сам по себе. "
            "В каждом результате дай passed, target_score, critical_error_ids; для style_topic_coverage также style_coverage. "
            "Кандидат может пройти только без новых критических ошибок, без регрессий обязательных сценариев, "
            "с улучшением хотя бы одного и покрытием тематических групп не ниже 70% и не ниже текущего.\n"
            + _json(
                {
                    "current_package": current_context,
                    "candidate_package": candidate_context,
                    "sources": source_context,
                    "fixed_regressions": fixed_regression_fixtures(),
                }
            )
        )
        fixtures = fixed_regression_fixtures()
        result = await self._model(job, "knowledge_verify", prompt, VERIFIER_SCHEMA)
        baseline_rows = self._with_calculated_coverage(
            result["baseline_results"], VitalyProfile.from_dict(dict(active["package"]["profile"]))
        )
        candidate_rows = self._with_calculated_coverage(
            result["candidate_results"], VitalyProfile.from_dict(dict(package_payload["profile"]))
        )
        try:
            gate = regression_gate(baseline_rows, candidate_rows, fixtures=fixtures)
        except InputError as error:
            raise ModelError("model_output_invalid_schema", str(error)) from error
        return {
            "verdict": result["verdict"],
            "reason": result["reason"],
            "gate": gate,
            "fixtures": fixtures,
            "fixture_sha256": regression_fixture_digest(fixtures),
            "baseline_results": baseline_rows,
            "candidate_results": candidate_rows,
            "model_results": result,
        }

    @staticmethod
    def analysis_rules_for_case_placeholder(package: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw = package.get("analysis_rules")
        if raw is None:
            raw = package.get("patch", {}).get("analysis_rules", [])
        if not isinstance(raw, list):
            return []
        result: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw:
            key = str(item.get("transfer_key") or "") if isinstance(item, dict) else ""
            if key in seen or key not in TRANSFERABLE_ANALYSIS_RULES:
                continue
            result.append(dict(TRANSFERABLE_ANALYSIS_RULES[key]))
            seen.add(key)
        return result

    @staticmethod
    def _with_calculated_coverage(rows: Sequence[Dict[str, Any]], profile: VitalyProfile) -> List[Dict[str, Any]]:
        result = [dict(row) for row in rows]
        for row in result:
            if row.get("scenario_id") == "style_topic_coverage":
                coverage = score_holdout_coverage(profile, list(row.get("first_ten_topics", [])))
                row["style_coverage"] = coverage["coverage"]
                row["coverage_detail"] = coverage
        return result

    async def _verify_knowledge(self, job: Dict[str, Any]) -> LearningOutcome:
        candidate_id = str(job.get("payload", {}).get("candidate_id") or "")
        if not candidate_id:
            raise StateError("Не указан кандидат для проверки")
        candidate = self.store.get_learning_candidate(candidate_id)
        if candidate.get("status") != "candidate":
            return LearningOutcome("knowledge_noop", {"reason": "Кандидат уже обработан.", "candidate": candidate})
        active = self.active_package()
        source_ids = [str(value) for value in candidate["source_ids"]]
        sources = self.store.learning_sources_by_ids(source_ids)
        if str(candidate.get("baseline_package_id") or "") != str(active["id"]):
            verification = {
                "stage": "verification",
                "reason": "Базовый пакет изменился до проверки; автоматическое включение остановлено.",
            }
            saved = self.store.set_candidate_result(candidate_id, status="needs_clarification", verification=verification)
            self.store.annotate_learning_sources(source_ids, "needs_clarification")
            return LearningOutcome("knowledge_needs_clarification", {"candidate": saved, "verification": verification})
        patch = dict(candidate["patch"])
        self._validate_candidate_provenance(patch, sources, str(candidate["scope"]))
        self._validate_candidate_source_references(patch, sources)
        version, package_payload = self._candidate_package(active, patch, sources)
        verification = await self._verify_candidate(
            job, candidate, active, package_payload, [self._source_context(item) for item in sources]
        )
        gate = verification["gate"]
        self.store.add_regression_run(candidate_id, str(active["id"]), verification, bool(gate["passing"]))
        if verification["verdict"] == "needs_clarification":
            saved = self.store.set_candidate_result(candidate_id, status="needs_clarification", verification=verification)
            self.store.annotate_learning_sources(source_ids, "needs_clarification")
            return LearningOutcome("knowledge_needs_clarification", {"candidate": saved, "verification": verification})
        if verification["verdict"] != "pass" or not gate["passing"]:
            saved = self.store.set_candidate_result(
                candidate_id, status="rejected", verification=verification, error="quality_gate_failed"
            )
            self.store.annotate_learning_sources(source_ids, "rejected")
            return LearningOutcome("knowledge_rejected", {"candidate": saved, "verification": verification})
        try:
            package = self.store.publish_knowledge_package(
                version=version,
                package=package_payload,
                parent_id=str(active["id"]),
                summary=str(patch.get("summary") or candidate["expected_improvement"]),
                verification=verification,
                candidate_id=candidate_id,
                source_ids=source_ids,
            )
        except StateError as error:
            clarification = dict(verification)
            clarification["reason"] = str(error)
            saved = self.store.set_candidate_result(candidate_id, status="needs_clarification", verification=clarification)
            self.store.annotate_learning_sources(source_ids, "needs_clarification")
            return LearningOutcome("knowledge_needs_clarification", {"candidate": saved, "verification": clarification})
        saved = self.store.get_learning_candidate(candidate_id)
        return LearningOutcome("knowledge_enabled", {"candidate": saved, "package": package, "verification": verification})

    async def execute(self, job: Dict[str, Any]) -> LearningOutcome:
        kind = str(job["kind"])
        if kind == "feedback_analyze":
            return await self._analyze_feedback(job)
        if kind == "meeting_analyze":
            return await self._analyze_meeting(job)
        if kind == "knowledge_generate":
            return await self._generate_knowledge(job)
        if kind == "knowledge_verify":
            return await self._verify_knowledge(job)
        raise StateError("Неизвестный тип фоновой задачи обучения")
