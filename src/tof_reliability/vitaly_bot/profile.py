"""Versioned, paraphrased behavioural profile used by the simulator.

User-approved transcripts are archived in discussions/vitaly. The model receives
paraphrased behavioural observations and source references, not raw transcripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .models import InputError


# The archive can retain a model's detailed paraphrase and the meeting evidence,
# but another user's case must receive only a small, code-owned behavioural
# projection.  This blocks an accidental customer/document fact embedded in a
# future learned pattern from being copied into a different case prompt.
_SAFE_PATTERN_CONTEXT: Dict[str, Dict[str, Any]] = {
    "evidence_scope": {
        "name": "Периметр и источник цифры",
        "behaviour": "Требует назвать источник, период, объект и метод подтверждения значимой цифры.",
        "followup_chain": ["Где источник?", "Какой периметр и период?"],
    },
    "effect_measurement": {
        "name": "Измеримость эффекта",
        "behaviour": "Отделяет подтверждённый результат от потенциала и запрашивает метод измерения эффекта.",
        "followup_chain": ["Как измерен эффект?", "Что подтверждает результат?"],
    },
    "mechanism": {
        "name": "Механизм изменения",
        "behaviour": "Проверяет причинно-следственную связь между действием, процессом и ожидаемым результатом.",
        "followup_chain": ["За счёт чего это сработает?", "Как это будет проверено?"],
    },
    "ownership_timing": {
        "name": "Владелец и срок",
        "behaviour": "Не принимает общий план без владельца, срока и критерия завершения.",
        "followup_chain": ["Кто отвечает?", "К какому сроку?"],
    },
    "counterexample": {
        "name": "Контрпример и границы применимости",
        "behaviour": "Просит назвать условия, при которых вывод не работает, и границы применимости.",
        "followup_chain": ["Где это не сработает?", "Какие ограничения?"],
    },
    "priority_business_case": {
        "name": "Приоритет и бизнес-кейс",
        "behaviour": "Проверяет приоритет, альтернативы и связь предложения с производственной задачей.",
        "followup_chain": ["Почему это приоритет?", "Что будет, если не делать?"],
    },
    "implementation_readiness": {
        "name": "Готовность к внедрению",
        "behaviour": "Проверяет готовность решения к реальному внедрению, а не только к презентации.",
        "followup_chain": ["Что нужно для запуска?", "Как закрепите результат?"],
    },
}

_SAFE_STYLE_SUFFIXES = {
    "direct": "",
    "cross_examine": " После первого ответа быстро сужает вопрос до проверяемого факта.",
    "commitment": " Завершает цепочку требованием назвать обязательство, владельца и срок.",
}


@dataclass(frozen=True)
class VitalyProfile:
    """A validated immutable profile for one simulator version."""

    version: str
    simulator_label: str
    meeting_index: List[Dict[str, Any]]
    question_patterns: List[Dict[str, Any]]
    holdout: Dict[str, Any]

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "VitalyProfile":
        required = {"version", "simulator_label", "meeting_index", "question_patterns", "holdout"}
        missing = sorted(required - set(value))
        if missing:
            raise InputError("Профиль Виталия неполный: " + ", ".join(missing))
        meetings = value["meeting_index"]
        patterns = value["question_patterns"]
        if not isinstance(meetings, list) or not isinstance(patterns, list):
            raise InputError("Профиль Виталия имеет неверную структуру")
        core = [meeting for meeting in meetings if meeting.get("use_for_simulation")]
        if len(core) < 10:
            raise InputError("В ядре профиля должны быть не менее 10 подтверждённых встреч")
        if len({meeting["id"] for meeting in meetings}) != len(meetings):
            raise InputError("В индексе профиля повторяются идентификаторы встреч")
        if len({meeting.get("dedupe_key", meeting["id"]) for meeting in core}) != len(core):
            raise InputError("В ядре профиля есть дубли встреч")
        if any(meeting.get("person") != "vitaly_hard_negotiator" for meeting in core):
            raise InputError("В ядро профиля попала запись другого Виталия")
        if not patterns:
            raise InputError("В профиле нет паттернов вопросов")
        return cls(
            version=str(value["version"]),
            simulator_label=str(value["simulator_label"]),
            meeting_index=meetings,
            question_patterns=patterns,
            holdout=dict(value["holdout"]),
        )

    @classmethod
    def load(cls, path: Path) -> "VitalyProfile":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @property
    def core_meetings(self) -> List[Dict[str, Any]]:
        return [meeting for meeting in self.meeting_index if meeting.get("use_for_simulation")]

    @property
    def topic_ids(self) -> List[str]:
        return [str(pattern["topic_id"]) for pattern in self.question_patterns]

    def evidence_for(self, topic_id: str) -> List[str]:
        pattern = next((item for item in self.question_patterns if item["topic_id"] == topic_id), None)
        if pattern is None:
            return []
        return list(pattern.get("evidence_meetings", []))

    def prompt_context(self, include_holdout: bool = True) -> Dict[str, Any]:
        meetings: Iterable[Dict[str, Any]] = self.core_meetings
        if not include_holdout:
            holdout_ids = set(self.holdout.get("meeting_ids", []))
            meetings = [meeting for meeting in meetings if meeting["id"] not in holdout_ids]
        # Evidence references remain in the immutable knowledge package for
        # auditability, but a new user's prompt needs only the transferable
        # negotiation pattern.  Source paths, transcript fragments and
        # per-meeting provenance must never be copied into another case.
        patterns: List[Dict[str, Any]] = []
        for pattern in self.question_patterns:
            topic_id = str(pattern.get("topic_id") or "")
            safe = _SAFE_PATTERN_CONTEXT.get(topic_id)
            if safe is None:
                # Unknown archived patterns remain auditable in the package but
                # deliberately never become cross-case model context.
                continue
            mode = str(pattern.get("style_mode") or "direct")
            patterns.append(
                {
                    "topic_id": topic_id,
                    "name": safe["name"],
                    "behaviour": safe["behaviour"] + _SAFE_STYLE_SUFFIXES.get(mode, ""),
                    "followup_chain": list(safe["followup_chain"]),
                }
            )
        return {
            "profile_version": self.version,
            "simulator_label": self.simulator_label,
            "meeting_ids": [meeting["id"] for meeting in meetings],
            "question_patterns": patterns,
            "rules": [
                "Симулируй стиль переговоров, а не личность и не дословные реплики.",
                "Проверяй цифры, периметр, метод, владельца, срок и способ внедрения.",
                "Не выдавай гипотезу или сценарий за подтверждённый эффект.",
                "Задавай короткие прямые вопросы и при слабом ответе сужай его уточнениями.",
                "Применяй цепочки вопросов по релевантности материалу; наблюдения встреч не являются фактами ledger или обязательными порогами эффективности.",
            ],
        }


def default_profile_path() -> Path:
    return Path(__file__).with_name("data") / "vitaly_profile.v1.1.json"


def load_default_profile() -> VitalyProfile:
    return VitalyProfile.load(default_profile_path())


def score_holdout_coverage(profile: VitalyProfile, predicted_topics: Sequence[str]) -> Dict[str, Any]:
    """Score a model's holdout prediction against the pre-labelled topic set."""

    expected = set(profile.holdout.get("expected_topic_ids", []))
    predicted = set(predicted_topics[:10])
    matched = sorted(expected & predicted)
    coverage = len(matched) / len(expected) if expected else 0.0
    return {
        "profile_version": profile.version,
        "expected_topics": sorted(expected),
        "predicted_topics": list(predicted_topics[:10]),
        "matched_topics": matched,
        "coverage": round(coverage, 3),
        "passing": coverage >= 0.7,
    }
