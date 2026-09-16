"""Pure, versioned knowledge-package primitives for the Vitaly simulator.

This module deliberately does not know about Telegram, SQLite, Codex, or the
filesystem layout.  The runtime persists the JSON payloads it receives from
here; keeping the rules pure makes a candidate package reproducible before it
is allowed to affect a new case.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .models import InputError
from .profile import VitalyProfile


_TOPIC_ID = re.compile(r"^[a-z][a-z0-9_]{1,79}$")
_RULE_ID = re.compile(r"^[a-z][a-z0-9_:-]{1,99}$")
_PATCH_KEYS = {
    "schema_version",
    "summary",
    "analysis_rules",
    "question_patterns",
    "meetings",
}
_ANALYSIS_RULE_KEYS = {
    "id",
    "transfer_key",
    "scope",
    "trigger",
    "guidance",
    "expected_improvement",
    "source_ids",
    "evidence_references",
}
_QUESTION_PATTERN_KEYS = {
    "operation",
    "topic_id",
    "name",
    "behaviour",
    "style_mode",
    "followup_chain",
    "evidence_meetings",
    "evidence_references",
}

# Only these code-owned abstractions may cross from one person's archived
# feedback into another person's future case.  The source-specific text of a
# model-produced rule stays in the immutable archive; it is not a transferable
# instruction by itself.
TRANSFERABLE_ANALYSIS_RULES: Dict[str, Dict[str, str]] = {
    "compare_definition_before_conflict": {
        "id": "compare_definition_before_conflict",
        "scope": "narrow",
        "trigger": "Похожие показатели выглядят несогласованными.",
        "guidance": "Сначала сопоставь определение, период, периметр и базу сравнения; называй противоречие только при конфликте в одинаковом контуре.",
    },
    "check_period_scope_before_conflict": {
        "id": "check_period_scope_before_conflict",
        "scope": "narrow",
        "trigger": "Сопоставляются значения из разных периодов или периметров.",
        "guidance": "Явно раздели план, факт, период и объект сравнения до вывода о расхождении.",
    },
    "check_units_before_conflict": {
        "id": "check_units_before_conflict",
        "scope": "narrow",
        "trigger": "Сопоставляются показатели в разных единицах.",
        "guidance": "Проверь перевод единиц и формулу сопоставления; не называй сравнение доказанным без них.",
    },
    "read_visual_caption_before_conflict": {
        "id": "read_visual_caption_before_conflict",
        "scope": "narrow",
        "trigger": "Вывод зависит от диаграммы, подписи или изображения.",
        "guidance": "Сверь подпись, легенду, фильтры и единицы на визуальном материале до замечания.",
    },
    "separate_fact_from_scenario": {
        "id": "separate_fact_from_scenario",
        "scope": "narrow",
        "trigger": "Эффект или результат заявлен без ясного статуса подтверждения.",
        "guidance": "Отдели подтверждённый факт от потенциала, сценария и гипотезы; запроси источник и метод проверки.",
    },
    "require_owner_deadline_mechanism": {
        "id": "require_owner_deadline_mechanism",
        "scope": "narrow",
        "trigger": "Предложение не связывает действие с исполнением.",
        "guidance": "Проверь наличие владельца, срока, механизма внедрения и критерия завершения.",
    },
    "require_source_for_number": {
        "id": "require_source_for_number",
        "scope": "narrow",
        "trigger": "В материале есть значимая цифра без подтверждения.",
        "guidance": "Запроси источник, метод расчёта, период и периметр цифры до вывода.",
    },
}

TRANSFERABLE_STYLE_MODES = {"direct", "cross_examine", "commitment"}


def public_release_summary(patch: Mapping[str, Any]) -> str:
    """Return a source-free description safe to show every Telegram user."""

    rules = len(patch.get("analysis_rules", [])) if isinstance(patch, Mapping) else 0
    patterns = len(patch.get("question_patterns", [])) if isinstance(patch, Mapping) else 0
    meetings = len(patch.get("meetings", [])) if isinstance(patch, Mapping) else 0
    parts: List[str] = []
    if rules:
        parts.append("правила проверки: {}".format(rules))
    if patterns:
        parts.append("паттерны вопросов: {}".format(patterns))
    if meetings:
        parts.append("встречи профиля: {}".format(meetings))
    return "Автоматическое обновление: " + ("; ".join(parts) if parts else "без переносимых изменений") + "."
_MEETING_KEYS = {
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
}


def normalize_text(value: str) -> str:
    """Return the conservative form used for deterministic exact deduplication.

    Case, Unicode compatibility variants and whitespace often differ between a
    Telegram text message and the same transcript saved as a file.  Punctuation
    is intentionally retained: removing it would turn a safe exact-dedup key
    into a semantic claim.
    """

    if not isinstance(value, str):
        raise InputError("Для нормализации нужен текст")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    return " ".join(normalized.casefold().split())


def normalized_text_hash(value: str) -> str:
    """Return a stable SHA-256 of :func:`normalize_text`."""

    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    """Serialize JSON without timestamps, ordering variance, or NaN values."""

    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InputError("Пакет знаний должен состоять из JSON-значений") from exc


def canonical_copy(value: Any) -> Any:
    """Return an independent JSON-only copy with deterministic key ordering."""

    return json.loads(canonical_json(value))


def _as_list(value: Any, field: str) -> List[Any]:
    if not isinstance(value, list):
        raise InputError("Поле {} должно быть списком".format(field))
    return value


def _nonempty_text(value: Any, field: str, limit: int = 4000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError("Поле {} не заполнено".format(field))
    text = value.strip()
    if len(text) > limit:
        raise InputError("Поле {} слишком длинное".format(field))
    return text


def _unique_strings(value: Any, field: str) -> List[str]:
    values = _as_list(value, field)
    result = [_nonempty_text(item, field, limit=200) for item in values]
    if not result:
        raise InputError("Поле {} не должно быть пустым".format(field))
    if len(set(result)) != len(result):
        raise InputError("В поле {} есть повторы".format(field))
    return result


def _reference_for_sources(
    references: Any,
    source_ids: Sequence[str],
    *,
    subject: str,
) -> List[Dict[str, Any]]:
    """Validate traceable document-feedback references for an analysis rule."""

    items = _as_list(references, "{}.evidence_references".format(subject))
    if not items:
        raise InputError("Для {} нужны ссылки на основания".format(subject))
    result: List[Dict[str, Any]] = []
    referenced: set[str] = set()
    for number, raw in enumerate(items, start=1):
        if not isinstance(raw, Mapping):
            raise InputError("Ссылка {} должна быть объектом".format(number))
        source_id = _nonempty_text(raw.get("source_id"), "source_id", limit=200)
        if source_id not in source_ids:
            raise InputError("Ссылка {} указывает на источник вне правила".format(number))
        source_document_id = raw.get("source_document_id")
        reference = raw.get("reference")
        fragment = raw.get("fragment") or raw.get("time_range")
        has_document = isinstance(source_document_id, str) and bool(source_document_id.strip())
        has_reference = isinstance(reference, str) and bool(reference.strip())
        if not has_document and not has_reference:
            raise InputError("Ссылка {} не содержит документа или ссылки".format(number))
        if not isinstance(fragment, str) or not fragment.strip():
            raise InputError("Ссылка {} не содержит фрагмента основания".format(number))
        clean = canonical_copy(dict(raw))
        result.append(clean)
        referenced.add(source_id)
    missing = sorted(set(source_ids) - referenced)
    if missing:
        raise InputError("Нет ссылки на основание для: {}".format(", ".join(missing)))
    return result


def _meeting_references(
    references: Any,
    meeting_ids: Sequence[str],
    *,
    subject: str,
) -> List[Dict[str, Any]]:
    """Validate a transcript fragment for every meeting cited by a pattern."""

    items = _as_list(references, "{}.evidence_references".format(subject))
    if not items:
        raise InputError("Для {} нужны ссылки на фрагменты встреч".format(subject))
    result: List[Dict[str, Any]] = []
    referenced: set[str] = set()
    for number, raw in enumerate(items, start=1):
        if not isinstance(raw, Mapping):
            raise InputError("Ссылка встречи {} должна быть объектом".format(number))
        meeting_id = _nonempty_text(raw.get("meeting_id"), "meeting_id", limit=200)
        if meeting_id not in meeting_ids:
            raise InputError("Ссылка встречи {} указывает на встречу вне паттерна".format(number))
        _nonempty_text(raw.get("source"), "source", limit=1000)
        if not isinstance(raw.get("time_range"), str) or not raw["time_range"].strip():
            raise InputError("Ссылка встречи {} не содержит диапазона реплики".format(number))
        clean = canonical_copy(dict(raw))
        result.append(clean)
        referenced.add(meeting_id)
    missing = sorted(set(meeting_ids) - referenced)
    if missing:
        raise InputError("Нет ссылки на фрагмент для встреч: {}".format(", ".join(missing)))
    return result


def _validate_meeting(raw: Any, accepted_meeting_ids: set[str]) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise InputError("Запись новой встречи должна быть объектом")
    unknown = sorted(set(raw) - _MEETING_KEYS)
    if unknown:
        raise InputError("В записи встречи есть неподдерживаемые поля: {}".format(", ".join(unknown)))
    value = canonical_copy(dict(raw))
    meeting_id = _nonempty_text(value.get("id"), "meetings.id", limit=200)
    if meeting_id not in accepted_meeting_ids:
        raise InputError("В пакет попала непроверенная встреча {}".format(meeting_id))
    _nonempty_text(value.get("date"), "meetings.date", limit=32)
    _nonempty_text(value.get("title"), "meetings.title", limit=1000)
    _nonempty_text(value.get("source"), "meetings.source", limit=2000)
    if value.get("person") != "vitaly_hard_negotiator":
        raise InputError("В профиль можно включить только подтверждённого Виталия")
    if value.get("use_for_simulation") is not True:
        raise InputError("Новая встреча для профиля должна быть одобрена для симуляции")
    return value


def validate_knowledge_patch(
    patch: Mapping[str, Any],
    *,
    accepted_source_ids: Iterable[str],
    accepted_meeting_ids: Iterable[str],
    known_topic_ids: Iterable[str] = (),
    existing_meeting_ids: Iterable[str] = (),
) -> Dict[str, Any]:
    """Validate and canonically copy a constrained candidate knowledge patch.

    ``analysis_rules`` carry corrections from document feedback.  A narrow rule
    needs at least one known, accepted source; a general rule needs two.  Every
    behavioural question pattern needs two distinct *accepted* meetings, so one
    transcript cannot silently become a universal representation of Vitaly.
    """

    if not isinstance(patch, Mapping):
        raise InputError("Кандидат пакета знаний должен быть объектом")
    unknown = sorted(set(patch) - _PATCH_KEYS)
    if unknown:
        raise InputError("В пакете есть неподдерживаемые поля: {}".format(", ".join(unknown)))
    if patch.get("schema_version") != 1:
        raise InputError("Поддерживается только schema_version=1")

    known_sources = {str(item) for item in accepted_source_ids}
    accepted_meetings = {str(item) for item in accepted_meeting_ids}
    existing_topics = {str(item) for item in known_topic_ids}
    existing_meetings = {str(item) for item in existing_meeting_ids}

    summary = patch.get("summary", "")
    if summary:
        summary = _nonempty_text(summary, "summary", limit=1000)

    meetings_raw = _as_list(patch.get("meetings", []), "meetings")
    meetings = [_validate_meeting(item, accepted_meetings) for item in meetings_raw]
    meeting_ids = [str(item["id"]) for item in meetings]
    if len(set(meeting_ids)) != len(meeting_ids):
        raise InputError("В пакете повторяются новые встречи")

    rules_raw = _as_list(patch.get("analysis_rules", []), "analysis_rules")
    rules: List[Dict[str, Any]] = []
    rule_ids: set[str] = set()
    for number, raw in enumerate(rules_raw, start=1):
        subject = "analysis_rules[{}]".format(number - 1)
        if not isinstance(raw, Mapping):
            raise InputError("{} должно быть объектом".format(subject))
        unknown = sorted(set(raw) - _ANALYSIS_RULE_KEYS)
        if unknown:
            raise InputError("В {} есть неподдерживаемые поля: {}".format(subject, ", ".join(unknown)))
        item = canonical_copy(dict(raw))
        rule_id = _nonempty_text(item.get("id"), "{}.id".format(subject), limit=100)
        if not _RULE_ID.fullmatch(rule_id):
            raise InputError("{}.id имеет недопустимый формат".format(subject))
        if rule_id in rule_ids:
            raise InputError("В пакете повторяется правило {}".format(rule_id))
        rule_ids.add(rule_id)
        transfer_key = _nonempty_text(item.get("transfer_key"), "{}.transfer_key".format(subject), limit=100)
        if transfer_key not in TRANSFERABLE_ANALYSIS_RULES:
            raise InputError("{}.transfer_key не входит в утверждённый общий каталог".format(subject))
        scope = item.get("scope")
        if scope not in {"narrow", "general"}:
            raise InputError("{}.scope должен быть narrow или general".format(subject))
        _nonempty_text(item.get("trigger"), "{}.trigger".format(subject))
        _nonempty_text(item.get("guidance"), "{}.guidance".format(subject))
        _nonempty_text(item.get("expected_improvement"), "{}.expected_improvement".format(subject))
        source_ids = _unique_strings(item.get("source_ids"), "{}.source_ids".format(subject))
        unknown_sources = sorted(set(source_ids) - known_sources)
        if unknown_sources:
            raise InputError("Правило {} ссылается на непроверенные источники: {}".format(
                rule_id, ", ".join(unknown_sources)
            ))
        if scope == "narrow" and len(source_ids) < 1:
            raise InputError("Узкому правилу нужно хотя бы одно подтверждённое основание")
        if scope == "general" and len(source_ids) < 2:
            raise InputError("Общему правилу нужны два независимых подтверждённых основания")
        item["evidence_references"] = _reference_for_sources(
            item.get("evidence_references"), source_ids, subject=subject
        )
        rules.append(item)

    patterns_raw = _as_list(patch.get("question_patterns", []), "question_patterns")
    patterns: List[Dict[str, Any]] = []
    pattern_ids: set[str] = set()
    permitted_pattern_meetings = accepted_meetings
    for number, raw in enumerate(patterns_raw, start=1):
        subject = "question_patterns[{}]".format(number - 1)
        if not isinstance(raw, Mapping):
            raise InputError("{} должно быть объектом".format(subject))
        unknown = sorted(set(raw) - _QUESTION_PATTERN_KEYS)
        if unknown:
            raise InputError("В {} есть неподдерживаемые поля: {}".format(subject, ", ".join(unknown)))
        item = canonical_copy(dict(raw))
        operation = item.get("operation", "add")
        if operation not in {"add", "replace"}:
            raise InputError("{}.operation должен быть add или replace".format(subject))
        topic_id = _nonempty_text(item.get("topic_id"), "{}.topic_id".format(subject), limit=80)
        if not _TOPIC_ID.fullmatch(topic_id):
            raise InputError("{}.topic_id имеет недопустимый формат".format(subject))
        if topic_id in pattern_ids:
            raise InputError("В пакете повторяется паттерн {}".format(topic_id))
        if topic_id not in existing_topics or operation != "replace":
            raise InputError("Новый поведенческий паттерн может только заменить существующую безопасную тему")
        pattern_ids.add(topic_id)
        _nonempty_text(item.get("name"), "{}.name".format(subject), limit=500)
        _nonempty_text(item.get("behaviour"), "{}.behaviour".format(subject), limit=4000)
        style_mode = _nonempty_text(item.get("style_mode"), "{}.style_mode".format(subject), limit=80)
        if style_mode not in TRANSFERABLE_STYLE_MODES:
            raise InputError("{}.style_mode не входит в утверждённый каталог".format(subject))
        followups = _unique_strings(item.get("followup_chain"), "{}.followup_chain".format(subject))
        if len(followups) > 5:
            raise InputError("В {} слишком длинная цепочка уточнений".format(subject))
        cited_meetings = _unique_strings(item.get("evidence_meetings"), "{}.evidence_meetings".format(subject))
        if len(cited_meetings) < 2:
            raise InputError("Поведенческий паттерн {} должен опираться минимум на две встречи".format(topic_id))
        unaccepted = sorted(set(cited_meetings) - permitted_pattern_meetings)
        if unaccepted:
            raise InputError("Паттерн {} ссылается на непроверенные встречи: {}".format(
                topic_id, ", ".join(unaccepted)
            ))
        item["operation"] = operation
        item["evidence_references"] = _meeting_references(
            item.get("evidence_references"), cited_meetings, subject=subject
        )
        patterns.append(item)

    # A cited new meeting must be carried into the profile index.  Otherwise a
    # profile can retain a question pattern whose stated evidence is absent
    # from its own immutable meeting ledger.
    cited_pattern_meetings = {
        str(meeting_id)
        for pattern in patterns
        for meeting_id in pattern.get("evidence_meetings", [])
    }
    missing_meeting_records = sorted(cited_pattern_meetings - set(meeting_ids) - existing_meetings)
    if missing_meeting_records:
        raise InputError(
            "Встречи, на которые ссылаются новые паттерны, отсутствуют в meetings: {}".format(
                ", ".join(missing_meeting_records)
            )
        )

    if not rules and not patterns:
        raise InputError("Пакет не содержит ни правила проверки, ни паттерна вопросов")
    return canonical_copy(
        {
            "schema_version": 1,
            "summary": summary,
            "analysis_rules": rules,
            "question_patterns": patterns,
            "meetings": meetings,
        }
    )


def profile_to_dict(profile: Union[VitalyProfile, Mapping[str, Any]]) -> Dict[str, Any]:
    """Return the profile shape accepted by :meth:`VitalyProfile.from_dict`."""

    if isinstance(profile, VitalyProfile):
        value: Dict[str, Any] = {
            "version": profile.version,
            "simulator_label": profile.simulator_label,
            "meeting_index": profile.meeting_index,
            "question_patterns": profile.question_patterns,
            "holdout": profile.holdout,
        }
    elif isinstance(profile, Mapping):
        value = dict(profile)
    else:
        raise InputError("Базовый профиль имеет неверный тип")
    # Validate the baseline before composing rather than trusting a caller's
    # mutable dict.
    validated = VitalyProfile.from_dict(canonical_copy(value))
    return canonical_copy(
        {
            "version": validated.version,
            "simulator_label": validated.simulator_label,
            "meeting_index": validated.meeting_index,
            "question_patterns": validated.question_patterns,
            "holdout": validated.holdout,
        }
    )


def compose_profile_dict(
    baseline: Union[VitalyProfile, Mapping[str, Any]],
    patch: Mapping[str, Any],
    *,
    release_version: Optional[str] = None,
    accepted_source_ids: Iterable[str],
    accepted_meeting_ids: Iterable[str],
) -> Dict[str, Any]:
    """Compose a validated profile dictionary without mutating its inputs."""

    source_ids = list(accepted_source_ids)
    meeting_ids = list(accepted_meeting_ids)
    base = profile_to_dict(baseline)
    base_meeting_ids = {str(item["id"]) for item in base["meeting_index"]}
    accepted_meetings = set(base_meeting_ids) | {str(item) for item in meeting_ids}
    checked = validate_knowledge_patch(
        patch,
        accepted_source_ids=source_ids,
        accepted_meeting_ids=accepted_meetings,
        known_topic_ids=[str(item["topic_id"]) for item in base["question_patterns"]],
        existing_meeting_ids=base_meeting_ids,
    )

    for meeting in checked["meetings"]:
        if meeting["id"] in base_meeting_ids:
            raise InputError("Новая встреча {} уже есть в профиле".format(meeting["id"]))
        base["meeting_index"].append(canonical_copy(meeting))

    positions = {str(item["topic_id"]): index for index, item in enumerate(base["question_patterns"])}
    for raw_pattern in checked["question_patterns"]:
        pattern = canonical_copy(raw_pattern)
        operation = pattern.pop("operation", "add")
        topic_id = str(pattern["topic_id"])
        if operation == "add":
            base["question_patterns"].append(pattern)
            positions[topic_id] = len(base["question_patterns"]) - 1
        else:
            base["question_patterns"][positions[topic_id]] = pattern

    if release_version is not None:
        base["version"] = _nonempty_text(release_version, "release_version", limit=200)
    return profile_to_dict(base)


def compose_profile(
    baseline: Union[VitalyProfile, Mapping[str, Any]],
    patch: Mapping[str, Any],
    *,
    release_version: Optional[str] = None,
    accepted_source_ids: Iterable[str],
    accepted_meeting_ids: Iterable[str],
) -> VitalyProfile:
    """Compose and revalidate a :class:`VitalyProfile` for a new case."""

    return VitalyProfile.from_dict(
        compose_profile_dict(
            baseline,
            patch,
            release_version=release_version,
            accepted_source_ids=accepted_source_ids,
            accepted_meeting_ids=accepted_meeting_ids,
        )
    )


def _package_digest_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = canonical_copy(dict(payload))
    result.pop("package_sha256", None)
    return result


def package_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a package excluding its derived digest field."""

    return hashlib.sha256(canonical_json(_package_digest_payload(payload)).encode("utf-8")).hexdigest()


def immutable_package_payload(
    baseline: Union[VitalyProfile, Mapping[str, Any]],
    patch: Mapping[str, Any],
    *,
    package_version: str,
    accepted_source_ids: Iterable[str],
    accepted_meeting_ids: Iterable[str],
    prior_analysis_rules: Iterable[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Create a reproducible, JSON-only package that can be pinned to a case."""

    source_ids = list(accepted_source_ids)
    meeting_ids = list(accepted_meeting_ids)
    version = _nonempty_text(package_version, "package_version", limit=200)
    base = profile_to_dict(baseline)
    profile = compose_profile_dict(
        base,
        patch,
        release_version=version,
        accepted_source_ids=source_ids,
        accepted_meeting_ids=meeting_ids,
    )
    checked = validate_knowledge_patch(
        patch,
        accepted_source_ids=source_ids,
        accepted_meeting_ids=set(str(item["id"]) for item in base["meeting_index"]) | set(meeting_ids),
        known_topic_ids=[str(item["topic_id"]) for item in base["question_patterns"]],
        existing_meeting_ids=set(str(item["id"]) for item in base["meeting_index"]),
    )
    # ``patch`` is deliberately the provenance of *this* release, whereas
    # ``analysis_rules`` is the cumulative rule set used by future cases.
    # Keeping both avoids re-applying old behavioural patches to an already
    # composed profile while also preventing older analysis rules from being
    # silently dropped on the next release.
    prior_rules = canonical_copy(list(prior_analysis_rules))
    if not isinstance(prior_rules, list):
        raise InputError("Накопленные правила анализа должны быть списком")
    new_rule_ids = {str(item["id"]) for item in checked["analysis_rules"]}
    prior_rule_ids: set[str] = set()
    for item in prior_rules:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str) or not item["id"].strip():
            raise InputError("Накопленное правило анализа имеет неверную структуру")
        rule_id = str(item["id"])
        if rule_id in prior_rule_ids:
            raise InputError("В накопленных правилах повторяется {}".format(rule_id))
        prior_rule_ids.add(rule_id)
    overlap = sorted(prior_rule_ids & new_rule_ids)
    if overlap:
        raise InputError("Новое правило повторяет уже выпущенное: {}".format(", ".join(overlap)))

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "package_version": version,
        "baseline_profile_version": base["version"],
        "patch": checked,
        "analysis_rules": prior_rules + checked["analysis_rules"],
        "profile": profile,
    }
    payload["package_sha256"] = package_sha256(payload)
    return canonical_copy(payload)


def baseline_package_payload(
    baseline: Union[VitalyProfile, Mapping[str, Any]], *, package_version: Optional[str] = None
) -> Dict[str, Any]:
    """Freeze the current profile as the first, no-op knowledge package.

    A no-op baseline is not a learning candidate and therefore intentionally
    does not pass through ``validate_knowledge_patch`` (which rejects empty
    candidate changes).  It still has the identical immutable envelope and can
    be pinned to new cases before the first accepted update exists.
    """

    profile = profile_to_dict(baseline)
    version = _nonempty_text(package_version or profile["version"], "package_version", limit=200)
    profile["version"] = version
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "package_version": version,
        "baseline_profile_version": profile_to_dict(baseline)["version"],
        "patch": {
            "schema_version": 1,
            "summary": "",
            "analysis_rules": [],
            "question_patterns": [],
            "meetings": [],
        },
        "analysis_rules": [],
        "profile": profile,
    }
    payload["package_sha256"] = package_sha256(payload)
    return canonical_copy(payload)


def verify_immutable_package(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Reject a changed payload rather than allowing a stale digest to be reused."""

    if not isinstance(payload, Mapping):
        raise InputError("Пакет знаний должен быть объектом")
    required = {"schema_version", "package_version", "baseline_profile_version", "patch", "profile", "package_sha256"}
    missing = sorted(required - set(payload))
    if missing:
        raise InputError("Пакет знаний неполный: {}".format(", ".join(missing)))
    if payload.get("schema_version") != 1:
        raise InputError("У пакета знаний неподдерживаемая схема")
    _nonempty_text(payload.get("package_version"), "package_version", limit=200)
    _nonempty_text(payload.get("baseline_profile_version"), "baseline_profile_version", limit=200)
    if not isinstance(payload.get("patch"), Mapping):
        raise InputError("В пакете отсутствует патч")
    if "analysis_rules" in payload and not isinstance(payload["analysis_rules"], list):
        raise InputError("Накопленные правила анализа имеют неверную структуру")
    expected = payload.get("package_sha256")
    if not isinstance(expected, str) or not expected:
        raise InputError("У пакета знаний нет контрольной суммы")
    actual = package_sha256(payload)
    if expected != actual:
        raise InputError("Контрольная сумма пакета знаний не совпадает")
    profile = payload.get("profile")
    if not isinstance(profile, Mapping):
        raise InputError("В пакете отсутствует профиль")
    validated_profile = VitalyProfile.from_dict(canonical_copy(profile))
    if validated_profile.version != payload["package_version"]:
        raise InputError("Версия профиля не совпадает с версией пакета знаний")
    return canonical_copy(dict(payload))


# Fixed, synthetic examples.  They never become evidence for a generated rule;
# the runtime only passes them to the separate verifier/evaluator.
_FIXED_REGRESSION_FIXTURES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "false_contradiction_definition",
        "category": "false_contradiction",
        "material": "Слайд 2: потенциал 20%; слайд 7: подтверждённый эффект 8% после исключений.",
        "expected": {"must_not_flag": ["contradiction"], "reason": "разные определения показателя"},
    },
    {
        "id": "true_contradiction_same_scope",
        "category": "true_contradiction",
        "material": "Один и тот же ТОФ и июль: на слайде 3 простой 12 ч, на слайде 8 — 19 ч.",
        "expected": {"must_flag": ["contradiction"], "reason": "один период и периметр"},
    },
    {
        "id": "different_periods_not_contradiction",
        "category": "periods",
        "material": "План на Q4 — 90 млн руб.; факт за август — 12 млн руб.",
        "expected": {"must_not_flag": ["contradiction"], "must_ask": ["period_comparison"]},
    },
    {
        "id": "convertible_units",
        "category": "units",
        "material": "На одном слайде 120 минут, на другом 2 часа по тому же эпизоду.",
        "expected": {"must_not_flag": ["contradiction"], "must_note": ["unit_conversion"]},
    },
    {
        "id": "incompatible_units",
        "category": "units",
        "material": "Эффект в рублях сопоставлен с количеством ремонтов без формулы перевода.",
        "expected": {"must_flag": ["unsupported_comparison"]},
    },
    {
        "id": "image_caption_context",
        "category": "image_captions",
        "material": "Подпись диаграммы: «доля 1/3 после фильтра по критичности»; текст слайда использует ту же долю.",
        "expected": {"must_not_flag": ["missing_share"], "must_use": ["image_caption"]},
    },
    {
        "id": "image_caption_conflict",
        "category": "image_captions",
        "material": "Подпись диаграммы: «данные за 2025», текст слайда называет их фактом за 2026.",
        "expected": {"must_flag": ["period_mismatch"], "must_use": ["image_caption"]},
    },
    {
        "id": "unsupported_figure",
        "category": "unsupported_figures",
        "material": "Заявлена экономия 42 млн руб., но нет источника, базы, формулы и периметра.",
        "expected": {"must_flag": ["unsupported_figure"], "must_ask": ["evidence_scope", "effect_measurement"]},
    },
    {
        "id": "confirmed_figure",
        "category": "unsupported_figures",
        "material": "Экономия 4,2 млн: источник ledger G-42, формула, период и периметр приведены на слайде.",
        "expected": {"must_not_flag": ["unsupported_figure"]},
    },
    {
        "id": "ownership_and_deadline",
        "category": "ownership_deadline",
        "material": "Команда «доработает» предложение; владельца, результата и даты нет.",
        "expected": {"must_ask": ["ownership_timing"], "must_flag": ["missing_owner_or_deadline"]},
    },
    {
        "id": "implementation_mechanism",
        "category": "mechanism",
        "material": "Обещано снизить простой, но не описано изменение процесса, системы или оборудования.",
        "expected": {"must_ask": ["mechanism"], "must_flag": ["missing_mechanism"]},
    },
    {
        "id": "ambiguous_speaker",
        "category": "speaker_attribution",
        "material": "Транскрипт без меток говорящих; один Виталий упомянут в пересказе.",
        "expected": {"must_not_use": ["behavioural_evidence"], "must_ask": ["speaker_attribution"]},
    },
    {
        "id": "style_topic_coverage",
        "category": "style_coverage",
        "material": "Сравнить первые десять вопросов с ожидаемыми тематическими группами holdout.",
        "expected": {"minimum_coverage": 0.7, "must_not_degrade": True},
    },
)


def fixed_regression_fixtures() -> List[Dict[str, Any]]:
    """Return an independent copy of the fixed evaluator-only regression set."""

    return canonical_copy(list(_FIXED_REGRESSION_FIXTURES))


def regression_fixture_digest(fixtures: Optional[Sequence[Mapping[str, Any]]] = None) -> str:
    """Hash the exact fixed inputs stored alongside every verifier result."""

    value = fixed_regression_fixtures() if fixtures is None else canonical_copy(list(fixtures))
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _result_index(
    results: Sequence[Mapping[str, Any]], fixtures: Sequence[Mapping[str, Any]], label: str
) -> Dict[str, Dict[str, Any]]:
    expected_ids = {str(item["id"]) for item in fixtures}
    indexed: Dict[str, Dict[str, Any]] = {}
    for raw in results:
        if not isinstance(raw, Mapping):
            raise InputError("Результат {} должен быть объектом".format(label))
        scenario_id = _nonempty_text(raw.get("scenario_id"), "scenario_id", limit=200)
        if scenario_id in indexed:
            raise InputError("В результатах {} повторяется сценарий {}".format(label, scenario_id))
        indexed[scenario_id] = canonical_copy(dict(raw))
    missing = sorted(expected_ids - set(indexed))
    unknown = sorted(set(indexed) - expected_ids)
    if missing or unknown:
        parts = []
        if missing:
            parts.append("нет: {}".format(", ".join(missing)))
        if unknown:
            parts.append("лишние: {}".format(", ".join(unknown)))
        raise InputError("Набор результатов {} не совпадает с регрессией ({})".format(label, "; ".join(parts)))
    return indexed


def _as_score(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError("{} должен быть числом".format(field))
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise InputError("{} должен быть в диапазоне 0..1".format(field))
    return score


def _critical_errors(result: Mapping[str, Any], label: str) -> List[str]:
    raw = result.get("critical_error_ids", [])
    if not isinstance(raw, list) or any(not isinstance(item, str) or not item.strip() for item in raw):
        raise InputError("{}.critical_error_ids должен быть списком непустых строк".format(label))
    if len(set(raw)) != len(raw):
        raise InputError("{}.critical_error_ids содержит повторы".format(label))
    return sorted(raw)


def regression_gate(
    baseline_results: Sequence[Mapping[str, Any]],
    candidate_results: Sequence[Mapping[str, Any]],
    *,
    fixtures: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compare equal evaluator runs and return a deterministic release decision.

    Each result needs ``scenario_id``, boolean ``passed``, a 0..1
    ``target_score`` and ``critical_error_ids``.  The special style scenario
    additionally needs ``style_coverage``.  The gate only passes if every
    candidate scenario passes, no new critical error appears, at least one
    target score improves, and style coverage is >=70% without regression.
    """

    active_fixtures = list(fixtures) if fixtures is not None else fixed_regression_fixtures()
    baseline = _result_index(baseline_results, active_fixtures, "базовой версии")
    candidate = _result_index(candidate_results, active_fixtures, "кандидата")

    failed: List[str] = []
    new_critical: List[Dict[str, Any]] = []
    improved: List[str] = []
    baseline_scores: Dict[str, float] = {}
    candidate_scores: Dict[str, float] = {}
    for fixture in active_fixtures:
        scenario_id = str(fixture["id"])
        before = baseline[scenario_id]
        after = candidate[scenario_id]
        if not isinstance(before.get("passed"), bool) or not isinstance(after.get("passed"), bool):
            raise InputError("Для {} нужен логический результат passed".format(scenario_id))
        baseline_scores[scenario_id] = _as_score(before.get("target_score"), "{}.target_score".format(scenario_id))
        candidate_scores[scenario_id] = _as_score(after.get("target_score"), "{}.target_score".format(scenario_id))
        before_errors = set(_critical_errors(before, "baseline.{}".format(scenario_id)))
        after_errors = set(_critical_errors(after, "candidate.{}".format(scenario_id)))
        appeared = sorted(after_errors - before_errors)
        if appeared:
            new_critical.append({"scenario_id": scenario_id, "error_ids": appeared})
        if not after["passed"]:
            failed.append(scenario_id)
        if candidate_scores[scenario_id] > baseline_scores[scenario_id]:
            improved.append(scenario_id)

    style_id = "style_topic_coverage"
    if style_id not in baseline or style_id not in candidate:
        raise InputError("В регрессии отсутствует проверка покрытия стиля")
    baseline_coverage = _as_score(baseline[style_id].get("style_coverage"), "baseline.style_coverage")
    candidate_coverage = _as_score(candidate[style_id].get("style_coverage"), "candidate.style_coverage")
    style_passing = candidate_coverage >= 0.7 and candidate_coverage >= baseline_coverage

    reasons: List[str] = []
    if failed:
        reasons.append("Кандидат не прошёл обязательные сценарии: {}".format(", ".join(failed)))
    if new_critical:
        reasons.append("Кандидат добавил критические ошибки")
    if not improved:
        reasons.append("Кандидат не улучшил ни одного целевого сценария")
    if candidate_coverage < 0.7:
        reasons.append("Покрытие тематических групп ниже 70%")
    elif candidate_coverage < baseline_coverage:
        reasons.append("Покрытие тематических групп ухудшилось")

    return {
        "passing": not reasons,
        "reasons": reasons,
        "failed_scenarios": failed,
        "new_critical_errors": new_critical,
        "improved_scenarios": improved,
        "baseline_scores": baseline_scores,
        "candidate_scores": candidate_scores,
        "baseline_style_coverage": baseline_coverage,
        "candidate_style_coverage": candidate_coverage,
        "style_passing": style_passing,
        "fixture_ids": [str(item["id"]) for item in active_fixtures],
    }


# Explicit aliases keep the storage/runtime layer readable while retaining a
# single canonical implementation.
build_knowledge_package = immutable_package_payload
evaluate_regression_gate = regression_gate
