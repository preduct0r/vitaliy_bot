"""Conservative, fixed-label projection of source-verified repair interests.

This is a narrow foundation, not a semantic or production safety guard. The
caller chooses the source index, including any holdout exclusions. Output
contains no source text; interest identifiers are passed through as supplied.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from .hypotheses_interests import assess_provenance

# Each topic requires every expression in its tuple in one sentence, in both
# the requirement and an independently verified evidence fragment. This is
# intentionally lossy and does not establish semantic equivalence.
_REPAIR = r"\b(?:ремонт\w*|техобслуживан\w*)"
_EQUIPMENT = r"\bоборудован\w*"
_TOPICS = (
    ("maintenance_history_repeated_episodes", "История ремонтов оборудования и повторные ремонтные эпизоды.",
     (_REPAIR, _EQUIPMENT, r"\b(?:истори\w*|повторн\w*)")),
    ("maintenance_plan_fact", "Сопоставление плановых и фактических показателей ремонта оборудования.",
     (_REPAIR, _EQUIPMENT, r"\bпланов\w*", r"\bфактическ\w*")),
    ("maintenance_scheduling_resource_load", "График ремонта оборудования и загрузка ресурсов.",
     (_REPAIR, _EQUIPMENT, r"\bграфик\w*", r"\bзагрузк\w*", r"\b(?:бригад\w*|ресурс\w*|персонал\w*)")),
    ("mtr_cost_supply", "Стоимость и поставки МТР для ремонта оборудования.",
     (_REPAIR, _EQUIPMENT, r"\bмтр\b", r"\bстоимост\w*", r"\bпостав\w*")),
    ("equipment_downtime", "Простои оборудования, связанные с ремонтом.",
     (_REPAIR, _EQUIPMENT, r"\bпросто\w*")),
    ("labor_norm_observed_measurement", "Сопоставление нормы трудозатрат на ремонт с измеренными трудозатратами.",
     (_REPAIR, r"\bтрудозатрат\w*", r"\bнорм\w*", r"\b(?:замер\w*|измерени\w*)")),
    ("equipment_criticality", "Критичность оборудования в контексте ремонта.",
     (_REPAIR, _EQUIPMENT, r"\bкритичност\w*")),
    ("techcard_coverage_verification", "Покрытие и проверка ремонтных технологических карт.",
     (_REPAIR, r"\bтехкарт\w*", r"\bпокрыти\w*", r"\b(?:проверк\w*|проверить|верификац\w*)")),
    ("maintenance_data_quality", "Качество данных о ремонтах оборудования.",
     (_REPAIR, _EQUIPMENT, r"\bкачеств\w*", r"\bданн\w*")),
    ("implementation_measurability", "Измеримость внедрения изменений в ремонте оборудования.",
     (_REPAIR, _EQUIPMENT, r"\bизмеримост\w*", r"\bвнедрени\w*")),
)
_UNCERTAIN = re.compile(r"\b(?:не|нет|без|возможно|возможн(?:о|ый|ая|ое)|предположительно|неизвестно)\b", re.IGNORECASE)


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    for sentence in re.split(r"[.!?;\n]", text.lower()):
        if not _UNCERTAIN.search(sentence) and all(re.search(pattern, sentence) for pattern in patterns):
            return True
    return False


def project_topics(profile: Mapping[str, Any], index: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return fixed topic descriptions only for verified and supported interests.

    Invalid provenance raises ValueError. An empty interest list produces no
    projection. All evidence is checked, even if only one fragment matches.
    """
    if not isinstance(profile, Mapping) or not isinstance(index, Mapping):
        raise ValueError("profile and index must be mappings")
    if profile.get("corpus_version") != index.get("version") or not index.get("version"):
        raise ValueError("corpus version mismatch")
    interests = profile.get("interests")
    if interests == []:
        return []
    try:
        sources = index["sources"]
        ids = [source["id"] for source in sources]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate source id")
        if not assess_provenance(profile, index)["verified"]:
            raise ValueError("source provenance verification failed")
        result = []
        for item in interests:
            for topic_id, description, patterns in _TOPICS:
                if _matches(item["requirement"], patterns) and any(
                    _matches(entry["fragment"], patterns) for entry in item["evidence"]
                ):
                    result.append({"topic_id": topic_id, "interest_id": item["id"], "description": description})
        return result
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid profile or source index") from exc
