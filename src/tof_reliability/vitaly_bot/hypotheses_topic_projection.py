"""Conservative, fixed-label projection of source-verified repair interests.

This is a narrow foundation, not a semantic or production safety guard. The
caller chooses the source index, including any holdout exclusions. Output
contains no source text; interest identifiers are passed through as supplied.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Callable, Mapping

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


def project_semantic_topics(
    profile: Mapping[str, Any], index: Mapping[str, Any], *,
    classifier: Callable[[dict], Any], verifier: Callable[[dict], Any],
) -> list[dict[str, str]]:
    """Run inside a private classification process, never in a generator context.

    The two independently provided callables see private summaries and verified
    excerpts. The caller is responsible for process isolation and for choosing
    independent implementations; this interface cannot establish model independence.
    Only high-confidence, independently supported, evidence-bound fixed labels
    are returned. Keep private classifier/verifier audit logs outside Git.
    """
    if not callable(classifier) or not callable(verifier) or classifier is verifier:
        raise ValueError("independent classifier and verifier required")
    try:
        if not isinstance(profile, Mapping) or not isinstance(index, Mapping):
            raise ValueError("invalid profile or index")
        if not index.get("version") or profile.get("corpus_version") != index["version"]:
            raise ValueError("corpus version mismatch")
        sources = index["sources"]
        if not isinstance(sources, list):
            raise ValueError("invalid sources")
        source_ids = set()
        for source in sources:
            sid, digest, path = source["id"], source["sha256"], source["path"]
            if not isinstance(sid, str) or not sid or sid in source_ids:
                raise ValueError("invalid or duplicate source")
            source_ids.add(sid)
            if (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or not Path(path).is_absolute()
                    or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest):
                raise ValueError("source hash verification failed")
        interests = profile["interests"]
        if not isinstance(interests, list):
            raise ValueError("invalid interests")
        if not interests:
            return []
        if not assess_provenance(profile, index)["verified"]:
            raise ValueError("source provenance verification failed")
        by_interest = {}
        for item in interests:
            ident = item["id"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", ident):
                raise ValueError("unsafe interest id")
            evidence = {}
            for entry in item["evidence"]:
                key = (entry["source_id"], entry["locator"])
                if key in evidence:
                    raise ValueError("duplicate evidence")
                evidence[key] = {k: entry[k] for k in ("source_id", "locator", "fragment", "attribution")}
            by_interest[ident] = {"interest_id": ident, "requirement": item["requirement"],
                                  "evidence": evidence}
    except (KeyError, TypeError, AttributeError, OSError) as exc:
        raise ValueError("invalid profile or source index") from exc

    labels = {topic_id: label for topic_id, label, _ in _TOPICS}
    private_input = {"interests": [
        {"interest_id": item["interest_id"], "requirement": item["requirement"],
         "evidence": [dict(e) for e in item["evidence"].values()]}
        for item in by_interest.values()],
        "topics": [{"topic_id": k, "description": v} for k, v in labels.items()]}
    try:
        candidates = classifier(private_input)
    except Exception as exc:
        raise ValueError("private classifier failed") from exc
    if not isinstance(candidates, list):
        raise ValueError("invalid classifier response")
    seen = set()
    checked = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("invalid candidate")
        try:
            ident, topic = candidate["interest_id"], candidate["topic_id"]
            source, locator = candidate["source_id"], candidate["locator"]
            key = (ident, topic)
            if key in seen:
                raise ValueError("duplicate candidate")
            seen.add(key)
            if (ident not in by_interest or topic not in labels or candidate.get("confidence") != "high"
                    or (source, locator) not in by_interest[ident]["evidence"]):
                raise ValueError("unsupported candidate")
            item = by_interest[ident]
            checked.append((ident, topic, item, item["evidence"][(source, locator)]))
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid candidate") from exc
    result = []
    for ident, topic, item, evidence in checked:
        try:
            verdict = verifier({"interest_id": ident, "requirement": item["requirement"],
                                "topic_id": topic, "description": labels[topic],
                                "evidence": dict(evidence),
                                "all_evidence": [dict(e) for e in item["evidence"].values()]})
        except Exception:
            continue
        if isinstance(verdict, Mapping) and verdict.get("supported") is True and verdict.get("confidence") == "high":
            result.append({"topic_id": topic, "interest_id": ident, "description": labels[topic]})
    return result
