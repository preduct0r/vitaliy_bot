"""Conservative, fixed-label projection of source-verified repair interests.

This is a narrow foundation, not a semantic or production safety guard. The
caller chooses the source index, including any holdout exclusions. Output
contains no source text; interest identifiers are passed through as supplied.
"""
from __future__ import annotations

import hashlib
import json
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


# Descriptions are application-owned; neither model may write generator text.
DIMENSION_VERSION = "repair-analytics-dimensions.v2"
_DIMENSIONS = {
    "repair_episode_count_by_equipment_period": ("maintenance_history_repeated_episodes", "Число и виды ремонтных эпизодов по оборудованию и периодам."),
    "repair_repeat_interval": ("maintenance_history_repeated_episodes", "Интервалы между повторными ремонтными эпизодами оборудования."),
    "repair_episode_causes": ("maintenance_history_repeated_episodes", "Причины отказов и повторных ремонтов оборудования."),
    "repair_cost_by_period_asset_work": ("mtr_cost_supply", "Затраты на ремонт по периодам, оборудованию и видам работ."),
    "repair_cost_per_action": ("mtr_cost_supply", "Стоимость отдельного ремонтного действия с различением заказа и ремонтного эпизода."),
    "repair_plan_fact_reasons": ("maintenance_plan_fact", "Сопоставление плана и факта ремонта и причин отклонений."),
    "repair_labor_norm_actual": ("labor_norm_observed_measurement", "Сопоставление нормативных и фактических трудозатрат на ремонт."),
    "repair_techcard_registry_completeness": ("techcard_coverage_verification", "Полнота ремонтных технологических карт относительно независимого реестра."),
    "repair_verification_status_owner": ("techcard_coverage_verification", "Статус и ответственный за проверку ремонтной документации и данных."),
    "repair_material_readiness": ("maintenance_scheduling_resource_load", "Готовность материалов до запланированного ремонта."),
    "repair_personnel_readiness": ("maintenance_scheduling_resource_load", "Готовность персонала до запланированного ремонта."),
    "repair_resource_load_schedule": ("maintenance_scheduling_resource_load", "График ремонтов и загрузка ремонтных ресурсов."),
    "repair_material_cost_supply": ("mtr_cost_supply", "Стоимость и поставки материалов для ремонта."),
    "repair_downtime": ("equipment_downtime", "Простои оборудования, связанные с ремонтом."),
    "repair_criticality_decisions": ("equipment_criticality", "Связь критичности оборудования с решениями о ремонте."),
    "repair_data_quality": ("maintenance_data_quality", "Качество и полнота данных о ремонтах оборудования."),
    "repair_implementation_measurability": ("implementation_measurability", "Измеримость результатов внедрения изменений в ремонте."),
}
_DIMENSIONS_V1 = dict(_DIMENSIONS)
_DIMENSIONS["repair_techcard_equipment_operations_coverage"] = (
    "techcard_coverage_verification",
    "Сопоставление ремонтируемого оборудования и операций с наличием и различиями технологических карт.",
)
_METHODOLOGY = {
    "repair_techcard_equipment_operations_coverage": (
        "Для вывода о полноте нужен независимый базовый перечень оборудования и операций; "
        "это метод проверки, а не обязательное содержание исходной цитаты."
    ),
}
_EVIDENCE_CONTRACT = (
    "Отличайте прямое требование в источнике от обоснованного аналитического вывода "
    "и пересказа третьего лица; не приписывайте вывод или пересказ как прямую цитату. "
    "Методологические условия не обязаны дословно присутствовать в запросе. "
    "Не относящаяся к измерению ссылка не является противоречием; подлинный конфликт "
    "одного предмета и области сохраняет неопределённость."
)
_CATALOGS = {"repair-analytics-dimensions.v1": _DIMENSIONS_V1, DIMENSION_VERSION: _DIMENSIONS}
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def project_semantic_dimensions(profile: Mapping[str, Any], index: Mapping[str, Any], *,
                                classifier: Callable[[dict], Any], verifier: Callable[[dict], Any]) -> dict:
    """Project verified private evidence; caller must isolate independent model processes.

    The supplied source index defines the boundary, including holdout exclusions.
    Every excerpt and source hash is checked before either component is called.
    Audit evidence is private and must never be passed to a generator.
    """
    if not callable(classifier) or not callable(verifier) or classifier is verifier:
        raise ValueError("independent classifier and verifier required")
    try:
        if not index.get("version") or profile.get("corpus_version") != index["version"]:
            raise ValueError("corpus version mismatch")
        sources = {}
        texts = {}
        for source in index["sources"]:
            sid, path, digest = source["id"], Path(source["path"]), source["sha256"]
            if (not isinstance(sid, str) or not _SAFE_ID.fullmatch(sid) or sid in sources
                    or not path.is_absolute() or not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
                raise ValueError("invalid source or hash")
            if source.get("date") in {"2026-10-26", "2026-10-28", "2026-10-30"}:
                raise ValueError("heldout source in training index")
            sources[sid] = source
            texts[sid] = path.read_text(encoding="utf-8")
        interests = profile["interests"]
        if not isinstance(interests, list):
            raise ValueError("invalid interests")
        if interests and not assess_provenance(profile, index)["verified"]:
            raise ValueError("source provenance verification failed")
        by_interest = {}
        for item in interests:
            ident = item["id"]
            if not isinstance(ident, str) or not _SAFE_ID.fullmatch(ident) or ident in by_interest:
                raise ValueError("invalid interest id")
            evidence = {}
            for entry in item["evidence"]:
                key = (entry["source_id"], entry["locator"])
                if key in evidence:
                    raise ValueError("duplicate evidence")
                start, end = map(int, re.fullmatch(r"chars:(\d+)-(\d+)", entry["locator"]).groups())
                evidence[key] = {**{k: entry[k] for k in ("source_id", "locator", "fragment", "attribution")},
                                 "sha256": sources[entry["source_id"]]["sha256"],
                                 "context": texts[entry["source_id"]][max(0, start - 500):end + 300]}
            by_interest[ident] = {"interest_id": ident, "requirement": item["requirement"], "evidence": evidence}
    except (KeyError, TypeError, AttributeError, OSError) as exc:
        raise ValueError("invalid profile or source index") from exc

    catalog = [{"dimension_id": key, "topic_id": topic, "description": desc,
                **({"methodology": _METHODOLOGY[key]} if key in _METHODOLOGY else {})}
               for key, (topic, desc) in _DIMENSIONS.items()]
    private_input = {"version": DIMENSION_VERSION, "dimensions": catalog, "evidence_contract": _EVIDENCE_CONTRACT,
                     "interests": [{"interest_id": i["interest_id"], "requirement": i["requirement"],
                                    "evidence": list(i["evidence"].values())} for i in by_interest.values()]}
    try:
        candidates = classifier(private_input) if interests else []
    except Exception as exc:
        raise ValueError("private classifier failed") from exc
    if not isinstance(candidates, list):
        raise ValueError("invalid classifier response")
    groups = {}
    seen = set()
    for c in candidates:
        try:
            ident, dim, topic, sid, loc = (c[k] for k in ("interest_id", "dimension_id", "topic_id", "source_id", "locator"))
            key = (ident, dim, sid, loc)
            if (key in seen or ident not in by_interest or dim not in _DIMENSIONS
                    or topic != _DIMENSIONS[dim][0] or (sid, loc) not in by_interest[ident]["evidence"]):
                raise ValueError("invalid or duplicate candidate")
            seen.add(key)
            groups.setdefault((ident, dim), []).append((sid, loc, c.get("confidence")))
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid candidate") from exc
    confirmed, unconfirmed = {}, []
    for (ident, dim), refs in groups.items():
        item = by_interest[ident]
        topic, desc = _DIMENSIONS[dim]
        checked = []
        for sid, loc, confidence in refs:
            evidence = item["evidence"][(sid, loc)]
            try:
                verdict = verifier({"interest_id": ident, "requirement": item["requirement"],
                                    "dimension_id": dim, "topic_id": topic, "description": desc,
                                    "methodology": _METHODOLOGY.get(dim, ""), "evidence_contract": _EVIDENCE_CONTRACT,
                                    "evidence": dict(evidence), "all_evidence": list(item["evidence"].values())})
            except Exception:
                verdict = None
            supported = (confidence == "high" and isinstance(verdict, Mapping)
                         and verdict.get("supported") is True and verdict.get("confidence") == "high"
                         and not verdict.get("conflicting"))
            conflicting = isinstance(verdict, Mapping) and bool(verdict.get("conflicting"))
            checked.append((evidence, supported, conflicting))
        # A contradiction holds the binding; a merely unsupported reference is
        # retained separately without erasing independently supported excerpts.
        conflict = any(conflicting for _, _, conflicting in checked)
        rejected = [e for e, ok, _ in checked if conflict or not ok]
        if rejected:
            unconfirmed.append({"interest_id": ident, "dimension_id": dim, "topic_id": topic,
                                "evidence": [dict(e) for e in rejected]})
        accepted = [e for e, ok, _ in checked if ok and not conflict]
        if not accepted:
            continue
        row = confirmed.setdefault(dim, {"dimension_id": dim, "topic_id": topic, "description": desc,
                                         "interest_ids": set(), "source_ids": set(), "evidence": [], "meetings": set()})
        row["interest_ids"].add(ident)
        for evidence in accepted:
            sid = evidence["source_id"]
            row["source_ids"].add(sid)
            row["evidence"].append({"interest_id": ident, **evidence})
            source = sources[sid]
            row["meetings"].add(source.get("meeting_id") or source.get("date") or sid)
    rows = []
    for row in confirmed.values():
        row["interest_ids"] = sorted(row["interest_ids"])
        row["source_ids"] = sorted(row["source_ids"])
        row["independent_meeting_count"] = len(row.pop("meetings"))
        rows.append(row)
    def digest(value):
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"version": DIMENSION_VERSION, "corpus_version": index["version"],
            "profile_version": profile.get("version"), "profile_sha256": digest(profile),
            "index_sha256": digest(index), "dimensions": rows, "unconfirmed": unconfirmed}


def dimensions_for_generator(projection: Mapping[str, Any]) -> dict:
    """Rebuild a minimal payload from the fixed catalog, never copy private text."""
    version = projection.get("version")
    if version not in _CATALOGS:
        raise ValueError("unsupported dimension version")
    catalog = _CATALOGS[version]
    rows = []
    for row in projection["dimensions"]:
        dim = row["dimension_id"]
        if dim not in catalog or row.get("topic_id") != catalog[dim][0]:
            raise ValueError("invalid dimension")
        ids, sources = row["interest_ids"], row["source_ids"]
        if not all(isinstance(i, str) and _SAFE_ID.fullmatch(i) for i in [*ids, *sources]):
            raise ValueError("unsafe identifier")
        rows.append({"dimension_id": dim, "topic_id": catalog[dim][0],
                     "description": catalog[dim][1], "interest_ids": list(ids), "source_ids": list(sources)})
    return {"version": version, "dimensions": rows}
