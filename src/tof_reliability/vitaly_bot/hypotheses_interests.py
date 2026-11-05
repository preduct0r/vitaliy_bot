"""Private, versioned analytical-interest evidence and independent holdout scoring.

The index stores references, never transcript text. Profiles and provenance reports
are private artifacts; only the explicitly projected generator context may be sent.
"""
from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

HOLDOUT_DATES = ("2026-10-26", "2026-10-28", "2026-10-30")
ATTRIBUTIONS = {"direct_request", "interpretation", "third_party_paraphrase", "third_party"}


def _required(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def _date(value: Any) -> str:
    value = _required(value, "date")
    try:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError()
    except ValueError as exc:
        raise ValueError("invalid ISO date") from exc
    return value


def index_corpus(sources: Sequence[Mapping[str, Any]], *, version: str) -> dict:
    version = _required(version, "version")
    result, seen_hashes, seen_ids = [], {}, {}
    for source in sources:
        if source.get("status") != "accepted":
            continue
        sid = _required(source.get("id"), "source id")
        digest = _required(source.get("sha256"), "sha256").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid sha256")
        when = _date(source.get("date"))
        kind = _required(source.get("kind"), "kind")
        if kind not in {"baseline", "transcript"}:
            raise ValueError("invalid source kind")
        if sid in seen_ids and seen_ids[sid] != digest:
            raise ValueError("source id has conflicting hashes")
        seen_ids[sid] = digest
        if digest in seen_hashes:
            if seen_hashes[digest] != (when, kind):
                raise ValueError("duplicate hash has conflicting metadata")
            continue
        seen_hashes[digest] = (when, kind)
        row = {"id": sid, "date": when, "kind": kind, "sha256": digest}
        if source.get("path"):
            row["path"] = str(source["path"])
        result.append(row)
    result.sort(key=lambda x: (x["date"], x["id"]))
    return {"version": version, "sources": result}


def prepare_private_corpus(sources: Sequence[Mapping[str, Any]], *, version: str) -> dict:
    # Verify every accepted input including duplicates; a stale duplicate is not evidence.
    for source in sources:
        if source.get("status") != "accepted":
            continue
        path = Path(_required(source.get("path"), "path"))
        if not path.is_absolute():
            raise ValueError("source path must be absolute")
        if hashlib.sha256(path.read_bytes()).hexdigest() != str(source.get("sha256", "")).lower():
            raise ValueError(f"sha256 mismatch for source {source.get('id')}")
    return index_corpus(sources, version=version)


def chronological_holdout(index: Mapping[str, Any], dates: Sequence[str] = HOLDOUT_DATES) -> dict:
    dates_set = {_date(d) for d in dates}
    if not dates_set:
        raise ValueError("holdout dates required")
    sources = index["sources"]
    holdout = [s for s in sources if s["date"] in dates_set]
    if {s["date"] for s in holdout} != dates_set:
        raise ValueError("missing holdout date")
    later = [s for s in sources if s["date"] not in dates_set and _date(s["date"]) >= min(dates_set)]
    # Preserve the original fixed evaluation's fail-closed contract.
    if dates_set == set(HOLDOUT_DATES) and later:
        raise ValueError("training source violates chronological holdout")
    training = [s for s in sources if _date(s["date"]) < min(dates_set)]
    return {"version": index["version"], "training": training, "holdout": holdout, "excluded": later}


def validate_interest_map(profile: Mapping[str, Any], index: Mapping[str, Any]) -> None:
    _required(profile.get("version"), "interest map version")
    if profile.get("corpus_version") != index.get("version"):
        raise ValueError("corpus version mismatch")
    source_ids = {s["id"] for s in index["sources"]}
    items = profile.get("interests")
    if not isinstance(items, list) or not items:
        raise ValueError("interests required")
    ids = set()
    for item in items:
        ident = _required(item.get("id"), "interest id")
        if ident in ids:
            raise ValueError("duplicate interest id")
        ids.add(ident)
        _required(item.get("requirement"), "requirement")
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("evidence required")
        for entry in evidence:
            if entry.get("source_id") not in source_ids:
                raise ValueError("unknown source id")
            _required(entry.get("locator"), "locator")
            if entry.get("attribution") not in ATTRIBUTIONS:
                raise ValueError("invalid attribution")


def pilot_training_index(index: Mapping[str, Any]) -> dict:
    """Exclude the fixed holdout and all later sources from pilot generation."""
    return {"version": _required(index.get("version"), "version"),
            "sources": [s for s in index["sources"] if _date(s.get("date")) < min(HOLDOUT_DATES)]}


def generator_context(split: Mapping[str, Any], profile: Mapping[str, Any] | None = None) -> dict:
    if profile is None:
        return {"interests": []}
    index = {"version": split["version"], "sources": split["training"]}
    validate_interest_map(profile, index)
    if not assess_provenance(profile, index)["verified"]:
        raise ValueError("source provenance verification failed")
    sources = {s["id"]: s for s in split["training"]}
    result = []
    for item in profile["interests"]:
        evidence = []
        for entry in item["evidence"]:
            source = sources[entry["source_id"]]
            sid, locator = source["id"], entry["locator"]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", sid) or not re.fullmatch(r"chars:[0-9]+-[0-9]+", locator):
                raise ValueError("unsafe source identifier or locator")
            evidence.append({"source_id": sid, "date": _date(source["date"]),
                             "locator": locator, "attribution": entry["attribution"],
                             "sha256": source["sha256"], "reference": f"{sid}:{locator}"})
        result.append({"id": item["id"], "requirement": item["requirement"],
                       "attributions": sorted({e["attribution"] for e in evidence}), "evidence": evidence})
    return {"version": profile["version"], "corpus_version": split["version"], "interests": result}


def assess_provenance(profile: Mapping[str, Any], index: Mapping[str, Any]) -> dict:
    validate_interest_map(profile, index)
    by_id = {s["id"]: s for s in index["sources"]}
    checks = []
    for item in profile["interests"]:
        for evidence in item["evidence"]:
            source = by_id[evidence["source_id"]]
            ok, reason = False, "missing private path, fragment or character locator"
            match = re.fullmatch(r"chars:(\d+)-(\d+)", evidence["locator"])
            fragment = evidence.get("fragment")
            if source.get("path") and match and isinstance(fragment, str) and fragment:
                try:
                    data = Path(source["path"]).read_bytes()
                    if hashlib.sha256(data).hexdigest() != source["sha256"]:
                        reason = "source hash mismatch"
                    else:
                        text = data.decode("utf-8")
                        start, end = map(int, match.groups())
                        ok = start < end <= len(text) and text[start:end] == fragment
                        reason = "verified" if ok else "fragment or locator mismatch"
                except (OSError, UnicodeError):
                    reason = "source unreadable or not UTF-8"
            checks.append({"interest_id": item["id"], "source_id": source["id"], "locator": evidence["locator"],
                           "date": source.get("date"), "sha256": source.get("sha256"),
                           "attribution": evidence["attribution"], "verified": ok, "reason": reason})
    return {"corpus_version": index["version"], "interest_version": profile["version"], "verified": bool(checks) and all(c["verified"] for c in checks), "checks": checks}


def score_interest_holdout(expected: Sequence[str], predicted: Sequence[str]) -> dict:
    expected_set = {_required(i, "expected topic") for i in expected}
    first = list(predicted[:7])
    matched = sorted(expected_set.intersection(first))
    coverage = len(matched) / len(expected_set) if expected_set else 0.0
    return {"expected": sorted(expected_set), "first_seven": first, "matched": matched, "coverage": coverage, "passing": bool(expected_set) and coverage >= 0.7}


def fixed_evaluation_scenarios() -> list[dict]:
    cases = [
        ("no_files", "Нет файлов и активного кейса", "Предложить вопросы без обязательного файла"),
        ("scope", "Область: насосы одного цеха", "Сохранить ограничение области"),
        ("unknown_prior_analyses", "История предыдущих анализов неизвестна", "Явно указать неизвестную новизну"),
        ("partial_periods", "Данные доступны только за часть периода", "Отразить неполное покрытие и недостающие периоды"),
        ("missing_joins", "Не подтверждено соединение заказов с оборудованием", "Не обещать анализ без ключей"),
        ("units", "Сроки ремонта в днях, трудозатраты в человеко-часах", "Не смешивать длительность и трудоёмкость"),
        ("unconfirmed_numbers", "Озвучена неподтверждённая экономия", "Не выдавать оценку за факт"),
        ("repeated_repairs", "Проверить повторные ремонты", "Определить эпизод и не приравнивать заказ к эпизоду"),
        ("labor", "Сравнить трудоёмкость с техкартой", "Различать норму, факт и экспериментальную оценку"),
        ("criticality", "Приоритизировать критичное оборудование", "Указать наличие и происхождение критичности"),
        ("server_outage", "Опубликованное поколение сервера недоступно", "Сообщить о недоступности без локальной подмены"),
        ("planned_effect", "Оценить потенциальный эффект изменения графика", "Представить как исследование и указать недостающую методику"),
    ]
    return [{"id": tag, "tags": [tag], "input": question, "expected_checks": [check]} for tag, question, check in cases]
