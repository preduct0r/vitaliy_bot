from __future__ import annotations

import copy

import pytest

from tof_reliability.vitaly_bot.learning import (
    baseline_package_payload,
    canonical_json,
    compose_profile,
    evaluate_regression_gate,
    fixed_regression_fixtures,
    immutable_package_payload,
    normalize_text,
    normalized_text_hash,
    validate_knowledge_patch,
    verify_immutable_package,
)
from tof_reliability.vitaly_bot.models import InputError
from tof_reliability.vitaly_bot.profile import load_default_profile


def _patch(*, cited_meetings=None):
    meetings = cited_meetings or ["2026-06-03", "meeting-2026-09-16"]
    return {
        "schema_version": 1,
        "summary": "Уточнить проверку разных определений показателя.",
        "meetings": [
            {
                "id": "meeting-2026-09-16",
                "date": "2026-09-16",
                "title": "Проверка презентации",
                "source": "knowledge_archive/meetings/meeting-2026-09-16/source.txt",
                "person": "vitaly_hard_negotiator",
                "use_for_simulation": True,
                "dedupe_key": "meeting-2026-09-16",
            }
        ],
        "analysis_rules": [
            {
                "id": "different_definitions",
                "transfer_key": "compare_definition_before_conflict",
                "scope": "narrow",
                "trigger": "Два числа имеют разные определения или базы сравнения.",
                "guidance": "Не называй их противоречием, пока не сопоставлены период и определение.",
                "expected_improvement": "Снять ложное замечание о противоречии в этом классе слайдов.",
                "source_ids": ["feedback-001"],
                "evidence_references": [
                    {
                        "source_id": "feedback-001",
                        "source_document_id": "artifact-001",
                        "fragment": "Слайд 2 — потенциал; слайд 7 — подтверждённый эффект.",
                    }
                ],
            }
        ],
        "question_patterns": [
            {
                "operation": "replace",
                "topic_id": "evidence_scope",
                "name": "Границы определения показателя",
                "behaviour": "Сначала просит назвать определение показателя и базу сравнения, затем проверяет сопоставимость.",
                "style_mode": "cross_examine",
                "followup_chain": ["Как определён каждый показатель?", "Одинаковы ли период и периметр?"],
                "evidence_meetings": meetings,
                "evidence_references": [
                    {
                        "meeting_id": "2026-06-03",
                        "source": "discussions/06_03_тоф_синк_виталий_original.txt",
                        "time_range": "10:00–10:20",
                    },
                    {
                        "meeting_id": "meeting-2026-09-16",
                        "source": "knowledge_archive/meetings/meeting-2026-09-16/source.txt",
                        "time_range": "02:00–02:20",
                    },
                ],
            }
        ],
    }


def _accepted_meetings():
    profile = load_default_profile()
    return [meeting["id"] for meeting in profile.meeting_index] + ["meeting-2026-09-16"]


def test_normalized_text_hash_and_canonical_json_are_stable():
    assert normalize_text("  ТЕСТ\u00a0\nтекст\u200b ") == "тест текст"
    assert normalized_text_hash("Тест\nтекст") == normalized_text_hash("  тест  текст ")
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_patch_requires_two_accepted_meetings_for_behavioral_pattern():
    profile = load_default_profile()
    patch = _patch(cited_meetings=["meeting-2026-09-16"])
    patch["question_patterns"][0]["evidence_references"] = [
        patch["question_patterns"][0]["evidence_references"][1]
    ]

    with pytest.raises(InputError, match="минимум на две встречи"):
        validate_knowledge_patch(
            patch,
            accepted_source_ids=["feedback-001"],
            accepted_meeting_ids=_accepted_meetings(),
            known_topic_ids=profile.topic_ids,
        )


def test_knowledge_package_is_traceable_reproducible_and_composes_profile():
    baseline = load_default_profile()
    patch = _patch()
    package = immutable_package_payload(
        baseline,
        patch,
        package_version="knowledge.v1.2.0",
        accepted_source_ids=(item for item in ["feedback-001"]),
        accepted_meeting_ids=(item for item in _accepted_meetings()),
    )

    assert package == immutable_package_payload(
        baseline,
        patch,
        package_version="knowledge.v1.2.0",
        accepted_source_ids=["feedback-001"],
        accepted_meeting_ids=_accepted_meetings(),
    )
    assert package["profile"]["version"] == "knowledge.v1.2.0"
    assert "meeting-2026-09-16" in {item["id"] for item in package["profile"]["meeting_index"]}
    assert "evidence_scope" in {item["topic_id"] for item in package["profile"]["question_patterns"]}
    assert verify_immutable_package(package)["package_sha256"] == package["package_sha256"]

    composed = compose_profile(
        baseline,
        patch,
        release_version="knowledge.v1.2.0",
        accepted_source_ids=["feedback-001"],
        accepted_meeting_ids=_accepted_meetings(),
    )
    assert composed.version == "knowledge.v1.2.0"
    assert composed.topic_ids.count("evidence_scope") == 1
    safe_context = composed.prompt_context()
    assert "definition" not in str(safe_context).casefold()
    assert "knowledge_archive" not in str(safe_context)

    tampered = copy.deepcopy(package)
    tampered["profile"]["simulator_label"] = "подмена"
    with pytest.raises(InputError, match="Контрольная сумма"):
        verify_immutable_package(tampered)


def test_new_pattern_cannot_cite_meeting_missing_from_profile_index():
    baseline = load_default_profile()
    patch = _patch()
    patch["meetings"] = []
    with pytest.raises(InputError, match="отсутствуют в meetings"):
        compose_profile(
            baseline,
            patch,
            release_version="knowledge.missing-meeting",
            accepted_source_ids=["feedback-001"],
            accepted_meeting_ids=_accepted_meetings(),
        )


def test_baseline_package_can_be_pinned_before_the_first_learning_release():
    baseline = load_default_profile()
    package = baseline_package_payload(baseline)

    assert package["package_version"] == baseline.version
    assert package["patch"]["analysis_rules"] == []
    assert verify_immutable_package(package)["profile"]["version"] == baseline.version


def _regression_results(*, score: float, coverage: float):
    rows = []
    for fixture in fixed_regression_fixtures():
        row = {
            "scenario_id": fixture["id"],
            "passed": True,
            "target_score": score,
            "critical_error_ids": [],
        }
        if fixture["id"] == "style_topic_coverage":
            row["style_coverage"] = coverage
        rows.append(row)
    return rows


def test_fixed_regression_gate_needs_improvement_without_new_critical_errors():
    fixtures = fixed_regression_fixtures()
    categories = {item["category"] for item in fixtures}
    assert len(fixtures) >= 12
    assert {
        "false_contradiction",
        "true_contradiction",
        "periods",
        "units",
        "image_captions",
        "unsupported_figures",
        "ownership_deadline",
        "mechanism",
        "speaker_attribution",
    } <= categories

    baseline = _regression_results(score=0.50, coverage=0.70)
    candidate = _regression_results(score=0.50, coverage=0.80)
    candidate[0]["target_score"] = 0.75
    decision = evaluate_regression_gate(baseline, candidate)
    assert decision["passing"] is True
    assert decision["improved_scenarios"] == ["false_contradiction_definition"]

    candidate[1]["critical_error_ids"] = ["invented_contradiction"]
    rejected = evaluate_regression_gate(baseline, candidate)
    assert rejected["passing"] is False
    assert rejected["new_critical_errors"] == [
        {"scenario_id": "true_contradiction_same_scope", "error_ids": ["invented_contradiction"]}
    ]
