from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pytest

from tof_reliability.vitaly_bot.learning import (
    baseline_package_payload,
    fixed_regression_fixtures,
    immutable_package_payload,
    public_release_summary,
    regression_fixture_digest,
)
from tof_reliability.vitaly_bot.learning_runtime import LearningManager, MEETING_CLASSIFICATION_SCHEMA
from tof_reliability.vitaly_bot.models import BotConfig, ModelError, ModelResult, StateError
from tof_reliability.vitaly_bot.profile import load_default_profile
from tof_reliability.vitaly_bot.service import VitalyService
from tof_reliability.vitaly_bot.storage import VitalyStore


def _config(tmp_path: Path) -> BotConfig:
    return BotConfig(
        token="test-token",
        state_dir=tmp_path / "state",
        knowledge_dir=tmp_path / "knowledge",
        codex_bin=Path("/usr/bin/codex"),
        min_free_bytes=0,
    )


class LearningRunner:
    def __init__(self) -> None:
        self.feedback_id = ""
        self.calls: list[str] = []

    async def run(
        self, prompt: str, schema: Dict[str, Any], image_paths: Optional[Sequence[Any]] = None
    ) -> ModelResult:
        del image_paths
        self.calls.append(prompt)
        properties = schema["properties"]
        if "result" in properties:
            return ModelResult(
                value={
                    "result": "example",
                    "reason": "Замечание подтверждено сопоставлением с исходным пунктом.",
                    "source_document_id": "audit",
                    "fragment": "Слайд 2: определения показателей различаются.",
                    "expected_improvement": "Не называть разные определения противоречием без сверки.",
                },
                stdout="",
                stderr="",
            )
        if "decision" in properties:
            return ModelResult(
                value={
                    "decision": "candidate",
                    "reason": "Есть одно подтверждённое узкое исправление.",
                    "source_ids": [self.feedback_id],
                    "scope": "analysis",
                    "expected_improvement": "Снимать ложный конфликт разных определений.",
                    "patch": {
                        "schema_version": 1,
                        "summary": "Перед противоречием сопоставлять определение, период и периметр.",
                        "analysis_rules": [
                            {
                                "id": "compare_definition_before_conflict",
                                "transfer_key": "compare_definition_before_conflict",
                                "scope": "narrow",
                                "trigger": "Два показателя названы одинаково, но имеют разные определения.",
                                "guidance": "Сначала сопоставь определение, период и периметр; не называй это противоречием без конфликта в одинаковом контуре.",
                                "expected_improvement": "Снять ложное замечание в сценарии разных определений.",
                                "source_ids": [self.feedback_id],
                                "evidence_references": [
                                    {
                                        "source_id": self.feedback_id,
                                        "source_document_id": "audit",
                                        "reference": "аудит G1",
                                        "fragment": "Слайд 2: разные определения показателя.",
                                    }
                                ],
                            }
                        ],
                        "question_patterns": [],
                        "meetings": [],
                    },
                },
                stdout="",
                stderr="",
            )
        if "verdict" in properties:
            baseline = []
            candidate = []
            for fixture in fixed_regression_fixtures():
                row = {
                    "scenario_id": fixture["id"],
                    "passed": True,
                    "target_score": 0.5,
                    "critical_error_ids": [],
                    "style_coverage": 0.7,
                    "first_ten_topics": [
                        "evidence_scope", "ownership_timing", "effect_measurement",
                        "priority_business_case", "implementation_readiness",
                    ],
                }
                baseline.append(dict(row))
                if fixture["id"] == "false_contradiction_definition":
                    row["target_score"] = 0.8
                if fixture["id"] == "style_topic_coverage":
                    row["style_coverage"] = 0.8
                candidate.append(row)
            return ModelResult(
                value={"verdict": "pass", "reason": "Все обязательные сценарии сохранены.", "baseline_results": baseline, "candidate_results": candidate},
                stdout="",
                stderr="",
            )
        raise AssertionError("unexpected schema")


def test_feedback_is_atomic_and_user_work_preempts_learning(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    service = VitalyService(store, load_default_profile(), _config(tmp_path), runner=LearningRunner())
    case = service.new_case(7)
    store.set_review(case["id"], {"readiness_score": 1})
    feedback = store.start_feedback(case["id"], 7, 70, "audit-1")
    background = store.submit_feedback(feedback["id"], "На слайдах разная методика, а не противоречие.")
    with pytest.raises(StateError, match="уже принято"):
        store.submit_feedback(feedback["id"], "Повтор")
    with pytest.raises(StateError, match="рассматривается"):
        store.start_feedback(case["id"], 7, 70, "audit-1")

    foreground = store.begin_job(case["id"], 7, "review")
    next_work = store.next_queued_work()
    assert next_work is not None
    assert next_work["id"] == foreground["id"]
    assert next_work["scope"] == "case"
    assert store.get_learning_job(background["id"])["status"] == "queued"
    assert store.feedback_in_flight(7, 70)["id"] == feedback["id"]


def test_user_learning_intake_preempts_queued_automatic_steps(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    service = VitalyService(store, load_default_profile(), _config(tmp_path), runner=LearningRunner())
    case = service.new_case(71)
    store.set_review(case["id"], {"readiness_score": 1})
    automatic = store.enqueue_learning_job(kind="knowledge_generate")
    feedback = store.start_feedback(case["id"], 71, 710, "audit-1")
    feedback_job = store.submit_feedback(feedback["id"], "Проверить подпись на слайде.")

    assert store.next_queued_work()["id"] == feedback_job["id"]
    store.finish_learning_job(feedback_job["id"])

    meeting = store.begin_meeting(72, 720)
    for value in ("2026-09-16", "Планёрка", "Виталий, руководитель"):
        meeting = store.submit_meeting_metadata(meeting["id"], value)
    transcript = tmp_path / "priority-meeting.txt"
    transcript.write_text("Виталий: где основание?", encoding="utf-8")
    _meeting, meeting_job = store.accept_meeting_source(
        meeting["id"], source_name="priority-meeting.txt", source_path=transcript, normalized_path=transcript,
        source_sha256="priority-source", normalized_sha256="priority-normalized",
    )
    assert meeting_job is not None
    assert store.next_queued_work()["id"] == meeting_job["id"]
    assert store.get_learning_job(automatic["id"])["status"] == "queued"


def test_meeting_intake_keeps_current_case_clean_and_exact_dedupes(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    service = VitalyService(store, load_default_profile(), _config(tmp_path), runner=LearningRunner())
    case = service.new_case(8)
    first = store.begin_meeting(8, 80)
    for value in ("2026-09-16", "Сверка проекта", "Виталий, руководитель проекта"):
        first = store.submit_meeting_metadata(first["id"], value)
    source = tmp_path / "meeting.txt"
    source.write_text("Виталий: где источник цифры?", encoding="utf-8")
    first, job = store.accept_meeting_source(
        first["id"],
        source_name="meeting.txt",
        source_path=source,
        normalized_path=source,
        source_sha256="same-source",
        normalized_sha256="same-normalized",
    )
    assert job is not None
    assert store.artifact_count(case["id"]) == 0
    assert store.meeting_in_flight(8, 80)["id"] == first["id"]
    store.set_meeting_result(first["id"], {
        "status": "accepted", "reason": "Тестовая атрибуция.", "identity_basis": "speaker_labels",
        "identity_confident": True, "duplicate_of": "", "continuation_of": "", "source_fragments": [], "observed_topics": [],
    })

    second = store.begin_meeting(8, 80)
    for value in ("2026-09-16", "Копия", "Виталий, руководитель проекта"):
        second = store.submit_meeting_metadata(second["id"], value)
    second, duplicate_job = store.accept_meeting_source(
        second["id"],
        source_name="copy.txt",
        source_path=source,
        normalized_path=source,
        source_sha256="same-source",
        normalized_sha256="same-normalized",
    )
    assert duplicate_job is None
    assert second["status"] == "duplicate"
    assert store.artifact_count(case["id"]) == 0


def test_normalized_meeting_duplicate_does_not_enqueue_second_model_run(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    VitalyService(store, load_default_profile(), _config(tmp_path), runner=LearningRunner())
    source = tmp_path / "meeting.txt"
    source.write_text("Виталий: где источник цифры?", encoding="utf-8")
    first = store.begin_meeting(81, 810)
    for value in ("2026-09-16", "Сверка", "Виталий, руководитель"):
        first = store.submit_meeting_metadata(first["id"], value)
    first, first_job = store.accept_meeting_source(
        first["id"], source_name="a.txt", source_path=source, normalized_path=source,
        source_sha256="source-a", normalized_sha256="normalized-same",
    )
    assert first_job is not None
    store.set_meeting_result(first["id"], {
        "status": "accepted", "reason": "Тестовая атрибуция.", "identity_basis": "speaker_labels",
        "identity_confident": True, "duplicate_of": "", "continuation_of": "", "source_fragments": [], "observed_topics": [],
    })
    second = store.begin_meeting(81, 810)
    for value in ("2026-09-16", "Переэкспорт", "Виталий, руководитель"):
        second = store.submit_meeting_metadata(second["id"], value)
    second, second_job = store.accept_meeting_source(
        second["id"], source_name="b.txt", source_path=source, normalized_path=source,
        source_sha256="source-b", normalized_sha256="normalized-same",
    )
    assert second["status"] == "duplicate"
    assert second_job is None


def test_meeting_metadata_duplicate_does_not_advance_next_field(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    meeting = store.begin_meeting(82, 820)
    accepted = store.submit_meeting_metadata(meeting["id"], "2026-09-16", telegram_message_id=100)
    assert accepted["step"] == "title"
    assert store.is_duplicate_meeting_metadata(meeting["id"], "2026-09-16", 100)
    with pytest.raises(StateError, match="уже принято"):
        store.submit_meeting_metadata(meeting["id"], "2026-09-16", telegram_message_id=101)
    unchanged = store.get_meeting(meeting["id"])
    assert unchanged["meeting_date"] == "2026-09-16"
    assert unchanged["title"] is None
    assert unchanged["step"] == "title"


def test_meeting_waits_for_received_feedback_to_finish(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    service = VitalyService(store, load_default_profile(), _config(tmp_path), runner=LearningRunner())
    case = service.new_case(83)
    store.set_review(case["id"], {"readiness_score": 1})
    feedback = store.start_feedback(case["id"], 83, 830, "audit")
    store.submit_feedback(feedback["id"], "Есть вопрос к трактовке показателя.")
    with pytest.raises(StateError, match="рассматривается"):
        store.begin_meeting(83, 830)


def test_terminal_intake_jobs_replay_persisted_result_without_second_model_call(tmp_path: Path):
    class FailIfCalled:
        def __init__(self) -> None:
            self.calls = 0

        async def run(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("a persisted terminal intake must not call the model again")

    config = _config(tmp_path)
    store = VitalyStore(config.state_dir, config.knowledge_dir)
    bootstrap_runner = FailIfCalled()
    LearningManager(store, load_default_profile(), config, bootstrap_runner)
    case = store.create_case(84, "v1", store.active_knowledge_package()["id"])
    store.set_review(case["id"], {"critical_gaps": [], "contradictions": [], "questions": []})
    feedback = store.start_feedback(case["id"], 84, 840, "audit")
    feedback_job = store.submit_feedback(feedback["id"], "Пояснение уже проверено.")
    store.mark_learning_job_running(feedback_job["id"])
    feedback_result = {
        "result": "example", "reason": "Основание найдено.", "source_document_id": "audit",
        "fragment": "пункт G1", "expected_improvement": "Проверять определение.",
    }
    store.set_feedback_result(feedback["id"], "example", feedback_result)

    restarted = VitalyStore(config.state_dir, config.knowledge_dir)
    restarted.recover_jobs()
    runner = FailIfCalled()
    manager = LearningManager(restarted, load_default_profile(), config, runner)
    feedback_outcome = asyncio.run(manager.execute(restarted.get_learning_job(feedback_job["id"])))
    assert feedback_outcome.kind == "feedback_analyzed"
    assert feedback_outcome.payload["assessment"] == feedback_result

    meeting = restarted.begin_meeting(85, 850)
    for value in ("2026-09-16", "Планёрка", "Виталий, руководитель"):
        meeting = restarted.submit_meeting_metadata(meeting["id"], value)
    transcript = tmp_path / "replay-meeting.txt"
    transcript.write_text("Виталий: где подтверждение?", encoding="utf-8")
    _meeting, meeting_job = restarted.accept_meeting_source(
        meeting["id"], source_name="replay-meeting.txt", source_path=transcript, normalized_path=transcript,
        source_sha256="replay-source", normalized_sha256="replay-normalized",
    )
    assert meeting_job is not None
    restarted.mark_learning_job_running(meeting_job["id"])
    classification = {
        "status": "needs_clarification", "reason": "Нужны метки говорящих.", "identity_basis": "ambiguous",
        "identity_confident": False, "duplicate_of": "", "continuation_of": "", "source_fragments": [], "observed_topics": [],
    }
    restarted.set_meeting_result(meeting["id"], classification)

    recovered_again = VitalyStore(config.state_dir, config.knowledge_dir)
    recovered_again.recover_jobs()
    meeting_outcome = asyncio.run(manager.execute(recovered_again.get_learning_job(meeting_job["id"])))
    assert meeting_outcome.kind == "meeting_analyzed"
    assert meeting_outcome.payload["classification"] == classification
    assert runner.calls == 0


def test_enabled_package_is_pinned_only_for_new_cases(tmp_path: Path):
    config = _config(tmp_path)
    store = VitalyStore(config.state_dir, config.knowledge_dir)
    runner = LearningRunner()
    service = VitalyService(store, load_default_profile(), config, runner=runner)
    old_case = service.new_case(9)
    old_package = old_case["knowledge_package_id"]

    current = store.active_knowledge_package()
    replacement_payload = baseline_package_payload(load_default_profile(), package_version="knowledge.test-rollforward")
    replacement = store.create_knowledge_package(
        version="knowledge.test-rollforward",
        package=replacement_payload,
        parent_id=current["id"],
        summary="Тестовый пакет.",
        verification={"passing": True},
    )
    store.activate_knowledge_package(replacement["id"])
    fresh_case = service.new_case(10)

    assert store.get_case(old_case["id"])["knowledge_package_id"] == old_package
    assert fresh_case["knowledge_package_id"] == replacement["id"]
    assert service.profile_for_case(old_case["id"]).version != service.profile_for_case(fresh_case["id"]).version


def test_candidate_runs_two_model_steps_and_activates_only_after_gate(tmp_path: Path):
    config = _config(tmp_path)
    store = VitalyStore(config.state_dir, config.knowledge_dir)
    runner = LearningRunner()
    manager = LearningManager(store, load_default_profile(), config, runner)
    case = store.create_case(11, "v1", store.active_knowledge_package()["id"])
    store.set_review(case["id"], {"critical_gaps": [], "contradictions": [], "questions": []})
    feedback = store.start_feedback(case["id"], 11, 110, "audit")
    feedback_job = store.submit_feedback(feedback["id"], "Здесь разные определения показателя.")
    runner.feedback_id = feedback["id"]

    async def scenario() -> None:
        await manager.execute(feedback_job)
        cycle = store.enqueue_learning_job(kind="knowledge_generate")
        generated = await manager.execute(cycle)
        assert generated.kind == "knowledge_candidate"
        verify = store.get_learning_job(generated.payload["verification_job"]["id"])
        foreground = store.begin_job(case["id"], 11, "review")
        next_work = store.next_queued_work()
        assert next_work is not None and next_work["id"] == foreground["id"]
        store.finish_job(foreground["id"])
        store.settle_job(case["id"])
        outcome = await manager.execute(verify)
        assert outcome.kind == "knowledge_enabled"

    asyncio.run(scenario())
    package = store.active_knowledge_package()
    assert package["version"].startswith("knowledge.")
    assert len(runner.calls) == 3  # feedback assessment, generator, independent verifier
    with store._connect() as connection:
        kinds = [row["kind"] for row in connection.execute("SELECT kind FROM learning_model_runs ORDER BY started_at")]
    assert kinds == ["feedback_analyze", "knowledge_generate", "knowledge_verify"]
    own = store.own_learning_status(110, 11)["feedback"][0]
    assert own["result"]["learning_outcome"] == "enabled"
    with store._connect() as connection:
        regression = connection.execute("SELECT result_json FROM learning_regression_runs").fetchone()
    assert regression is not None
    assert json.loads(regression["result_json"])["fixture_sha256"] == regression_fixture_digest()


def _narrow_patch(rule_id: str, source_id: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "summary": "Тестовое узкое правило.",
        "analysis_rules": [
            {
                "id": rule_id,
                "transfer_key": "compare_definition_before_conflict",
                "scope": "narrow",
                "trigger": "Сравниваются показатели с разными определениями.",
                "guidance": "Сначала сопоставь определение, период и периметр.",
                "expected_improvement": "Не помечать разные определения как противоречие.",
                "source_ids": [source_id],
                "evidence_references": [
                    {
                        "source_id": source_id,
                        "source_document_id": "private-artifact-" + source_id,
                        "reference": "слайд 2",
                        "fragment": "Частный фрагмент исходного документа.",
                    }
                ],
            }
        ],
        "question_patterns": [],
        "meetings": [],
    }


def test_analysis_rules_accumulate_but_provenance_never_reaches_new_case_prompt(tmp_path: Path):
    config = _config(tmp_path)
    store = VitalyStore(config.state_dir, config.knowledge_dir)
    service = VitalyService(store, load_default_profile(), config, runner=LearningRunner())
    baseline = store.active_knowledge_package()
    first_payload = immutable_package_payload(
        baseline["package"]["profile"], _narrow_patch("rule_one", "feedback-one"),
        package_version="knowledge.rules-one", accepted_source_ids=["feedback-one"], accepted_meeting_ids=[],
    )
    first = store.create_knowledge_package(
        version="knowledge.rules-one", package=first_payload, parent_id=baseline["id"], summary="one", verification={"passing": True},
    )
    store.activate_knowledge_package(first["id"])
    second_payload = immutable_package_payload(
        first_payload["profile"], _narrow_patch("rule_two", "feedback-two"),
        package_version="knowledge.rules-two", accepted_source_ids=["feedback-two"], accepted_meeting_ids=[],
        prior_analysis_rules=first_payload["analysis_rules"],
    )
    second = store.create_knowledge_package(
        version="knowledge.rules-two", package=second_payload, parent_id=first["id"], summary="two", verification={"passing": True},
    )
    store.activate_knowledge_package(second["id"])
    case = service.new_case(14)
    rules = service.learning.analysis_rules_for_case(case["id"])
    assert [rule["id"] for rule in rules] == ["compare_definition_before_conflict"]
    assert all("source_ids" not in rule and "evidence_references" not in rule for rule in rules)
    assert "private-artifact-feedback-one" in str(second_payload["analysis_rules"])
    assert "private-artifact-feedback-one" not in service._base_prompt(case["id"])


def test_candidate_provenance_cannot_cross_feedback_and_meeting_lanes(tmp_path: Path):
    sources = [
        {"id": "feedback-1", "source_kind": "feedback"},
        {"id": "meeting-1", "source_kind": "meeting"},
    ]
    invalid_analysis = _narrow_patch("wrong_lane", "meeting-1")
    with pytest.raises(ModelError) as analysis_error:
        LearningManager._validate_candidate_provenance(invalid_analysis, sources, "analysis")
    assert "обратную связь" in analysis_error.value.detail
    invalid_behaviour = {
        "schema_version": 1,
        "summary": "wrong lane",
        "analysis_rules": [],
        "question_patterns": [
            {
                "operation": "add", "topic_id": "source_lane", "name": "source lane", "behaviour": "check",
                "style_mode": "direct",
                "followup_chain": ["why"], "evidence_meetings": ["feedback-1"],
                "evidence_references": [{"meeting_id": "feedback-1", "source": "x", "time_range": "00:01"}],
            }
        ],
        "meetings": [],
    }
    with pytest.raises(ModelError) as behaviour_error:
        LearningManager._validate_candidate_provenance(invalid_behaviour, sources, "behaviour")
    assert "выбранные встречи" in behaviour_error.value.detail


def test_candidate_references_must_bind_to_real_feedback_and_meeting_archive(tmp_path: Path):
    config = _config(tmp_path)
    store = VitalyStore(config.state_dir, config.knowledge_dir)
    manager = LearningManager(store, load_default_profile(), config, LearningRunner())
    source = {"id": "feedback-1", "source_kind": "feedback", "result": {"source_document_id": "audit"}}
    patch = _narrow_patch("real_reference", "feedback-1")
    with pytest.raises(ModelError) as reference_error:
        manager._validate_candidate_source_references(patch, [source])
    assert "не совпадает с исходным документом" in reference_error.value.detail
    patch["analysis_rules"][0]["evidence_references"][0]["source_document_id"] = "audit"
    manager._validate_candidate_source_references(patch, [source])


def test_learning_batch_is_bounded_and_empty_clarification_cannot_consume_everything(tmp_path: Path):
    config = _config(tmp_path)
    store = VitalyStore(config.state_dir, config.knowledge_dir)
    manager = LearningManager(store, load_default_profile(), config, LearningRunner())
    sources = [
        {"id": "feedback-{}".format(index), "source_kind": "feedback", "result": {}, "text": "x" * 20_000}
        for index in range(20)
    ]
    selected, context = manager._bounded_learning_batch(sources)
    assert len(selected) <= 8
    assert len(json.dumps(context, ensure_ascii=False)) <= 100_000
    with pytest.raises(ModelError) as empty_error:
        manager._source_ids({"source_ids": []}, selected, require=True)
    assert "нет оснований" in empty_error.value.detail


def test_generator_clarification_is_persisted_without_changing_active_package(tmp_path: Path):
    class ClarifyingRunner(LearningRunner):
        async def run(self, prompt, schema, image_paths=None):
            del prompt, image_paths
            if "decision" in schema["properties"]:
                return ModelResult(
                    value={
                        "decision": "needs_clarification", "reason": "Источники расходятся по области применения.",
                        "source_ids": [self.feedback_id], "scope": "analysis", "expected_improvement": "Нужно уточнить границу.",
                        "patch": {"schema_version": 1, "summary": "", "analysis_rules": [], "question_patterns": [], "meetings": []},
                    }, stdout="", stderr="",
                )
            raise AssertionError("unexpected model step")

    config = _config(tmp_path)
    store = VitalyStore(config.state_dir, config.knowledge_dir)
    runner = ClarifyingRunner()
    manager = LearningManager(store, load_default_profile(), config, runner)
    case = store.create_case(15, "v1", store.active_knowledge_package()["id"])
    store.set_review(case["id"], {"critical_gaps": [], "contradictions": [], "questions": []})
    feedback = store.start_feedback(case["id"], 15, 150, "audit")
    store.submit_feedback(feedback["id"], "Нужно понять границы правила.")
    store.set_feedback_result(feedback["id"], "example", {"reason": "fixture"})
    runner.feedback_id = feedback["id"]
    before = store.active_knowledge_package()["id"]
    outcome = asyncio.run(manager.execute(store.enqueue_learning_job(kind="knowledge_generate")))
    assert outcome.kind == "knowledge_needs_clarification"
    assert store.active_knowledge_package()["id"] == before
    assert outcome.payload["candidate"]["status"] == "needs_clarification"
    assert store.own_learning_status(150, 15)["feedback"][0]["result"]["learning_outcome"] == "needs_clarification"


def test_ambiguous_or_inferred_vitaly_is_never_accepted(tmp_path: Path):
    class InferredRunner(LearningRunner):
        async def run(self, prompt, schema, image_paths=None):
            del prompt, image_paths
            assert schema == MEETING_CLASSIFICATION_SCHEMA
            return ModelResult(
                value={
                    "status": "accepted", "reason": "Секретная деталь чужой встречи.", "identity_basis": "inferred",
                    "identity_confident": True, "duplicate_of": "", "continuation_of": "",
                    "source_fragments": ["00:01"], "observed_topics": ["evidence_scope"],
                }, stdout="", stderr="",
            )

    config = _config(tmp_path)
    store = VitalyStore(config.state_dir, config.knowledge_dir)
    manager = LearningManager(store, load_default_profile(), config, InferredRunner())
    meeting = store.begin_meeting(16, 160)
    for value in ("2026-09-16", "Без меток", "Виталий"):
        meeting = store.submit_meeting_metadata(meeting["id"], value)
    transcript = tmp_path / "ambiguous.txt"
    transcript.write_text("Кто-то сказал: где цифры?", encoding="utf-8")
    _meeting, job = store.accept_meeting_source(
        meeting["id"], source_name="ambiguous.txt", source_path=transcript, normalized_path=transcript,
        source_sha256="ambiguous-source", normalized_sha256="ambiguous-normalized",
    )
    # The fake runner deliberately bypasses the JSON-schema gateway to prove
    # the runtime independently rejects an inferred identity as well.
    outcome = asyncio.run(manager.execute(job))
    assert outcome.payload["meeting"]["status"] == "needs_clarification"
    assert "Секретная" not in outcome.payload["classification"]["reason"]
    assert "Секретная" in outcome.payload["classification"]["internal_reason"]


def test_general_feedback_rechecks_rendered_pages(tmp_path: Path):
    class VisualRunner(LearningRunner):
        def __init__(self) -> None:
            super().__init__()
            self.images: list[Path] = []

        async def run(self, prompt, schema, image_paths=None):
            del prompt
            assert "result" in schema["properties"]
            self.images = list(image_paths or [])
            return ModelResult(
                value={
                    "result": "example", "reason": "Визуальная проверка подтверждает замечание.",
                    "source_document_id": "audit", "fragment": "слайд 2",
                    "expected_improvement": "Учитывать подпись диаграммы.",
                }, stdout="", stderr="",
            )

    config = _config(tmp_path)
    store = VitalyStore(config.state_dir, config.knowledge_dir)
    runner = VisualRunner()
    manager = LearningManager(store, load_default_profile(), config, runner)
    case = store.create_case(17, "v1", store.active_knowledge_package()["id"])
    source = tmp_path / "slides.txt"
    source.write_text("Слайд 2: подпись диаграммы.", encoding="utf-8")
    image = tmp_path / "slide-2.png"
    image.write_bytes(b"not-a-real-png")
    store.add_artifact(
        case["id"], "visual", "slides.txt", "slides.txt", "hash", ".txt", source.stat().st_size,
        source, source, [image], 1, [],
    )
    store.set_review(case["id"], {"critical_gaps": [], "contradictions": [], "questions": []})
    feedback = store.start_feedback(case["id"], 17, 170, "audit")
    job = store.submit_feedback(feedback["id"], "Подпись диаграммы уже объясняет различие.")
    asyncio.run(manager.execute(job))
    assert runner.images == [image]


def test_calculated_holdout_coverage_overrides_verifier_scalar_score():
    profile = load_default_profile()
    rows = [{
        "scenario_id": "style_topic_coverage", "passed": True, "target_score": 1.0,
        "critical_error_ids": [], "style_coverage": 1.0, "first_ten_topics": ["unrelated"],
    }]
    calculated = LearningManager._with_calculated_coverage(rows, profile)
    assert calculated[0]["style_coverage"] == 0.0
    assert calculated[0]["coverage_detail"]["passing"] is False


def test_pause_stops_queued_automatic_steps_and_restart_recovers_running_learning(tmp_path: Path):
    store = VitalyStore(tmp_path / "state")
    queued = store.enqueue_learning_job(kind="knowledge_generate")
    store.set_learning_paused(True, actor_user_id=1)
    assert store.get_learning_job(queued["id"])["status"] == "paused"
    assert store.next_queued_work() is None
    store.set_learning_paused(False, actor_user_id=1)
    assert store.get_learning_job(queued["id"])["status"] == "queued"
    store.mark_learning_job_running(queued["id"])
    restarted = VitalyStore(store.state_dir)
    restarted.recover_jobs()
    assert restarted.get_learning_job(queued["id"])["status"] == "queued"

    restarted.mark_learning_job_running(queued["id"])
    restarted.set_learning_paused(True, actor_user_id=1)
    recovered_while_paused = VitalyStore(store.state_dir)
    recovered_while_paused.recover_jobs()
    assert recovered_while_paused.get_learning_job(queued["id"])["status"] == "paused"

    # If the operator pauses while a generator is still running, its completed
    # step may enqueue a verifier afterwards.  That verifier must inherit the
    # pause rather than silently becoming the next runnable model call.
    race_store = VitalyStore(tmp_path / "pause-race")
    generator = race_store.enqueue_learning_job(kind="knowledge_generate")
    race_store.mark_learning_job_running(generator["id"])
    race_store.set_learning_paused(True, actor_user_id=1)
    verifier = race_store.enqueue_learning_job(kind="knowledge_verify", payload={"candidate_id": "candidate"})
    assert race_store.get_learning_job(verifier["id"])["status"] == "paused"
    assert race_store.next_queued_work() is None


def test_failed_verification_marks_candidate_terminal_and_keeps_active_package(tmp_path: Path):
    config = _config(tmp_path)
    store = VitalyStore(config.state_dir, config.knowledge_dir)
    manager = LearningManager(store, load_default_profile(), config, LearningRunner())
    case = store.create_case(18, "v1", store.active_knowledge_package()["id"])
    store.set_review(case["id"], {"critical_gaps": [], "contradictions": [], "questions": []})
    feedback = store.start_feedback(case["id"], 18, 180, "audit")
    store.submit_feedback(feedback["id"], "Нужен воспроизводимый пример.")
    store.set_feedback_result(feedback["id"], "example", {"reason": "fixture"})
    candidate = store.create_learning_candidate(
        [feedback["id"]], {"schema_version": 1, "summary": "x", "analysis_rules": [], "question_patterns": [], "meetings": []},
        "analysis", "Проверить сбой", baseline_package_id=store.active_knowledge_package()["id"],
    )
    verify_job = store.enqueue_learning_job(kind="knowledge_verify", payload={"candidate_id": candidate["id"]})
    active_before = store.active_knowledge_package()["id"]

    store.mark_learning_submission_failed(verify_job["id"], "subscription_exhausted")

    failed = store.get_learning_candidate(candidate["id"])
    assert failed["status"] == "failed"
    assert failed["error"] == "subscription_exhausted"
    assert store.active_knowledge_package()["id"] == active_before
    assert store.get_feedback(feedback["id"])["result"]["learning_outcome"] == "verification_failed"


def test_atomic_publish_updates_package_candidate_and_source_together(tmp_path: Path):
    config = _config(tmp_path)
    store = VitalyStore(config.state_dir, config.knowledge_dir)
    LearningManager(store, load_default_profile(), config, LearningRunner())
    case = store.create_case(19, "v1", store.active_knowledge_package()["id"])
    store.set_review(case["id"], {"critical_gaps": [], "contradictions": [], "questions": []})
    feedback = store.start_feedback(case["id"], 19, 190, "audit")
    store.submit_feedback(feedback["id"], "Проверочный пример.")
    store.set_feedback_result(feedback["id"], "example", {"source_document_id": "audit"})
    active = store.active_knowledge_package()
    candidate = store.create_learning_candidate(
        [feedback["id"]], {"schema_version": 1, "summary": "x", "analysis_rules": [], "question_patterns": [], "meetings": []},
        "analysis", "Проверить атомарность", baseline_package_id=active["id"],
    )
    payload = baseline_package_payload(load_default_profile(), package_version="knowledge.atomic")
    package = store.publish_knowledge_package(
        version="knowledge.atomic", package=payload, parent_id=active["id"], summary="секретные детали",
        verification={"passing": True}, candidate_id=candidate["id"], source_ids=[feedback["id"]],
    )
    assert store.active_knowledge_package()["id"] == package["id"]
    assert store.get_learning_candidate(candidate["id"])["status"] == "enabled"
    assert store.get_feedback(feedback["id"])["result"]["learning_outcome"] == "enabled"
    assert package["summary"] == public_release_summary(payload["patch"])


def test_learning_scheduler_is_paused_or_delayed_by_user_work(tmp_path: Path):
    config = _config(tmp_path)
    store = VitalyStore(config.state_dir, config.knowledge_dir)
    service = VitalyService(store, load_default_profile(), config, runner=LearningRunner())
    case = service.new_case(12)
    store.set_review(case["id"], {"critical_gaps": [], "contradictions": [], "questions": []})
    feedback = store.start_feedback(case["id"], 12, 120, "audit")
    feedback_job = store.submit_feedback(feedback["id"], "Нужен проверочный пример.")
    store.set_feedback_result(feedback["id"], "example", {"reason": "fixture"})

    # A queued foreground review always wins and prevents an auto-cycle from
    # being admitted until the user queue is empty.
    store.begin_job(case["id"], 12, "review")
    assert store.due_learning_cycle(1) is False
    store.set_learning_paused(True, actor_user_id=120)
    assert store.due_learning_cycle(1) is False
    store.set_learning_paused(False, actor_user_id=120)
    foreground = store.next_queued_job()
    assert foreground is not None
    store.finish_job(foreground["id"])
    store.settle_job(case["id"])
    # A test-only direct result must also model the worker's terminal job
    # transition; otherwise the scheduler correctly sees pending user work.
    store.finish_learning_job(feedback_job["id"])
    assert store.due_learning_cycle(1) is True


def test_archive_cleanup_is_non_destructive_and_disk_guard_fails_closed(tmp_path: Path, monkeypatch):
    import shutil

    store = VitalyStore(tmp_path / "state")
    case = store.create_case(13, "v1")
    source = store.case_dir(case["id"]) / "artifacts" / "kept.txt"
    source.write_text("keep", encoding="utf-8")
    assert store.cleanup_expired_files(1) == []
    assert source.exists()

    usage = shutil.disk_usage(store.knowledge_root)
    monkeypatch.setattr(shutil, "disk_usage", lambda _: usage.__class__(usage.total, usage.used, 0))
    with pytest.raises(Exception, match="Недостаточно свободного места"):
        store.ensure_free_space(1)


def test_four_day_context_reset_keeps_archived_case_files(tmp_path: Path):
    config = _config(tmp_path)
    store = VitalyStore(config.state_dir, config.knowledge_dir)
    service = VitalyService(store, load_default_profile(), config, runner=LearningRunner())
    original = service.new_case(18)
    kept = store.case_dir(original["id"]) / "artifacts" / "kept.txt"
    kept.write_text("архив остаётся", encoding="utf-8")
    old = "2026-01-01T00:00:00+00:00"
    with store._connect() as connection:
        connection.execute("UPDATE cases SET last_user_activity_at = ?, updated_at = ? WHERE id = ?", (old, old, original["id"]))
    current, expired = service.case_for_user(18)
    assert expired == original["id"]
    assert current["id"] != original["id"]
    assert kept.exists()
    assert store.get_case(original["id"])["status"] == "expired"
