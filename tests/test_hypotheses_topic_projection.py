"""Contract for an isolated, provenance-checked topic projection.

All source texts and names in this file are synthetic.  Holdout selection belongs
at the caller boundary; this module receives the caller-selected source index.
"""
import copy
import hashlib
import importlib
import json

import pytest


@pytest.fixture
def projection():
    name = "tof_reliability.vitaly_bot.hypotheses_topic_projection"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        class MissingProjection:
            def __getattr__(self, attribute):
                pytest.fail("topic projection module is not implemented")

        return MissingProjection()


@pytest.fixture
def make_case(tmp_path):
    def make(requirement, fragment=None, *, interest_id="interest-1"):
        text = fragment if fragment is not None else requirement
        path = tmp_path / "synthetic-source.txt"
        path.write_text(text, encoding="utf-8")
        index = {"version": "synthetic-corpus", "sources": [{
            "id": "source-1", "date": "2030-01-01", "kind": "transcript",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "path": str(path),
        }]}
        profile = {"version": "synthetic-profile", "corpus_version": "synthetic-corpus",
                   "interests": [{"id": interest_id, "requirement": requirement,
                                  "evidence": [{"source_id": "source-1",
                                                "locator": f"chars:0-{len(text)}",
                                                "attribution": "direct_request",
                                                "fragment": text}]}]}
        return profile, index
    return make


def project(projection, case):
    profile, index = case
    return projection.project_topics(profile, index)


def test_repeated_repairs_are_projected_to_a_generic_topic(projection, make_case):
    case = make_case("Проанализировать историю ремонтов оборудования и повторные ремонтные эпизоды")
    result = project(projection, case)
    assert len(result) == 1
    assert result[0]["topic_id"] == "maintenance_history_repeated_episodes"
    assert result[0]["interest_id"] == "interest-1"
    assert result[0]["description"]
    assert set(result[0]) == {"topic_id", "interest_id", "description"}
    assert result == project(projection, case)


def test_excavation_and_single_domain_words_do_not_imply_repair_analytics(projection, make_case):
    for text in ("Раскопки древнего поселения и история находок",
                 "План раскопок и график экскурсий в музее",
                 "Ремонт", "Критичность", "План и факт"):
        assert project(projection, make_case(text)) == []


def test_source_evidence_must_support_the_topic_not_just_exist(projection, make_case):
    case = make_case("Проанализировать повторные ремонты оборудования",
                     fragment="Обсудить раскопки и историю древнего поселения")
    assert project(projection, case) == []


@pytest.mark.parametrize("damage", ["missing_evidence", "unknown_source", "wrong_fragment",
                                    "wrong_hash", "wrong_locator", "wrong_version"])
def test_invalid_or_absent_provenance_fails_closed(projection, make_case, damage):
    profile, index = make_case("Проанализировать повторные ремонты оборудования")
    if damage == "missing_evidence":
        profile["interests"][0]["evidence"] = []
    elif damage == "unknown_source":
        profile["interests"][0]["evidence"][0]["source_id"] = "unknown"
    elif damage == "wrong_fragment":
        profile["interests"][0]["evidence"][0]["fragment"] = "несуществующий фрагмент"
    elif damage == "wrong_hash":
        index["sources"][0]["sha256"] = "0" * 64
    elif damage == "wrong_locator":
        profile["interests"][0]["evidence"][0]["locator"] = "chars:0-99999"
    else:
        profile["corpus_version"] = "other"
    try:
        result = projection.project_topics(profile, index)
    except ValueError:
        return
    assert result == []


def test_projection_redacts_names_figures_requirements_and_evidence(projection, make_case):
    text = ("Проанализировать повторные ремонты оборудования: Александр Пушкин, "
            "987654321 рублей, СЕКРЕТНАЯ_ФРАЗА")
    case = make_case(text, interest_id="synthetic-1")
    result = project(projection, case)
    assert result and all(row["interest_id"] == "synthetic-1" for row in result)
    serialized = json.dumps(result, ensure_ascii=False)
    for forbidden in (text, "Александр", "Пушкин", "987654321", "СЕКРЕТНАЯ_ФРАЗА",
                      "source-1", "chars:", "fragment", "sha256", "synthetic-source.txt"):
        assert forbidden not in serialized
    assert all(set(row) == {"topic_id", "interest_id", "description"} for row in result)
    clean = project(projection, make_case("Проанализировать повторные ремонты оборудования", interest_id="synthetic-1"))
    assert result == clean


def test_evidence_backed_interests_keep_their_own_source_interest_ids(projection, make_case):
    first, index = make_case("Сравнить плановые и фактические сроки ремонтов оборудования", interest_id="plan-1")
    text = "Проверить плановые и фактические сроки ремонтов оборудования"
    second = copy.deepcopy(first["interests"][0])
    second["id"] = "plan-2"
    second["requirement"] = text
    # Both source excerpts are independently verified against the same synthetic text.
    source_text = first["interests"][0]["evidence"][0]["fragment"]
    second["requirement"] = source_text
    first["interests"].append(second)
    rows = projection.project_topics(first, index)
    assert {row["interest_id"] for row in rows} == {"plan-1", "plan-2"}
    assert {row["topic_id"] for row in rows} == {"maintenance_plan_fact"}
    assert len({row["description"] for row in rows}) == 1


def test_one_interest_can_map_to_multiple_supported_topics(projection, make_case):
    text = ("Сравнить плановые и фактические сроки ремонтов оборудования "
            "и проанализировать повторные ремонтные эпизоды оборудования")
    result = project(projection, make_case(text))
    assert {row["topic_id"] for row in result} == {
        "maintenance_plan_fact", "maintenance_history_repeated_episodes"}
    assert {row["interest_id"] for row in result} == {"interest-1"}


def test_empty_and_uncertain_interests_are_not_guessed(projection, make_case):
    profile, index = make_case("Что-нибудь полезное про оборудование")
    assert projection.project_topics(profile, index) == []
    profile["interests"] = []
    assert projection.project_topics(profile, index) == []


@pytest.mark.parametrize(("text", "topic"), [
    ("Проанализировать график ремонтов оборудования и загрузку бригад", "maintenance_scheduling_resource_load"),
    ("Проанализировать стоимость и поставки МТР для ремонта оборудования", "mtr_cost_supply"),
    ("Проанализировать простои оборудования из-за ремонта", "equipment_downtime"),
    ("Сравнить норму трудозатрат на ремонт с замерами трудозатрат", "labor_norm_observed_measurement"),
    ("Оценить критичность оборудования для ремонта", "equipment_criticality"),
    ("Проверить покрытие и верификацию техкарт ремонта", "techcard_coverage_verification"),
    ("Проверить качество данных о ремонтах оборудования", "maintenance_data_quality"),
    ("Оценить измеримость внедрения изменений в ремонте оборудования", "implementation_measurability"),
])
def test_curated_topics_require_matching_verified_evidence(projection, make_case, text, topic):
    rows = project(projection, make_case(text))
    assert topic in {row["topic_id"] for row in rows}
    assert project(projection, make_case(text, fragment="Неизвестная тема")) == []


def test_negated_or_uncertain_repair_topics_are_omitted(projection, make_case):
    for text in ("Не анализировать повторные ремонты оборудования",
                 "Возможно повторные ремонты оборудования"):
        assert project(projection, make_case(text)) == []
