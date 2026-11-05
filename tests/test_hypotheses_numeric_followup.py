"""Regression tests for singular repair followups in hard questions."""

import pytest

from tof_reliability.vitaly_bot import hypotheses_pilot as pilot


DEFAULT_SCOPE = "Проверить повторные ремонты насосов"
FOLLOWUP_QUESTIONS = (
    "Как отличить продолжение одного ремонта от нового отказа?",
    "Как понять, является ли это продолжением одного ремонта, а не новым отказом?",
    "Можно ли считать новый симптом продолжением одного ремонта?",
)


def validate_text(text, field):
    card = {field: [text] if field == "hard_questions" else text}
    pilot._validate_numeric_claims(card, DEFAULT_SCOPE)


@pytest.mark.parametrize("field", ["management_question", "hard_questions"])
@pytest.mark.parametrize("question", FOLLOWUP_QUESTIONS)
def test_singular_repair_followups_are_not_numeric_assertions(field, question):
    validate_text(question, field)


def test_multiple_singular_repair_followups_are_allowed_in_hard_questions():
    pilot._validate_numeric_claims(
        {"hard_questions": list(FOLLOWUP_QUESTIONS)}, DEFAULT_SCOPE
    )


@pytest.mark.parametrize("field", ["management_question", "hard_questions"])
@pytest.mark.parametrize("question", [
    "Почему простой насоса составил 15 часов?",
    "Не означает ли стоимость 15 рублей продолжение одного ремонта?",
    "Как отличить продолжение одного ремонта после 15 часов простоя?",
    "Как отличить один ремонт от другого?",
])
def test_unsupported_or_unrelated_numeric_questions_remain_rejected(field, question):
    with pytest.raises(ValueError, match="unsupported numerical value"):
        validate_text(question, field)


@pytest.mark.parametrize("field", ["management_question", "hard_questions"])
@pytest.mark.parametrize("statement", [
    "Выполнен ровно один ремонт.",
    "Сэкономлено 15 рублей.",
])
def test_numeric_statements_remain_rejected(field, statement):
    with pytest.raises(ValueError, match="unsupported numerical value"):
        validate_text(statement, field)


@pytest.mark.parametrize(
    "field", [field for field in pilot.TEXT_FIELDS if field != "management_question"]
)
def test_followup_exemption_does_not_apply_to_other_text_fields(field):
    with pytest.raises(ValueError, match="unsupported numerical value"):
        validate_text(FOLLOWUP_QUESTIONS[0], field)
