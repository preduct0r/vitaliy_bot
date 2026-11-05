"""Regression tests for singular repair references in interrogative text."""

import pytest

from tof_reliability.vitaly_bot import hypotheses_pilot as pilot


DEFAULT_SCOPE = "Проверить повторные ремонты насосов"


def validate_question(text, *, scope=DEFAULT_SCOPE):
    pilot._validate_numeric_claims({"management_question": text}, scope)


@pytest.mark.parametrize("question", [
    "Как отличить продолжение одного ремонта от нового отказа?",
    "Как понять, является ли это продолжением одного ремонта, а не новым отказом?",
    "Можно ли считать новый симптом продолжением одного ремонта?",
    "Что считать продолжением одного ремонта при повторном отказе?",
])
def test_interrogative_singular_repair_reference_is_not_a_numeric_assertion(question):
    validate_question(question)


@pytest.mark.parametrize("question", [
    "Почему простой насоса составил 15 часов?",
    "Подтверждают ли затраты в 15 рублей продолжение одного ремонта?",
    "Как отличить продолжение одного ремонта после 15 часов простоя?",
    "Можно ли отнести случай к одному из трех ремонтов?",
    "Это второй отказ после предыдущего ремонта?",
])
def test_interrogative_context_does_not_allow_unsupported_numeric_claims(question):
    with pytest.raises(ValueError, match="(?i)numerical|numeric|числ"):
        validate_question(question)


@pytest.mark.parametrize("statement", [
    "Выполнен ровно один ремонт.",
    "Это продолжение одного ремонта.",
])
def test_statement_about_one_existing_repair_remains_rejected(statement):
    with pytest.raises(ValueError, match="(?i)numerical|numeric|числ"):
        validate_question(statement)


def test_exact_case_scoped_numeric_statement_remains_allowed():
    fact = "Проверены 2 насоса."
    validate_question(fact, scope=fact)


def test_numeric_statement_with_changed_case_context_remains_rejected():
    with pytest.raises(ValueError, match="(?i)numerical|numeric|числ"):
        validate_question("Проверены 2 двигателя.", scope="Проверены 2 насоса.")
