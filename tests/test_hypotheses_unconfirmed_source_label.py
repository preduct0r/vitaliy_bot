from tof_reliability.vitaly_bot.hypotheses_ui import render


def empty_card(**updates):
    card = {
        'number': 1,
        'management_question': '',
        'decision': '',
        'scope': '',
        'metrics_method': '',
        'meeting_interest': {},
        'source_references': [],
        'case_material_references': [],
        'source_readiness': 'unavailable',
        'missing_evidence': [],
        'falsification': '',
        'hard_questions': [],
        'analyst_brief': '',
    }
    card.update(updates)
    return card


def test_unconfirmed_empty_references_are_not_presented_as_missing_data():
    lines = render(empty_card()).splitlines()

    assert 'Доступные источники: Источники не подтверждены' in lines
    assert 'Готовность источников: не подтверждена' in lines
    assert 'Периметр: Нет данных' in lines
    assert 'Пробелы: Нет данных' in lines
    assert 'Опровержение: Нет данных' in lines


def test_present_source_and_case_material_references_are_preserved():
    lines = render(empty_card(
        source_references=['analytic-route:weekly'],
        case_material_references=['case:repair-123'],
    )).splitlines()

    assert 'Доступные источники: analytic-route:weekly; case:repair-123' in lines
