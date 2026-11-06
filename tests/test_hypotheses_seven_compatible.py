"""Regression contracts for source-backed breadth and legacy synthetic fixtures.

All texts, source files and coverage records are synthetic; route registration
is checked against the repository registry, not against a live ledger.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

import test_hypotheses_pilot as legacy
from pilot_projection_support import project_cards, projection
from tof_reliability.vitaly_bot import hypotheses_pilot as pilot
from tof_reliability.vitaly_bot import hypotheses_topic_projection as topics
from tof_reliability.vitaly_bot import server_coverage


DIRECTIONS = [
    ('maintenance_history_repeated_episodes', 'AN-REPAIR-EPISODES', 'Проверить историю ремонтов оборудования и повторные ремонтные эпизоды'),
    ('maintenance_plan_fact', 'AN-PLAN-FACT', 'Сравнить плановые и фактические показатели ремонта оборудования'),
    ('mtr_cost_supply', 'AN-MTR-COST', 'Проверить стоимость МТР и поставки для ремонта оборудования'),
    ('equipment_downtime', 'AN-DOWNTIME', 'Проверить простои оборудования при ремонте'),
    ('labor_norm_observed_measurement', 'AN-TECHCARD-LABOR', 'Сопоставить нормы трудозатрат на ремонт и замеры трудозатрат'),
    ('equipment_criticality', 'AN-EQUIPMENT-CRITICALITY', 'Проверить критичность оборудования при ремонте'),
    ('techcard_coverage_verification', 'AN-TECHCARD-LABOR', 'Проверить покрытие и проверку техкарт ремонта'),
]


@pytest.fixture
def case(tmp_path):
    interests, sources = [], []
    for n, (_, _, text) in enumerate(DIRECTIONS):
        path = tmp_path / f'source-{n}.txt'
        path.write_text(text, encoding='utf-8')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        source_id = f'source-{n}'
        sources.append({'id': source_id, 'date': '2030-01-01', 'kind': 'transcript',
                        'sha256': digest, 'path': str(path)})
        interests.append({'id': f'interest-{n}', 'requirement': text, 'evidence': [
            {'source_id': source_id, 'locator': f'chars:0-{len(text)}',
             'attribution': 'interpretation', 'fragment': text}]})
    profile = {'version': 'i1', 'corpus_version': 'c1', 'interests': interests}
    index = {'version': 'c1', 'sources': sources}
    labels = topics.project_topics(profile, index)
    assert {(row['interest_id'], row['topic_id']) for row in labels} == {
        (f'interest-{n}', topic) for n, (topic, _, _) in enumerate(DIRECTIONS)}
    audit = [{'interest_id': row['interest_id'], 'topic_id': row['topic_id'],
              'source_id': interests[n]['evidence'][0]['source_id'],
              'locator': interests[n]['evidence'][0]['locator'],
              'verdict': {'supported': True, 'confidence': 'high'}}
             for n, row in enumerate(labels)]
    context = copy.deepcopy(profile)
    for item, source in zip(context['interests'], sources):
        evidence = item['evidence'][0]
        evidence.update(date=source['date'], sha256=source['sha256'],
                        reference=f"{source['id']}:{evidence['locator']}")
        item['attributions'] = ['interpretation']
    registry = json.loads((Path(__file__).resolve().parents[1] / 'harness/capabilities/analytics.json').read_text())
    registered = {item['id']: item for item in registry['capabilities']}
    # A partial route can support an investigative idea while explicitly
    # identifying absent supply or measured-labour data. It does not establish
    # availability of those facts or readiness to calculate the result.
    routes = []
    for route_id in dict.fromkeys(route for _, route, _ in DIRECTIONS):
        assert route_id in server_coverage.ROUTES
        assert registered[route_id]['lifecycle'] == 'available'
        missing = ['синтетическое ограничение данных']
        if route_id == 'AN-MTR-COST':
            missing.append('поставки и доступность МТР до начала работ не подтверждены')
        if route_id == 'AN-TECHCARD-LABOR':
            missing.append('измеренная фактическая трудоёмкость не подтверждена')
        routes.append({'capability_id': route_id, 'availability': 'partial',
                       'missing_evidence': missing,
                       'source_references': [f'synthetic:{route_id}']})
    cards = []
    for n, (topic, route, text) in enumerate(DIRECTIONS):
        route_missing = next(item['missing_evidence'] for item in routes if item['capability_id'] == route)
        cards.append({'number': n + 1, 'capability_id': route,
                      'management_question': f'Как проверить: {text}?',
                      'decision': f'Выбрать проверку: {text}', 'scope': 'насосы',
                      'metrics_method': f'Сопоставить записи: {text}',
                      'meeting_interest': {'interest_id': f'interest-{n}', 'topic_id': topic},
                      'source_references': [f'synthetic:{route}'], 'source_readiness': 'partial',
                      'missing_evidence': route_missing,
                      'falsification': 'Сопоставление не выявит различий',
                      'hard_questions': ['Как проверить полноту данных?'],
                      'analyst_brief': 'Проверить применимость источников', 'novelty': 'unknown',
                      'ranking': {'utility': 3, 'feasibility': 3, 'interest': 3, 'novelty': 0}})
    return context, labels, audit, {'coverage_version': 1, 'generation': {'run_id': 'g1'}, 'routes': routes}, cards


def generate(case, cards=None):
    context, labels, audit, coverage, defaults = case
    return pilot.generate_pilot(scope='насосы', knowledge_version='k1', interest_context=context,
        verified_labels=labels, verifier_audit=audit,
        coverage_provider=lambda: copy.deepcopy(coverage),
        generator=lambda _: copy.deepcopy(defaults if cards is None else cards),
        private_verifier=lambda _: {'supported': True, 'confidence': 'high'})


def test_seven_source_verified_directions_use_registered_applicable_routes(case):
    run = generate(case)
    assert len(run['cards']) == 7
    assert {c['meeting_interest']['topic_id'] for c in run['cards']} == {d[0] for d in DIRECTIONS}
    assert all(c['meeting_interest']['reference'] and c['missing_evidence'] for c in run['cards'])
    pilot.validate_pilot(run, coverage=case[3], interest_context=case[0])
    with pytest.raises(ValueError, match='(?i)seven|breadth|topic|direction|сем|тем|направлен'):
        generate(case, case[4][:5])


def test_one_verified_topic_cannot_be_inflated_to_five(case):
    context, labels, audit, coverage, cards = copy.deepcopy(case)
    context['interests'] = context['interests'][:1]
    labels, audit = labels[:1], audit[:1]
    cards = [dict(copy.deepcopy(cards[0]), number=n) for n in range(1, 6)]
    from pilot_projection_support import distinct_directions
    distinct_directions(cards)
    with pytest.raises(ValueError, match='(?i)insufficien|fewer|breadth|недостат|тем|направлен'):
        generate((context, labels, audit, coverage, cards))


def test_old_success_fixture_has_distinct_verified_topics_and_real_routes():
    inputs = legacy.inputs.__wrapped__()
    cards = legacy.cards.__wrapped__()
    projected = projection(inputs['interest_context'])
    bindings = {(x['interest_id'], x['topic_id']) for x in projected['verified_labels']}
    audits = {(x['interest_id'], x['topic_id']) for x in projected['verifier_audit']
              if x['verdict']['supported'] is True and x['verdict']['confidence'] == 'high'}
    prepared = project_cards(copy.deepcopy(cards))
    selected = {(c['meeting_interest']['interest_id'], c['meeting_interest']['topic_id']) for c in prepared}
    assert len({topic for _, topic in selected}) >= 5
    assert selected <= bindings <= audits
    routes = {r['capability_id'] for r in inputs['coverage']['routes'] if r['availability'] != 'unavailable'}
    assert routes <= set(server_coverage.ROUTES)
    assert {c['capability_id'] for c in prepared} <= routes
    run = legacy.generate(pilot, inputs, cards)
    assert len(run['cards']) >= 5
    pilot.validate_pilot(run, coverage=inputs['coverage'], interest_context=inputs['interest_context'])


def test_legacy_synthetic_labels_are_backed_by_source_file():
    inputs = legacy.inputs.__wrapped__()
    context = inputs['interest_context']
    path = Path(__file__).parent / 'fixtures/synthetic_repair_topics.txt'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source_ids = {item['evidence'][0]['source_id'] for item in context['interests']}
    index = {'version': context['corpus_version'], 'sources': [
        {'id': source_id, 'date': '2026-10-01', 'kind': 'transcript',
         'sha256': digest, 'path': str(path.resolve())} for source_id in sorted(source_ids)]}
    expected = {(x['interest_id'], x['topic_id']) for x in projection(context)['verified_labels']}
    actual = {(x['interest_id'], x['topic_id']) for x in topics.project_topics(context, index)}
    assert len({topic for _, topic in actual}) >= 5
    assert actual == expected



def test_five_of_six_discloses_and_binds_the_eligible_topics(case):
    context, labels, audit, coverage, cards = copy.deepcopy(case)
    context['interests'], labels, audit = context['interests'][:6], labels[:6], audit[:6]
    run = generate((context, labels, audit, coverage, cards), cards[:5])
    assert len(run['cards']) == 5
    assert len(run['breadth_justification']['eligible_topic_ids']) == 6
    run['breadth_justification']['eligible_topic_ids'] = []
    with pytest.raises(ValueError, match='(?i)breadth|justification|eligible'):
        pilot.validate_pilot(run, coverage=coverage, interest_context=context)


def test_incompatible_or_unavailable_route_is_rejected(case):
    context, labels, audit, coverage, cards = copy.deepcopy(case)
    cards[2]['capability_id'] = 'AN-DOWNTIME'
    cards[2]['source_references'] = next(r['source_references'] for r in coverage['routes'] if r['capability_id'] == 'AN-DOWNTIME')
    with pytest.raises(ValueError, match='(?i)route|compatib|маршрут|тем'):
        generate((context, labels, audit, coverage, cards))
    context, labels, audit, coverage, cards = copy.deepcopy(case)
    next(r for r in coverage['routes'] if r['capability_id'] == 'AN-DOWNTIME')['availability'] = 'unavailable'
    with pytest.raises(ValueError, match='(?i)unavailable|route|маршрут|недоступ'):
        generate((context, labels, audit, coverage, cards))


def test_insufficient_topics_stop_before_generator(case):
    context, labels, audit, coverage, cards = copy.deepcopy(case)
    calls = []
    with pytest.raises(ValueError, match='(?i)insufficien|fewer|breadth|недостат|тем'):
        pilot.generate_pilot(scope='насосы', knowledge_version='k1',
            interest_context=context, verified_labels=labels[:1], verifier_audit=audit[:1],
            coverage_provider=lambda: coverage, generator=lambda _: calls.append(True) or cards,
            private_verifier=lambda _: {'supported': True, 'confidence': 'high'})
    assert not calls


def test_route_applicability_is_grounded_in_registry_contract():
    registry = json.loads((Path(__file__).resolve().parents[1] / 'harness/capabilities/analytics.json').read_text())
    routes = {entry['id']: entry for entry in registry['capabilities']}
    for route, terms in {
        'AN-REPAIR-EPISODES': ('эпизод',),
        'AN-PLAN-FACT': ('план', 'фактич'),
        'AN-MTR-COST': ('списан', 'материал'),
        'AN-DOWNTIME': ('просто',),
        'AN-TECHCARD-LABOR': ('операци', 'техкарт', 'фактическ'),
        'AN-EQUIPMENT-CRITICALITY': ('критичност',),
    }.items():
        entry = routes[route]
        contract = ' '.join([entry['behavior'], *entry['verification'], *entry['forbidden']]).casefold()
        assert entry['lifecycle'] == 'available' and entry['task_types'] == ['analysis']
        assert all(term in contract for term in terms)
        assert route in server_coverage.ROUTES
        assert server_coverage.ROUTES[route]  # semantic limitations are explicit
