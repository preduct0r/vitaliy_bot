"""Empty user requests remain trusted while candidate scopes stay explicit.

All interests, coverage and candidate cards come from synthetic fixtures.
"""
import copy
import json
from dataclasses import replace

import pytest

from test_hypotheses_dimensions_pilot import dimension_case, topic_case
from tof_reliability.vitaly_bot import hypotheses_pilot as pilot


@pytest.fixture
def setup_case(dimension_case, tmp_path):
    context, coverage, projection, cards = copy.deepcopy(dimension_case)
    for card in cards:
        card['scope'] = ''
    store = pilot.TrustedRequestStore(tmp_path / 'trusted.sqlite')
    trusted = store.create({'run_id': 'run', 'case_id': 'case', 'user_id': 'user', 'scope': ''})
    return context, coverage, projection, cards, store, trusted


def generate(data, *, cards=None, scope='', verifier=None, provider=None, trusted=None):
    context, coverage, projection, defaults, store, original = data
    calls, checks = [], []

    def generator(request):
        calls.append(copy.deepcopy(request))
        return copy.deepcopy(defaults if cards is None else cards)

    def verify(request):
        checks.append(copy.deepcopy(request))
        return verifier(request) if verifier else {'supported': True, 'confidence': 'high'}

    run = pilot.generate_dimension_pilot(
        scope=scope, knowledge_version='synthetic-k', interest_context=context,
        dimension_projection=projection,
        coverage_provider=provider or (lambda: copy.deepcopy(coverage)),
        generator=generator, private_verifier=verify,
        trusted_context=original if trusted is None else trusted, trusted_store=store)
    return run, calls, checks


def test_empty_request_generates_neutral_scopes_and_persists_binding(setup_case, tmp_path):
    run, calls, checks = generate(setup_case)
    context, coverage, _, _, store, trusted = setup_case
    assert trusted.scope == run['scope'] == calls[0]['scope'] == ''
    assert len(calls) == 1 and 5 <= len(run['cards']) <= 7
    assert len(checks) == 6
    assert all(check['scope'] == '' for check in checks)
    for card in run['cards']:
        scope = card['scope'].casefold()
        assert 'периметр' in scope
        assert any(word in scope for word in ('не задан', 'не определ', 'не указан'))
        assert any(word in scope for word in ('планирован', 'предлож', 'определить'))
        assert 'насос' not in scope
        assert card['missing_evidence']
    assert len({c['decision'] for c in run['cards']}) == len(run['cards'])
    assert len({c['metrics_method'] for c in run['cards']}) == len(run['cards'])
    assert run['generation_id'] == coverage['generation']['run_id']
    path = tmp_path / 'result.json'
    path.write_text(json.dumps(run, ensure_ascii=False), encoding='utf-8')
    restored = json.loads(path.read_text(encoding='utf-8'))
    assert restored == run
    store.verify(trusted, restored, sealed=True)
    pilot.validate_pilot(restored, coverage=coverage, interest_context=context,
                         trusted_context=trusted, trusted_store=store)
    assert '987654321' not in path.read_text(encoding='utf-8')


def test_hypothetical_model_scope_is_kept_and_verified(setup_case):
    cards = copy.deepcopy(setup_case[3])
    proposed = 'Предложить для проверки оборудование и период при планировании'
    cards[0]['scope'] = proposed
    run, _, checks = generate(setup_case, cards=cards)
    assert proposed in [card['scope'] for card in run['cards']]
    assert any(check['card']['scope'] == proposed for check in checks)
    assert run['scope'] == ''


def test_empty_scope_does_not_retry_rejected_candidates(setup_case):
    rejected_dim = setup_case[3][0]['meeting_interest']['dimension_id']
    def verifier(request):
        if request['dimension_id'] == rejected_dim:
            return {'supported': False, 'confidence': 'high', 'reason': 'не подтверждено'}
        return {'supported': True, 'confidence': 'high'}
    run, calls, checks = generate(setup_case, verifier=verifier)
    assert len(calls) == 1 and len(checks) == 6
    assert len(run['cards']) == 5
    assert rejected_dim not in {c['meeting_interest']['dimension_id'] for c in run['cards']}
    assert run['rejected_cards'][0]['verdict']['reason'] == 'не подтверждено'


def test_explicit_scope_requires_ordinary_scope_validation(setup_case, tmp_path):
    store = setup_case[4]
    trusted = store.create({'run_id': 'explicit', 'case_id': 'case', 'user_id': 'user',
                            'scope': 'насосы участка сгущения'})
    cards = copy.deepcopy(setup_case[3])
    for card in cards:
        card['scope'] = trusted.scope
    cards[0]['scope'] = ''
    cards[1]['scope'] = ''
    with pytest.raises(ValueError):
        generate(setup_case, cards=cards, scope=trusted.scope, trusted=trusted)
    cards[0]['scope'] = 'всё оборудование'
    cards[1]['scope'] = 'всё оборудование'
    with pytest.raises(ValueError):
        generate(setup_case, cards=cards, scope=trusted.scope, trusted=trusted)
    cards[0]['scope'] = trusted.scope
    cards[1]['scope'] = trusted.scope
    run, _, _ = generate(setup_case, cards=cards, scope=trusted.scope, trusted=trusted)
    assert all(c['scope'] == trusted.scope for c in run['cards'])


def test_empty_scope_still_requires_authentic_binding_and_stable_generation(setup_case):
    with pytest.raises(ValueError):
        generate(setup_case, trusted=replace(setup_case[5], scope='invented'))
    calls = []
    def drifting():
        coverage = copy.deepcopy(setup_case[1])
        calls.append(1)
        if len(calls) > 1:
            coverage['generation']['run_id'] = 'different-generation'
        return coverage
    with pytest.raises(ValueError):
        generate(setup_case, provider=drifting)


def test_empty_scope_does_not_hide_missing_coverage_or_invent_facts(setup_case):
    def unavailable():
        raise OSError('synthetic coverage unavailable')
    with pytest.raises(RuntimeError):
        generate(setup_case, provider=unavailable)
    cards = copy.deepcopy(setup_case[3])
    cards[0]['analyst_brief'] = 'Компания СЕКРЕТ потратила 987654321 рублей.'
    run, _, _ = generate(setup_case, cards=cards)
    assert len(run['cards']) == 5
    assert all('987654321' not in card['analyst_brief'] for card in run['cards'])
    assert run['rejected_cards']
