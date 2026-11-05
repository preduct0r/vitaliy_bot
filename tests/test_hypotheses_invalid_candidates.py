"""Development-only regressions using synthetic fixtures and an offline runner.

These tests provide no new independent holdout and do not establish model or
production readiness. All evidence, case materials and model answers are fake.
"""
import asyncio
import copy
import hashlib
import json
import re

import pytest

from test_hypotheses_dimensions_pilot import dimension_case, topic_case  # fixtures
from test_hypotheses_service_runtime import add_text, drain, make_runtime, persisted, submit
from tof_reliability.vitaly_bot.hypotheses_ui import render
from tof_reliability.vitaly_bot.telegram import user_error


UNSUPPORTED = 'согласованный периметр'
PRIVATE_MARKER = 'SYNTHETIC_PRIVATE_CANDIDATE_TEXT'


def instrument(runner, mutate):
    """Alter only generated cards; record independent verifier inputs."""
    original = runner.run
    generated, checked = [], []

    async def run(prompt, schema, image_paths=None):
        result = await original(prompt, schema, image_paths)
        if 'cards' in schema['properties']:
            cards = result.value['cards']
            mutate(cards, prompt)
            generated.append(copy.deepcopy(cards))
            assert len(generated) <= 5, 'regeneration must be bounded'
        else:
            checked.append(json.loads(prompt.split('\n', 1)[1]))
        return result

    runner.run = run
    return generated, checked


def assert_public_binding(card, context):
    meeting = card['meeting_interest']
    interest = next(item for item in context['interests'] if item['id'] == meeting['interest_id'])
    evidence = interest['evidence'][0]
    digest = hashlib.sha256((evidence['source_id'] + '\0' + evidence['locator']).encode()).hexdigest()[:16]
    assert digest in meeting['reference']
    assert meeting['attribution'] == evidence['attribution']
    assert all(key not in meeting for key in ('source_id', 'locator', 'sha256'))
    assert 'chars:' not in render(card)


@pytest.mark.parametrize('scope,with_material', [('', False), ('насосы участка сгущения', True)])
def test_service_discards_invalid_candidate_and_returns_five_independently_verified_cards(
        tmp_path, dimension_case, scope, with_material):
    store, service, hypotheses, runner, provider, knowledge, trusted = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    if with_material:
        add_text(store, case, tmp_path, 'current.txt', 'CURRENT_CASE_SYNTHETIC')

    def invalidate_one(cards, _prompt):
        cards[0]['scope'] = (scope + ', ' if scope else '') + UNSUPPORTED
        cards[0]['analyst_brief'] += ' ' + PRIVATE_MARKER
        if with_material:
            # The fixture attaches the current material to card zero by default;
            # attach it to card one instead to check attribution after filtering.
            cards[1]['case_material_references'] = cards[0]['case_material_references']
            cards[0]['case_material_references'] = []

    generated, checked = instrument(runner, invalidate_one)
    job = submit(hypotheses, case, description=scope)
    completed, failures = asyncio.run(drain(store, service, job))
    assert not failures, f'one invalid candidate must not fail the service: {failures}'
    assert len(completed) == 1
    run = persisted(hypotheses, job)
    assert run['status'] == 'completed' and len(run['cards']) == 5
    assert run['generation_id'] == provider.coverage['generation']['run_id']
    assert len(provider.calls) >= 3 and runner.max_active == 1
    bad = generated[0][0]['meeting_interest']['dimension_id']
    expected = {c['meeting_interest']['dimension_id']: c for c in generated[0][1:]}
    assert {c['meeting_interest']['dimension_id'] for c in run['cards']} == set(expected)
    assert bad not in {check['dimension_id'] for check in checked}
    assert {check['dimension_id'] for check in checked} == set(expected)
    assert len(checked) == len(expected)
    assert [c['number'] for c in run['cards']] == list(range(1, 6))
    for card in run['cards']:
        original = expected[card['meeting_interest']['dimension_id']]
        assert card['decision'] == original['decision']
        assert card['metrics_method'] == original['metrics_method']
        assert card['case_material_references'] == original['case_material_references']
        if scope:
            assert card['scope'] == scope
        assert_public_binding(card, knowledge.context)
    assert PRIVATE_MARKER not in json.dumps(run['cards'], ensure_ascii=False)
    assert UNSUPPORTED not in '\n'.join(render(card) for card in run['cards'])
    if with_material:
        assert sum(bool(c['case_material_references']) for c in run['cards']) == 1
        assert 'current.txt' in json.dumps(run['cards'], ensure_ascii=False)
    request = {'run_id': run['id'], 'case_id': case['id'], 'user_id': '101', 'scope': scope}
    _, sealed = trusted.recover(request)
    diagnostics = json.dumps(sealed.get('rejected_cards', []), ensure_ascii=False).casefold()
    assert 'agreement' in diagnostics or 'согласован' in diagnostics
    assert bad in diagnostics


@pytest.mark.parametrize('bad_count', [2, 6])
def test_too_few_valid_candidates_fail_with_safe_classification(tmp_path, dimension_case, bad_count):
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)

    def invalidate(cards, _prompt):
        for card in cards[:bad_count]:
            card['scope'] = UNSUPPORTED
            card['analyst_brief'] += ' ' + PRIVATE_MARKER

    generated, _ = instrument(runner, invalidate)
    job = submit(hypotheses, case)
    completed, failures = asyncio.run(drain(store, service, job))
    run = persisted(hypotheses, job)
    assert not completed and len(failures) == 1
    assert run['status'] == 'failed' and not run['cards']
    assert 1 <= len(generated) <= 5
    error = failures[0][1]
    classification = (str(getattr(error, 'code', '')) + ' ' + str(error) + ' ' + str(run['error'])).casefold()
    assert re.search(r'insufficient|fewer|five|not.enough|недостат|меньше.пяти', classification)
    message = user_error(error)
    assert re.search(r'недостат|меньше|не удалось.*провер|insufficient|fewer', message.casefold())
    assert PRIVATE_MARKER not in message and UNSUPPORTED not in message


def test_rejected_duplicate_cannot_steal_valid_cards_case_materials(tmp_path, dimension_case):
    store, service, hypotheses, runner, _, knowledge, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    add_text(store, case, tmp_path, 'first.txt', 'CURRENT_CASE_SYNTHETIC')
    add_text(store, case, tmp_path, 'second.txt', 'SECOND_SYNTHETIC_MATERIAL')

    def append_invalid_duplicate(cards, prompt):
        materials = json.loads(prompt.rsplit('иначе пустой список.\n', 1)[1])
        valid = cards[0]
        duplicate = copy.deepcopy(valid)
        duplicate['scope'] = UNSUPPORTED
        duplicate['case_material_references'] = [next(m['reference'] for m in materials if 'SECOND_SYNTHETIC_MATERIAL' in m['text'])]
        cards.append(duplicate)

    generated, checked = instrument(runner, append_invalid_duplicate)
    job = submit(hypotheses, case)
    completed, failures = asyncio.run(drain(store, service, job))
    assert not failures and completed
    run = persisted(hypotheses, job)
    assert run['status'] == 'completed'
    keys = [(c['meeting_interest']['interest_id'], c['meeting_interest']['dimension_id']) for c in run['cards']]
    assert len(keys) == len(set(keys)) == len({key[1] for key in keys})
    original = generated[0][0]
    dim = original['meeting_interest']['dimension_id']
    selected = next(c for c in run['cards'] if c['meeting_interest']['dimension_id'] == dim)
    assert selected['case_material_references'] == original['case_material_references']
    assert 'second.txt' not in json.dumps(selected, ensure_ascii=False)
    assert_public_binding(selected, knowledge.context)
    assert sum(c['dimension_id'] == dim for c in checked) == 1


def test_generation_drift_after_filtering_and_verification_never_returns_stale_cards(tmp_path, dimension_case):
    store, service, hypotheses, runner, provider, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    original_provider = provider.build

    async def drifting(request):
        result = await original_provider(request)
        # Both pilot attempts see a stable initial snapshot and a different
        # snapshot after review; a bounded service retry cannot publish either.
        if len(provider.calls) in (3, 6):
            result['generation']['run_id'] += '-changed'
        return result

    provider.build = drifting
    generated, checked = instrument(runner, lambda cards, _prompt: cards[0].update(scope=UNSUPPORTED))
    job = submit(hypotheses, case)
    completed, failures = asyncio.run(drain(store, service, job))
    run = persisted(hypotheses, job)
    assert not completed and failures and run['status'] == 'failed' and not run['cards']
    assert len(generated) >= 1 and checked
    assert len(provider.calls) >= 3
    assert re.search(r'generation|coverage|поколен', str(failures[0][1]).casefold())
