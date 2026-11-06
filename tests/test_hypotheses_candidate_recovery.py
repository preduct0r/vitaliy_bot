"""Offline recovery contracts; all inputs and model responses are synthetic."""
import asyncio
import copy
import json
from collections import Counter

import pytest

from test_hypotheses_dimensions_pilot import dimension_case, topic_case  # fixtures
from test_hypotheses_service_runtime import add_text, drain, make_runtime, persisted, submit
from tof_reliability.vitaly_bot.models import ModelResult


PRIVATE = 'SYNTHETIC_PRIVATE_REJECTED_CONTENT'


def instrument(runner, mutate):
    original = runner.run
    batches, reviews = [], []

    async def run(prompt, schema, image_paths=None):
        result = await original(prompt, schema, image_paths)
        if 'cards' in schema['properties']:
            cards = copy.deepcopy(result.value['cards'])
            mutate(cards, len(batches), prompt)
            batches.append((copy.deepcopy(cards), prompt))
            assert len(batches) <= 5, 'candidate regeneration must be bounded'
            return ModelResult({'cards': cards}, '', '')
        request = json.loads(prompt.split('\n', 1)[1])
        reviews.append(request)
        return result

    runner.run = run
    return batches, reviews


def test_recovery_preserves_verified_cards_and_case_evidence(tmp_path, dimension_case):
    store, service, hypotheses, runner, provider, _, trusted = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    add_text(store, case, tmp_path, 'case.txt', 'CURRENT_CASE_SYNTHETIC')

    def mutate(cards, batch, prompt):
        if batch == 0:
            # Four valid cards survive; the originals in the next response are
            # valid replacements. One bad reference and one unsupported claim.
            cards[1]['case_material_references'] = ['Файл: чужой-кейс.txt']
            cards[2]['scope'] = 'согласованный периметр'
            cards[2]['analyst_brief'] += ' ' + PRIVATE

    batches, reviews = instrument(runner, mutate)
    job = submit(hypotheses, case)
    completed, failures = asyncio.run(drain(store, service, job))
    assert not failures, f'partial approvals should be recovered: {failures}'
    assert completed and 2 <= len(batches) <= 5
    run = persisted(hypotheses, job)
    assert run['status'] == 'completed' and 5 <= len(run['cards']) <= 7
    assert run['generation_id'] == provider.coverage['generation']['run_id']
    dims = [c['meeting_interest']['dimension_id'] for c in run['cards']]
    assert len(dims) == len(set(dims))
    initial = {c['meeting_interest']['dimension_id']: c for i, c in enumerate(batches[0][0]) if i not in (1, 2)}
    assert set(initial) <= set(dims)
    for card in run['cards']:
        dim = card['meeting_interest']['dimension_id']
        if dim in initial:
            assert card['decision'] == initial[dim]['decision']
            assert card['metrics_method'] == initial[dim]['metrics_method']
            assert card['case_material_references'] == initial[dim]['case_material_references']
    assert any('case.txt' in str(c['case_material_references']) for c in run['cards'])
    assert set(dims) <= {r['dimension_id'] for r in reviews}
    assert runner.max_active == 1
    retry_prompt = batches[1][1]
    assert all(dim in retry_prompt for dim in initial), 'retry must exclude accepted dimensions'
    assert PRIVATE not in retry_prompt
    request = {'run_id': run['id'], 'case_id': case['id'], 'user_id': '101', 'scope': ''}
    _, sealed = trusted.recover(request)
    assert sealed['generation_id'] == run['generation_id']
    assert sealed['knowledge_version'] == run['knowledge_version']
    assert sealed['recovery_diagnostics']['attempts'] == len(batches)
    assert sealed['recovery_diagnostics']['rejection_counts']['case_references'] == 1
    assert PRIVATE not in json.dumps(sealed.get('rejected_cards', []), ensure_ascii=False)


def test_exhausted_recovery_is_bounded_and_fails_closed(tmp_path, dimension_case):
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)

    def mutate(cards, batch, prompt):
        for card in cards:
            card['scope'] = 'согласованный периметр'
            card['analyst_brief'] += ' ' + PRIVATE

    batches, reviews = instrument(runner, mutate)
    job = submit(hypotheses, case)
    completed, failures = asyncio.run(drain(store, service, job))
    run = persisted(hypotheses, job)
    assert not completed and len(failures) == 1
    assert run['status'] == 'failed' and not run['cards']
    assert 2 <= len(batches) <= 5
    assert len(reviews) <= 12 * len(batches)
    assert PRIVATE not in str(run['error'])
    assert 'diagnostics=' in str(run['error'])
    assert 'unsupported_agreement' in str(run['error'])
    assert PRIVATE not in ' '.join(prompt for _, prompt in batches[1:])


def test_recovery_revalidates_duplicate_and_unsupported_replacements(tmp_path, dimension_case):
    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)

    def mutate(cards, batch, prompt):
        if batch == 0:
            for card in cards[:2]:
                card['scope'] = 'согласованный периметр'
        else:
            # Repeated valid dimensions cannot substitute for missing ones;
            # the unsupported numeric claim must never enter the result.
            cards[0] = copy.deepcopy(cards[3])
            cards[1]['decision'] = 'Простой составил 987654 часов.'

    batches, reviews = instrument(runner, mutate)
    job = submit(hypotheses, case)
    completed, failures = asyncio.run(drain(store, service, job))
    assert not completed and len(failures) == 1
    assert 2 <= len(batches) <= 5
    assert persisted(hypotheses, job)['status'] == 'failed'
    assert not persisted(hypotheses, job)['cards']
    assert '987654' not in json.dumps(reviews, ensure_ascii=False)
    counts = Counter(r['dimension_id'] for r in reviews)
    assert all(count <= len(batches) for count in counts.values())


def test_drift_during_recovery_discards_old_candidates_and_uses_one_retry(tmp_path, dimension_case):
    store, service, hypotheses, runner, provider, _, trusted = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    normal = provider.build

    async def drift(request):
        result = await normal(request)
        if len(provider.calls) >= 3:
            result['generation']['run_id'] += '-new'
        return result

    provider.build = drift

    def mutate(cards, batch, prompt):
        if batch == 0:
            for card in cards[:2]:
                card['scope'] = 'согласованный периметр'

    batches, _ = instrument(runner, mutate)
    job = submit(hypotheses, case)
    completed, failures = asyncio.run(drain(store, service, job))
    assert completed and not failures
    run = persisted(hypotheses, job)
    assert run['generation_id'].endswith('-new')
    assert len(batches) == 3 and len(provider.calls) == 6
    request = {'run_id': run['id'], 'case_id': case['id'], 'user_id': '101', 'scope': ''}
    _, sealed = trusted.recover(request)
    assert sealed['generation_id'] == run['generation_id']
    assert sealed['recovery_diagnostics']['attempts'] == 1


def test_recovery_selection_never_drops_previously_verified_cards(monkeypatch):
    from tof_reliability.vitaly_bot import hypotheses_pilot as pilot

    cards = [{'identity': n} for n in range(10)]
    monkeypatch.setattr(pilot, '_balanced_dimension_selection', lambda pool, rows: list(reversed(pool))[:7])
    result = pilot._recovery_selection(cards, {}, preserved_count=4)
    assert len(result) == 7
    assert all(card in result for card in cards[:4])
    assert len({card['identity'] for card in result}) == 7


def test_recovery_time_budget_cancels_stalled_model(tmp_path, dimension_case, monkeypatch):
    import time
    from types import SimpleNamespace
    from tof_reliability.vitaly_bot import hypotheses_service as service_module

    store, service, hypotheses, runner, _, _, _ = make_runtime(tmp_path, dimension_case)
    case = service.new_case(101)
    cancelled = []
    original = runner.run

    async def slow(prompt, schema, image_paths=None):
        if 'cards' not in schema['properties']:
            return await original(prompt, schema, image_paths)
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)

    calls = 0
    def clock():
        nonlocal calls
        calls += 1
        # Leave setup and the first coverage check unhurried; shorten only
        # the budget seen by the generator after it has been scheduled.
        return time.monotonic() + (599.5 if calls >= 3 else 0)

    monkeypatch.setattr(service_module, 'time', SimpleNamespace(monotonic=clock))
    runner.run = slow
    job = submit(hypotheses, case)
    completed, failures = asyncio.run(drain(store, service, job))
    assert not completed and failures
    assert persisted(hypotheses, job)['status'] == 'failed'
    assert cancelled == [True]
