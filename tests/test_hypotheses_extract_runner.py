"""Synthetic contracts for resumable, tool-isolated training extraction."""
import hashlib
import importlib
import json

import pytest


@pytest.fixture
def runner():
    name = 'tof_reliability.vitaly_bot.hypotheses_extract_runner'
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        class MissingRunner:
            def __getattr__(self, name):
                pytest.fail(f'batched extraction runner API {name} is not implemented')
        return MissingRunner()


def chunk(source_id, text):
    return {'source_id': source_id, 'start': 0, 'end': len(text),
            'sha256': hashlib.sha256(text.encode()).hexdigest()}


def events(content):
    return '\n'.join(json.dumps(event) for event in [
        {'type': 'thread.started', 'thread_id': 'synthetic'},
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': content}},
        {'type': 'turn.completed'},
    ])


def test_resumes_completed_batches_and_validates_full_manifest(runner, tmp_path):
    texts = {'private-source-a': 'first meeting', 'private-source-b': 'second meeting',
             'private-source-c': 'third meeting', 'holdout-private-id': 'HOLDOUT_SECRET'}
    index = {'sources': [{'id': key, 'date': date} for key, date in [
        ('private-source-a', '2026-10-01'), ('private-source-b', '2026-10-02'),
        ('private-source-c', '2026-10-03'), ('holdout-private-id', '2026-10-26')]]}
    chunks = [chunk(key, text) for key, text in texts.items()]
    checkpoint = tmp_path / 'checkpoint.json'
    calls = []

    def fail_second(prompt):
        calls.append(prompt)
        assert 'HOLDOUT_SECRET' not in prompt and 'holdout-private-id' not in prompt
        assert all(key not in prompt for key in texts)
        if len(calls) == 2:
            raise RuntimeError('synthetic temporary model failure')
        return events(json.dumps({'chunks': [{'themes': []}]}))

    with pytest.raises(RuntimeError, match='synthetic temporary model failure'):
        runner.run_batches(index=index, chunks=chunks, source_texts=texts,
                           checkpoint_path=checkpoint, batch_size=1, invoke=fail_second)
    assert checkpoint.exists()
    saved = json.loads(checkpoint.read_text())
    processed = [row for row in saved['manifest'] if row['status'] == 'processed']
    assert len(processed) == 1
    assert processed[0]['sha256'] == chunks[0]['sha256']

    resumed = []
    def success(prompt):
        resumed.append(prompt)
        assert 'first meeting' not in prompt
        assert 'HOLDOUT_SECRET' not in prompt
        assert all(key not in prompt for key in texts)
        return events(json.dumps({'chunks': [{'themes': []}]}))

    result = runner.run_batches(index=index, chunks=chunks, source_texts=texts,
                                checkpoint_path=checkpoint, batch_size=1, invoke=success)
    assert len(resumed) == 2
    assert len(result['manifest']) == 3
    from tof_reliability.vitaly_bot.hypotheses_interest_extraction import validate_training_coverage
    validate_training_coverage(chunks[:3], result['manifest'])


@pytest.mark.parametrize('tool_event', [
    {'type': 'item.started', 'item': {'type': 'command_execution', 'command': 'cat canary'}},
    {'type': 'item.completed', 'item': {'type': 'mcp_tool_call', 'tool': 'read_file'}},
    {'type': 'item.completed', 'item': {'type': 'web_search', 'query': 'secret'}},
    {'type': 'tool_call', 'name': 'shell'},
])
def test_any_tool_event_rejects_batch_without_checkpointing_success(runner, tmp_path, tool_event):
    text = 'synthetic training text'
    index = {'sources': [{'id': 'private-id', 'date': '2026-10-01'}]}
    chunks = [chunk('private-id', text)]
    log = events(json.dumps({'chunks': [{'themes': []}]})) + '\n' + json.dumps(tool_event)
    checkpoint = tmp_path / 'checkpoint.json'
    with pytest.raises(ValueError, match='(?i)tool|isolation'):
        runner.run_batches(index=index, chunks=chunks, source_texts={'private-id': text},
                           checkpoint_path=checkpoint, batch_size=1, invoke=lambda _: log)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        assert not any(row['status'] == 'processed' for row in state['manifest'])


def test_bad_evidence_and_changed_source_are_not_accepted(runner, tmp_path):
    text = 'a verified quote in a meeting'
    index = {'sources': [{'id': 'private-id', 'date': '2026-10-01'}]}
    chunks = [chunk('private-id', text)]
    bad = {'chunks': [{'themes': [{'requirement': 'test', 'evidence': [
        {'fragment': 'fabricated', 'attribution': 'direct_request'}]}]}]}
    checkpoint = tmp_path / 'checkpoint.json'
    with pytest.raises(ValueError, match='fragment'):
        runner.run_batches(index=index, chunks=chunks, source_texts={'private-id': text},
                           checkpoint_path=checkpoint, batch_size=1,
                           invoke=lambda _: events(json.dumps(bad)))
    assert json.loads(checkpoint.read_text())['manifest'][0]['status'] == 'pending'
    with pytest.raises(ValueError, match='hash'):
        runner.run_batches(index=index, chunks=chunks, source_texts={'private-id': 'X' + text[1:]},
                           checkpoint_path=checkpoint, batch_size=1,
                           invoke=lambda _: pytest.fail('must reject before calling the model'))


def test_multiple_agent_messages_rejected_as_possible_tool_attempt(runner):
    trace = events('{}') + '\n' + json.dumps(
        {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '{}'}})
    with pytest.raises(ValueError, match='tool attempt'):
        runner.parse_trace(trace)


def test_valid_chunk_survives_failed_peer_after_trace_validation(runner, tmp_path):
    texts = {'a': 'first quote', 'b': 'second quote'}
    index = {'sources': [{'id': k, 'date': '2026-10-01'} for k in texts]}
    chunks = [chunk(k, v) for k, v in texts.items()]
    theme = lambda fragment: {'themes': [{'requirement': 'claim', 'evidence': [
        {'fragment': fragment, 'attribution': 'interpretation'}]}]}
    checkpoint = tmp_path / 'state.json'
    with pytest.raises(ValueError, match='fragment'):
        runner.run_batches(index=index, chunks=chunks, source_texts=texts,
                           checkpoint_path=checkpoint, batch_size=2,
                           invoke=lambda _: events(json.dumps({'chunks': [theme('first quote'), theme('invented')]})))
    state = json.loads(checkpoint.read_text())
    assert [r['status'] for r in state['manifest']] == ['processed', 'pending']
    result = runner.run_batches(index=index, chunks=chunks, source_texts=texts,
                                checkpoint_path=checkpoint, batch_size=2,
                                invoke=lambda _: events(json.dumps({'chunks': [theme('second quote')]})))
    assert [r['status'] for r in result['manifest']] == ['processed', 'processed']


def test_grouping_preserves_observations_across_batches_and_resume(tmp_path):
    from scripts.hypotheses_group_training import group_observations
    requirements = [f'claim {i}' for i in range(5)]
    path = tmp_path / 'groups.json'
    calls = []

    def fail_second(batch):
        calls.append(batch)
        if len(calls) == 2:
            raise RuntimeError('temporary')
        return [{'requirement': 'first group', 'members': list(range(len(batch)))}]

    with pytest.raises(RuntimeError, match='temporary'):
        group_observations(requirements, path, fail_second, batch_size=2)
    resumed = []
    def success(batch):
        resumed.append(batch)
        return [{'requirement': 'merged', 'members': list(range(len(batch)))}]
    result = group_observations(requirements, path, success, batch_size=2)
    assert all('claim 0' not in batch for batch in resumed)
    assert sorted(i for group in result for i in group['members']) == list(range(5))


def test_direct_attribution_requires_identifiable_speaker():
    from scripts.hypotheses_group_training import ground_attribution
    evidence = {'attribution': 'direct_request', 'fragment': 'Проверьте данные',
                'locator': 'chars:9-25', 'source_id': 'synthetic'}
    direct = ground_attribution(evidence, 'Виталий: Проверьте данные')
    assert direct['attribution'] == 'direct_request'
    ambiguous = ground_attribution(evidence, 'Неизвест: Проверьте данные')
    assert ambiguous['attribution'] == 'interpretation'
    assert ambiguous['reported_attribution'] == 'direct_request'


def test_final_map_retains_individual_claims_inside_group(tmp_path, monkeypatch):
    import sys
    from scripts import hypotheses_group_training as grouping
    source = tmp_path / 'source.txt'
    text = 'The original observation is here.'
    source.write_text(text)
    row = chunk('source', text)
    index = {'version': 'v1', 'sources': [{'id': 'source', 'date': '2026-10-01',
             'kind': 'transcript', 'path': str(source), 'sha256': row['sha256']}]}
    index_path, chunks_path = tmp_path / 'index.json', tmp_path / 'chunks.json'
    index_path.write_text(json.dumps(index))
    chunks_path.write_text(json.dumps([row]))
    state = {'manifest': [{**row, 'status': 'processed', 'attempts': 1}],
             'results': {'0': {'themes': [{'requirement': 'specific observation',
                 'evidence': [{'fragment': text, 'start': 0, 'end': len(text),
                               'attribution': 'interpretation'}]}]}}}
    (tmp_path / 'extraction-checkpoint.json').write_text(json.dumps(state))
    monkeypatch.setattr(sys, 'argv', ['group', '--index', str(index_path), '--chunks',
                                     str(chunks_path), '--output', str(tmp_path)])
    grouping.main()
    interest = json.loads((tmp_path / 'training-interest-map.json').read_text())['interests'][0]
    assert interest['observations'][0]['requirement'] == 'specific observation'
    assert interest['observations'][0]['evidence'][0]['fragment'] == text
