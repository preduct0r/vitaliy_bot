"""Synthetic red-first contracts for the fresh, private training extraction.

No actual transcripts, old observations, or model invocations are used here.
"""
from __future__ import annotations

import hashlib
import importlib

import pytest


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@pytest.fixture
def fresh():
    name = 'tof_reliability.vitaly_bot.hypotheses_fresh_training'
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        class MissingFreshTraining:
            def __getattr__(self, attribute):
                pytest.fail('fresh training extraction orchestration is not implemented')

        return MissingFreshTraining()


@pytest.fixture
def corpus():
    texts = {'early': 'training alpha', 'early2': 'training beta',
             'holdout': 'HOLDOUT_SECRET', 'later': 'OCTOBER_SECRET'}
    index = {'sources': [
        {'id': sid, 'date': date, 'sha256': digest(texts[sid])}
        for sid, date in [('early', '2026-09-16'), ('early2', '2026-09-01'),
                          ('holdout', '2026-09-17'), ('later', '2026-10-26')]
    ]}
    chunks = [{'id': 'chunk-' + sid, 'source_id': sid, 'start': 0,
               'end': len(text), 'sha256': digest(text)}
              for sid, text in texts.items()]
    return index, chunks, texts


def test_fresh_prompts_bind_only_verified_pre_cutoff_chunks(fresh, corpus):
    index, chunks, texts = corpus
    batches = fresh.prepare_fresh_batches(index=index, chunks=chunks,
                                           source_texts=texts, batch_size=1)
    assert {sid for b in batches for sid in b['source_ids']} == {'early', 'early2'}
    assert {cid for b in batches for cid in b['chunk_ids']} == {'chunk-early', 'chunk-early2'}
    for batch in batches:
        prompt = batch['prompt']
        assert batch['prompt_sha256'] == digest(prompt)
        assert set(batch['source_hashes']) == set(batch['source_ids'])
        assert all(batch['source_hashes'][sid] == digest(texts[sid]) for sid in batch['source_ids'])
        assert all(texts[sid] in prompt for sid in batch['source_ids'])
        assert 'HOLDOUT_SECRET' not in prompt and 'OCTOBER_SECRET' not in prompt
        assert 'holdout' not in prompt and 'later' not in prompt


def test_modified_original_or_chunk_is_rejected_before_prompt(fresh, corpus):
    index, chunks, texts = corpus
    texts['early'] = 'tampered text'
    with pytest.raises(ValueError, match='(?i)hash|source|integrity'):
        fresh.prepare_fresh_batches(index=index, chunks=chunks, source_texts=texts, batch_size=1)


def test_resume_uses_only_fresh_verified_results_and_checks_each_invocation(fresh, corpus, tmp_path):
    index, chunks, texts = corpus
    checkpoint = tmp_path / 'fresh-checkpoint.json'
    calls = []
    checks = []

    def queues():
        checks.append('checked')
        return {'production_user_idle': True, 'learning_queue_idle': True}

    def invoke(prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise RuntimeError('temporary provider failure')
        return {'trace': 'synthetic tool-disabled trace',
                'themes': [{'label': 'generic topic', 'fragment': 'training alpha'}]}

    with pytest.raises(RuntimeError, match='temporary provider failure'):
        fresh.run_fresh_extraction(index=index, chunks=chunks, source_texts=texts,
                                   checkpoint_path=checkpoint, batch_size=1,
                                   check_queues=queues, invoke=invoke)
    assert checkpoint.exists()
    assert len(checks) == len(calls) == 2
    completed_prompt = calls[0]

    def resume(prompt):
        calls.append(prompt)
        assert prompt != completed_prompt
        return {'trace': 'synthetic tool-disabled trace',
                'themes': [{'label': 'generic topic', 'fragment': 'training beta'}]}

    result = fresh.run_fresh_extraction(index=index, chunks=chunks, source_texts=texts,
                                        checkpoint_path=checkpoint, batch_size=1,
                                        check_queues=queues, invoke=resume)
    assert len(checks) == len(calls) == 3
    assert {row['chunk_id'] for row in result['coverage']} == {'chunk-early', 'chunk-early2'}
    assert all(row['status'] == 'processed' and row['prompt_sha256'] and row['trace']
               for row in result['coverage'])
    assert {e['fragment'] for e in result['evidence']} == {'training alpha', 'training beta'}
    assert all(e['fragment'] in texts[e['source_id']] for e in result['evidence'])


def test_old_observation_checkpoint_cannot_be_resumed(fresh, corpus, tmp_path):
    index, chunks, texts = corpus
    checkpoint = tmp_path / 'old.json'
    checkpoint.write_text('{"observations": [{"label": "OLD_OUTPUT_CANARY"}]}')
    with pytest.raises(ValueError, match='(?i)fresh|checkpoint|provenance'):
        fresh.run_fresh_extraction(index=index, chunks=chunks, source_texts=texts,
                                   checkpoint_path=checkpoint, batch_size=1,
                                   check_queues=lambda: pytest.fail('must reject old outputs first'),
                                   invoke=lambda _: pytest.fail('must not invoke model'))


def test_busy_production_queue_prevents_model_invocation(fresh, corpus, tmp_path):
    index, chunks, texts = corpus
    with pytest.raises(RuntimeError, match='(?i)queue|busy|production'):
        fresh.run_fresh_extraction(index=index, chunks=chunks, source_texts=texts,
                                   checkpoint_path=tmp_path / 'state.json', batch_size=1,
                                   check_queues=lambda: {'production_user_idle': False,
                                                         'learning_queue_idle': True},
                                   invoke=lambda _: pytest.fail('model must not run while busy'))


def test_resume_checks_result_matches_independently_parsed_trace(fresh, corpus, tmp_path):
    index, chunks, texts = corpus
    checkpoint = tmp_path / 'state.json'
    def validate(trace):
        return {'themes': [{'label': 'generic topic', 'fragment': trace}]}
    kwargs = dict(index=index, chunks=chunks, source_texts=texts, checkpoint_path=checkpoint,
                  batch_size=1, check_queues=lambda: {'production_user_idle': True,
                                                     'learning_queue_idle': True},
                  validate_trace=validate)
    def invoke(prompt):
        fragment = 'training alpha' if 'training alpha' in prompt else 'training beta'
        return {'trace': fragment, 'themes': [{'label': 'generic topic', 'fragment': fragment}]}
    fresh.run_fresh_extraction(**kwargs, invoke=invoke)
    state = __import__('json').loads(checkpoint.read_text())
    state['results']['0']['trace'] = 'unrelated old trace'
    checkpoint.write_text(__import__('json').dumps(state))
    with pytest.raises(ValueError, match='(?i)trace'):
        fresh.run_fresh_extraction(**kwargs, invoke=lambda _: pytest.fail('must reject altered trace'))
