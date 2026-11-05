"""Versioned chronological training contracts, using synthetic data only."""
import hashlib
import json
import sys
from datetime import date

import pytest

from scripts import hypotheses_fresh_training as cli
from tof_reliability.vitaly_bot import hypotheses_fresh_training as fresh


SEPTEMBER = date(2026, 9, 17)
OCTOBER = date(2026, 10, 26)
DAYS = ('2026-09-16', '2026-09-17', '2026-10-25', '2026-10-26', '2026-10-30')


def sha(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def synthetic(tmp_path):
    directory = tmp_path / 'indexes'
    directory.mkdir()
    sources, chunks, texts, reconciled = [], [], {}, []
    for i, day in enumerate(DAYS):
        sid = f's{i}'
        text = f'unique private synthetic source {sid}'
        path = directory / f'{sid}.txt'
        path.write_text(text, encoding='utf-8')
        row = {'id': sid, 'date': day, 'kind': 'baseline' if i % 2 else 'transcript',
               'path': str(path), 'sha256': sha(text)}
        sources.append(row)
        reconciled.append({**row, 'status': 'accepted', 'original_path': str(path),
                           'original_sha256': sha(text)})
        chunks.append({'id': f'chunk-{sid}', 'source_id': sid, 'start': 0,
                       'end': len(text), 'sha256': sha(text)})
        texts[sid] = text
    for name, content in [('source-index.json', {'sources': sources}),
                          ('chunk-index-v2.json', chunks),
                          ('reconciled-sources-v2.json', reconciled)]:
        (directory / name).write_text(json.dumps(content), encoding='utf-8')
    return directory, sources, chunks, texts


def file_hashes(directory):
    return {str(p.relative_to(directory)): sha(p.read_bytes())
            for p in directory.rglob('*') if p.is_file()}


def batch_ids(batches):
    return {sid for batch in batches for sid in batch['source_ids']}


def test_library_default_preserves_september_holdout(synthetic):
    _, sources, chunks, texts = synthetic
    batches = fresh.prepare_fresh_batches(index={'sources': sources}, chunks=chunks,
                                          source_texts=texts, batch_size=1)
    assert batch_ids(batches) == {'s0'}
    assert all(texts[sid] not in batch['prompt'] for sid in ('s1', 's2', 's3', 's4')
               for batch in batches)


@pytest.mark.parametrize(('cutoff', 'expected'), [
    (SEPTEMBER, {'s0'}),
    (OCTOBER, {'s0', 's1', 's2'}),
])
def test_library_explicit_cutoff_is_exclusive(synthetic, cutoff, expected):
    _, sources, chunks, texts = synthetic
    batches = fresh.prepare_fresh_batches(index={'sources': sources}, chunks=chunks,
                                          source_texts=texts, batch_size=1, cutoff=cutoff)
    assert batch_ids(batches) == expected
    assert all(texts[sid] not in batch['prompt'] for sid in set(texts) - expected
               for batch in batches)


@pytest.mark.parametrize(('option', 'expected'), [
    ([], {'s0'}),
    (['--cutoff', '2026-09-17'], {'s0'}),
    (['--cutoff', '2026-10-26'], {'s0', 's1', 's2'}),
])
def test_cli_cutoff_reconciles_and_propagates_to_extraction(synthetic, tmp_path, monkeypatch, option, expected):
    directory, _, _, texts = synthetic
    original_hashes = file_hashes(directory)
    output = tmp_path / 'private-output'
    prompts = []
    monkeypatch.setattr(cli, 'production_idle', lambda: None)
    monkeypatch.setattr(cli, 'invoke_factory', lambda *_: lambda prompt: prompts.append(prompt) or 'trace')
    monkeypatch.setattr(cli, 'parse_trace', lambda _: {'themes': []})
    monkeypatch.setattr(sys, 'argv', ['fresh', '--index-dir', str(directory), '--output', str(output), *option])
    cli.main()
    result = json.loads((output / 'fresh-extraction.json').read_text())
    assert {row['source_id'] for row in result['coverage']} == expected
    assert len(prompts) == len(expected)
    assert all(texts[sid] not in prompt for sid in set(texts) - expected for prompt in prompts)
    manifest = json.loads((output / 'coverage-manifest.json').read_text())
    assert set(manifest['source_ids']) == expected
    assert file_hashes(directory) == original_hashes


@pytest.mark.parametrize(('first', 'second'), [(SEPTEMBER, OCTOBER), (OCTOBER, SEPTEMBER)])
def test_checkpoint_rejects_changed_cutoff_even_with_identical_batches(synthetic, tmp_path, first, second):
    _, sources, chunks, texts = synthetic
    # Only pre-September material: changing the cutoff leaves batches identical,
    # so matching prompt or source hashes alone cannot enforce the contract.
    checkpoint = tmp_path / 'fresh-checkpoint.json'
    kwargs = dict(index={'sources': sources[:1]}, chunks=chunks[:1], source_texts=texts,
                  checkpoint_path=checkpoint, batch_size=1,
                  check_queues=lambda: {'production_user_idle': True, 'learning_queue_idle': True})
    fresh.run_fresh_extraction(**kwargs, cutoff=first,
                               invoke=lambda _: {'trace': 'synthetic', 'themes': []})
    original = checkpoint.read_bytes()
    with pytest.raises(ValueError, match='(?i)cutoff|checkpoint|provenance|version'):
        fresh.run_fresh_extraction(**kwargs, cutoff=second,
                                   invoke=lambda _: pytest.fail('must reject before invocation'))
    assert checkpoint.read_bytes() == original


def test_cli_rejects_changed_cutoff_without_modifying_existing_artifacts(synthetic, tmp_path, monkeypatch):
    directory, _, _, _ = synthetic
    output = tmp_path / 'private-output'
    monkeypatch.setattr(cli, 'production_idle', lambda: None)
    monkeypatch.setattr(cli, 'invoke_factory', lambda *_: lambda _: 'trace')
    monkeypatch.setattr(cli, 'parse_trace', lambda _: {'themes': []})
    base = ['fresh', '--index-dir', str(directory), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', base + ['--cutoff', '2026-10-26'])
    cli.main()
    before_input = file_hashes(directory)
    before_output = file_hashes(output)
    monkeypatch.setattr(cli, 'production_idle', lambda: pytest.fail('must reject before queue check'))
    monkeypatch.setattr(cli, 'invoke_factory', lambda *_: pytest.fail('must reject before model setup'))
    monkeypatch.setattr(sys, 'argv', base + ['--cutoff', '2026-09-17'])
    with pytest.raises(ValueError, match='(?i)cutoff|checkpoint|provenance|version'):
        cli.main()
    assert file_hashes(directory) == before_input
    assert file_hashes(output) == before_output
