"""Contracts for the historical October evaluation; all content is synthetic."""
import hashlib
import json
import sys
from datetime import date

import pytest

from scripts import hypotheses_fresh_training as cli
from tof_reliability.vitaly_bot import hypotheses_fresh_training as fresh


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize(('day', 'included'), [
    ('2026-09-16', True), ('2026-09-17', True), ('2026-10-25', True),
    ('2026-10-26', False), ('2026-10-28', False), ('2026-10-30', False),
    ('2026-11-01', False),
])
def test_cutoff_is_strictly_before_october_26(day, included):
    text = 'synthetic fragment'
    source = {'id': 's', 'date': day, 'kind': 'baseline', 'sha256': digest(text)}
    chunk = {'id': 'c', 'source_id': 's', 'start': 0, 'end': len(text), 'sha256': digest(text)}
    batches = fresh.prepare_fresh_batches(index={'sources': [source]}, chunks=[chunk],
                                          source_texts={'s': text}, batch_size=1, cutoff=date(2026, 10, 26))
    assert bool(batches) is included


@pytest.fixture
def inventory(tmp_path):
    directory = tmp_path / 'indexes'
    directory.mkdir()
    sources, reconciled, chunks = [], [], []
    days = ['2026-09-16', '2026-09-17', '2026-09-24', '2026-09-25',
            '2026-10-02', '2026-10-02', '2026-10-05', '2026-10-23',
            '2026-10-26', '2026-10-28', '2026-10-30']
    for i, day in enumerate(days):
        for kind in ('baseline', 'transcript'):
            sid = f's{i}-{kind}'
            text = f'canary-{sid}-first|canary-{sid}-second'
            path = directory / (sid + '.txt')
            path.write_text(text, encoding='utf-8')
            source = {'id': sid, 'date': day, 'kind': kind, 'path': str(path), 'sha256': digest(text)}
            sources.append(source)
            reconciled.append({**source, 'status': 'accepted', 'original_path': str(path),
                               'original_sha256': digest(text)})
            split = text.index('|') + 1
            for n, (start, end) in enumerate(((0, split), (split, len(text)))):
                chunks.append({'id': f'{sid}-{n}', 'source_id': sid, 'start': start,
                               'end': end, 'sha256': digest(text[start:end])})
    # Same bytes are one source, while a non-accepted source is never sent.
    reconciled.append({**reconciled[0], 'id': 'duplicate', 'duplicate_of': reconciled[0]['id']})
    rejected = {**reconciled[1], 'id': 'rejected', 'status': 'rejected'}
    reconciled.append(rejected)
    for name, value in [('source-index.json', {'sources': sources}),
                        ('chunk-index-v2.json', chunks),
                        ('reconciled-sources-v2.json', reconciled)]:
        (directory / name).write_text(json.dumps(value), encoding='utf-8')
    return directory, sources, chunks


def test_cli_includes_all_accepted_sources_and_chunks_without_evaluation_leakage(inventory, tmp_path, monkeypatch):
    directory, sources, chunks = inventory
    output = tmp_path / 'private-output'
    prompts = []
    monkeypatch.setattr(cli, 'production_idle', lambda: None)
    monkeypatch.setattr(cli, 'invoke_factory', lambda *_: lambda prompt: prompts.append(prompt) or 'trace')
    monkeypatch.setattr(cli, 'parse_trace', lambda _: {'themes': []})
    monkeypatch.setattr(sys, 'argv', ['fresh', '--index-dir', str(directory), '--output', str(output), '--cutoff', '2026-10-26'])
    cli.main()
    expected = {s['id'] for s in sources if s['date'] < '2026-10-26'}
    expected_chunks = {c['id'] for c in chunks if c['source_id'] in expected}
    result = json.loads((output / 'fresh-extraction.json').read_text())
    coverage = result['coverage']
    assert {c['source_id'] for c in coverage} == expected
    assert {c['chunk_id'] for c in coverage} == expected_chunks
    assert len(coverage) == len(expected_chunks) == len(prompts)
    assert all(f'canary-{s["id"]}' not in prompt for s in sources if s['id'] not in expected for prompt in prompts)
    manifest = json.loads((output / 'coverage-manifest.json').read_text())
    assert manifest['expected']['sources'] == manifest['found']['sources'] == len(expected)
    assert manifest['expected']['chunks'] == manifest['found']['chunks'] == len(expected_chunks)
    assert not manifest['missing']['sources'] and not manifest['missing']['chunks']


def test_missing_late_chunk_blocks_every_model_call(inventory, tmp_path, monkeypatch):
    directory, _, chunks = inventory
    chunks = [c for c in chunks if c['id'] != 's7-transcript-1']
    (directory / 'chunk-index-v2.json').write_text(json.dumps(chunks))
    monkeypatch.setattr(cli, 'production_idle', lambda: pytest.fail('must validate coverage before model calls'))
    monkeypatch.setattr(cli, 'invoke_factory', lambda *_: pytest.fail('must not initialize model'))
    monkeypatch.setattr(sys, 'argv', ['fresh', '--index-dir', str(directory), '--output', str(tmp_path / 'out'), '--cutoff', '2026-10-26'])
    with pytest.raises(ValueError, match='(?i)chunk|coverage|missing'):
        cli.main()
