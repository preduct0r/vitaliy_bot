"""Red-first contracts for complete, private pre-cutoff training coverage.

All source text and model responses in these tests are synthetic.
"""
from __future__ import annotations

import hashlib
import json
import sys

import pytest

from scripts import hypotheses_fresh_training as cli
from tof_reliability.vitaly_bot import hypotheses_fresh_training as fresh


def sha(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def corpus(tmp_path):
    index_dir = tmp_path / 'indexes'
    index_dir.mkdir()
    sources, reconciled, chunks, texts = [], [], [], {}

    def add(sid, kind, status, when, text, parts=1, *, utf16=False, indexed=True):
        converted = index_dir / (sid + '.txt')
        converted.write_text(text, encoding='utf-8')
        original = index_dir / (sid + '.original')
        original.write_bytes(text.encode('utf-16') if utf16 else text.encode('utf-8'))
        row = {'id': sid, 'kind': kind, 'date': when, 'path': str(converted), 'sha256': sha(text)}
        reconciled.append({**row, 'status': status, 'original_path': str(original),
                           'original_sha256': sha(original.read_bytes())})
        if indexed:
            sources.append(row)
            texts[sid] = text
            for i in range(parts):
                start, end = len(text) * i // parts, len(text) * (i + 1) // parts
                chunks.append({'source_id': sid, 'start': start, 'end': end,
                               'sha256': sha(text[start:end])})
        return row

    # The same 12+8 source and 62 chunk shape as the canonical eligible inventory.
    for i in range(20):
        add(f's{i:02}', 'baseline' if i < 12 else 'transcript', 'accepted',
            '2026-09-01', f'unique source {i:02} synthetic material',
            4 if i < 2 else 3, utf16=i == 0)
    add('excluded', 'baseline', 'excluded', '2026-09-01', 'EXCLUDED CANARY')
    add('different', 'baseline', 'different-person', '2026-09-01', 'OTHER PERSON CANARY')
    add('disputed', 'transcript', 'disputed', '2026-09-01', 'DISPUTED CANARY')
    add('late', 'transcript', 'accepted', '2026-10-26', 'HOLDOUT CANARY')
    # Duplicate content under another identity must not produce a second extraction.
    add('duplicate', 'baseline', 'accepted', '2026-09-01', texts['s00'], indexed=False)
    (index_dir / 'source-index.json').write_text(json.dumps({'sources': sources}))
    (index_dir / 'chunk-index-v2.json').write_text(json.dumps(chunks))
    (index_dir / 'reconciled-sources-v2.json').write_text(json.dumps(reconciled))
    return index_dir, sources, reconciled, chunks, texts


def test_batch_selection_includes_baseline_and_every_expected_chunk(corpus):
    _, sources, _, chunks, texts = corpus
    # The caller supplies the reconciled accepted inventory; the batch builder
    # must not silently throw away its baseline records.
    accepted = [s for s in sources if s['id'].startswith('s')]
    selected = [c for c in chunks if c['source_id'].startswith('s')]
    batches = fresh.prepare_fresh_batches(index={'sources': accepted}, chunks=selected,
                                          source_texts=texts, batch_size=1)
    assert {sid for b in batches for sid in b['source_ids']} == {s['id'] for s in accepted}
    assert len(batches) == len(selected) == 62
    assert sum(len(b['chunks']) for b in batches) == 62


def test_cli_reconciles_full_inventory_and_emits_coverage(corpus, tmp_path, monkeypatch):
    index_dir, sources, _, chunks, _ = corpus
    output = tmp_path / 'private-output'
    prompts = []
    monkeypatch.setattr(cli, 'production_idle', lambda: None)
    monkeypatch.setattr(cli, 'invoke_factory', lambda *_: lambda prompt: prompts.append(prompt) or 'trace')
    monkeypatch.setattr(cli, 'parse_trace', lambda _: {'themes': []})
    monkeypatch.setattr(sys, 'argv', ['fresh', '--index-dir', str(index_dir), '--output', str(output)])
    cli.main()
    result = json.loads((output / 'fresh-extraction.json').read_text())
    coverage = result['coverage']
    assert {r['source_id'] for r in coverage} == {s['id'] for s in sources if s['id'].startswith('s')}
    assert len(coverage) == len(prompts) == 62
    assert all(canary not in prompt for prompt in prompts for canary in
               ('EXCLUDED CANARY', 'OTHER PERSON CANARY', 'DISPUTED CANARY', 'HOLDOUT CANARY'))
    provenance = json.loads((output / 'source-provenance.json').read_text())
    converted = next(r for r in provenance if r['source_id'] == 's00')
    original = next(r for r in json.loads((index_dir / 'reconciled-sources-v2.json').read_text()) if r['id'] == 's00')
    assert converted['original_sha256'] == original['original_sha256']
    assert converted['original_path'] == original['original_path']
    manifest = json.loads((output / 'coverage-manifest.json').read_text())
    assert manifest['expected']['sources'] == manifest['found']['sources'] == 20
    assert manifest['expected']['chunks'] == manifest['found']['chunks'] == 62
    assert not manifest['missing']['sources'] and not manifest['missing']['chunks']


@pytest.mark.parametrize('damage', ['missing_source_chunks', 'chunk_hash', 'original_hash', 'converted_origin'])
def test_cli_fails_closed_before_invocation_on_missing_or_bad_evidence(corpus, tmp_path, monkeypatch, damage):
    index_dir, _, reconciled, chunks, _ = corpus
    if damage == 'missing_source_chunks':
        chunks = [c for c in chunks if c['source_id'] != 's00']
        (index_dir / 'chunk-index-v2.json').write_text(json.dumps(chunks))
    elif damage == 'chunk_hash':
        next(c for c in chunks if c['source_id'] == 's00')['sha256'] = '0' * 64
        (index_dir / 'chunk-index-v2.json').write_text(json.dumps(chunks))
    elif damage == 'original_hash':
        next(r for r in reconciled if r['id'] == 's00')['original_sha256'] = '0' * 64
        (index_dir / 'reconciled-sources-v2.json').write_text(json.dumps(reconciled))
    else:
        # Both individual hashes remain valid, but the UTF-16 original decodes
        # to different text than the indexed UTF-8 conversion.
        row = next(r for r in reconciled if r['id'] == 's00')
        data = 'unrelated source text'.encode('utf-16')
        from pathlib import Path
        Path(row['original_path']).write_bytes(data)
        row['original_sha256'] = sha(data)
        (index_dir / 'reconciled-sources-v2.json').write_text(json.dumps(reconciled))
    monkeypatch.setattr(cli, 'production_idle', lambda: pytest.fail('must validate before queue check'))
    monkeypatch.setattr(cli, 'invoke_factory', lambda *_: lambda _: pytest.fail('must not invoke model'))
    monkeypatch.setattr(sys, 'argv', ['fresh', '--index-dir', str(index_dir), '--output', str(tmp_path / 'out')])
    with pytest.raises((ValueError, FileNotFoundError), match='(?i)source|chunk|hash|integrity|origin|convert|missing'):
        cli.main()
    manifest = json.loads((tmp_path / 'out' / 'coverage-manifest.json').read_text())
    assert manifest['missing']['sources'] or manifest['missing']['chunks']


@pytest.mark.parametrize('field', ['prompt_sha256', 'source_hashes', 'chunk_sha256'])
def test_resume_rejects_altered_checkpoint_hashes_before_model(corpus, tmp_path, field):
    _, sources, _, chunks, texts = corpus
    selected_sources = [s for s in sources if s['id'] == 's00']
    selected_chunks = [c for c in chunks if c['source_id'] == 's00']
    path = tmp_path / 'checkpoint.json'
    kwargs = dict(index={'sources': selected_sources}, chunks=selected_chunks, source_texts=texts,
                  checkpoint_path=path, batch_size=1,
                  check_queues=lambda: {'production_user_idle': True, 'learning_queue_idle': True})
    fresh.run_fresh_extraction(**kwargs, invoke=lambda _: {'trace': 'synthetic', 'themes': []})
    state = json.loads(path.read_text())
    if field == 'prompt_sha256':
        state['batches'][0]['prompt_sha256'] = '0' * 64
    elif field == 'source_hashes':
        state['batches'][0]['source_hashes']['s00'] = '0' * 64
    else:
        state['batches'][0]['chunks'][0]['sha256'] = '0' * 64
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='(?i)checkpoint|provenance|hash|integrity'):
        fresh.run_fresh_extraction(**kwargs, invoke=lambda _: pytest.fail('must not reuse or invoke'))
