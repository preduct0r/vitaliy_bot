"""Synthetic contracts for the dated final corpus, separate from historical evaluations."""
import hashlib
import json
import sys
from datetime import date

import pytest

from scripts import hypotheses_fresh_training as cli
from tof_reliability.vitaly_bot import hypotheses_fresh_training as fresh
from tof_reliability.vitaly_bot.hypotheses_interest_extraction import parse_chunk_extraction


FINAL = date(2026, 11, 7)  # Exclusive boundary for a 2026-11-06 snapshot.
OCTOBER = date(2026, 10, 26)


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@pytest.fixture
def inventory(tmp_path):
    directory = tmp_path / 'index'
    directory.mkdir()
    sources, reconciled, chunks, texts = [], [], [], {}
    for i, day in enumerate(('2026-09-16', '2026-10-25', '2026-10-26',
                             '2026-10-28', '2026-10-30', '2026-11-06', '2026-11-07')):
        sid = f'source-{i}'
        text = f'synthetic meeting {i}: a unique observation'
        path = directory / f'{sid}.txt'
        path.write_text(text, encoding='utf-8')
        source = {'id': sid, 'date': day, 'meeting_id': f'meeting-{i}',
                  'kind': 'baseline' if i % 2 == 0 else 'transcript',
                  'path': str(path), 'sha256': sha(text)}
        sources.append(source)
        reconciled.append({**source, 'status': 'accepted', 'original_path': str(path),
                           'original_sha256': sha(text)})
        chunks.append({'id': f'chunk-{i}', 'source_id': sid, 'start': 0,
                       'end': len(text), 'sha256': sha(text)})
        texts[sid] = text
    def save():
        for name, data in [('source-index.json', {'sources': sources}),
                           ('chunk-index-v2.json', chunks),
                           ('reconciled-sources-v2.json', reconciled)]:
            (directory / name).write_text(json.dumps(data), encoding='utf-8')
    save()
    return directory, sources, reconciled, chunks, texts, save


def run_cli(monkeypatch, directory, output, prompts):
    monkeypatch.setattr(cli, 'production_idle', lambda: None)
    monkeypatch.setattr(cli, 'invoke_factory', lambda *_: lambda prompt: prompts.append(prompt) or 'trace')
    monkeypatch.setattr(cli, 'parse_trace', lambda _: {'themes': []})
    monkeypatch.setattr(sys, 'argv', ['fresh', '--index-dir', str(directory), '--output',
                                     str(output), '--cutoff', FINAL.isoformat()])
    cli.main()


def test_final_cutoff_includes_all_october_dates_but_keeps_historical_boundary(inventory):
    _, sources, _, chunks, texts, _ = inventory
    def ids(cutoff):
        return {sid for batch in fresh.prepare_fresh_batches(index={'sources': sources},
                    chunks=chunks, source_texts=texts, batch_size=1, cutoff=cutoff)
                for sid in batch['source_ids']}
    assert ids(OCTOBER) == {'source-0', 'source-1'}
    assert ids(FINAL) == {f'source-{i}' for i in range(6)}


def test_final_cli_reconciles_all_accepted_and_preserves_input_files(inventory, tmp_path, monkeypatch):
    directory, sources, _, chunks, texts, _ = inventory
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    prompts, output = [], tmp_path / 'private-output'
    run_cli(monkeypatch, directory, output, prompts)
    result = json.loads((output / 'fresh-extraction.json').read_text())
    coverage = result['coverage']
    assert {row['source_id'] for row in coverage} == {s['id'] for s in sources[:6]}
    assert {row['chunk_id'] for row in coverage} == {c['id'] for c in chunks[:6]}
    assert len(prompts) == 6
    assert all(texts['source-6'] not in prompt for prompt in prompts)
    manifest = json.loads((output / 'coverage-manifest.json').read_text())
    assert manifest['cutoff'] == FINAL.isoformat()
    assert manifest['expected'] == manifest['found'] == {'sources': 6, 'chunks': 6}
    assert not manifest['missing']['sources'] and not manifest['missing']['chunks']
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


@pytest.mark.parametrize('damage', ['missing_index_entry', 'missing_chunk', 'bad_chunk_hash',
                                    'unavailable_file', 'bad_original_hash'])
def test_final_inventory_fails_closed_before_model_setup(inventory, tmp_path, monkeypatch, damage):
    directory, sources, reconciled, chunks, _, save = inventory
    if damage == 'missing_index_entry':
        sources.pop(4)
    elif damage == 'missing_chunk':
        chunks.pop(4)
    elif damage == 'bad_chunk_hash':
        chunks[4]['sha256'] = '0' * 64
    elif damage == 'unavailable_file':
        from pathlib import Path
        Path(sources[4]['path']).unlink()
    else:
        reconciled[4]['original_sha256'] = '0' * 64
    save()
    monkeypatch.setattr(cli, 'production_idle', lambda: pytest.fail('model queue checked before validation'))
    monkeypatch.setattr(cli, 'invoke_factory', lambda *_: pytest.fail('model setup before validation'))
    monkeypatch.setattr(sys, 'argv', ['fresh', '--index-dir', str(directory), '--output',
                                     str(tmp_path / 'out'), '--cutoff', FINAL.isoformat()])
    with pytest.raises((ValueError, OSError)):
        cli.main()


def test_same_content_distinct_meetings_are_not_deduplicated(inventory, tmp_path, monkeypatch):
    directory, sources, reconciled, chunks, texts, save = inventory
    # Same text can occur at two independently identified meetings; only the
    # explicitly linked copy of the first meeting is a duplicate.
    first, other = sources[0], sources[1]
    from pathlib import Path
    Path(other['path']).write_text(texts[first['id']], encoding='utf-8')
    other['sha256'] = first['sha256']
    reconciled[1].update(sha256=first['sha256'], original_sha256=first['sha256'])
    chunks[1].update(end=len(texts[first['id']]), sha256=first['sha256'])
    reconciled.append({**reconciled[0], 'id': 'same-meeting-copy', 'duplicate_of': first['id']})
    save()
    prompts, output = [], tmp_path / 'out'
    run_cli(monkeypatch, directory, output, prompts)
    result = json.loads((output / 'fresh-extraction.json').read_text())
    assert {row['source_id'] for row in result['coverage']} == {s['id'] for s in sources[:6]}
    assert len(prompts) == 6


@pytest.mark.parametrize('damage', ['source_hash', 'prompt_hash', 'trace'])
def test_checkpoint_tampering_never_reuses_a_result(inventory, tmp_path, damage):
    _, sources, _, chunks, texts, _ = inventory
    path = tmp_path / 'checkpoint.json'
    kwargs = dict(index={'sources': sources[:1]}, chunks=chunks[:1], source_texts=texts,
                  checkpoint_path=path, batch_size=1, cutoff=OCTOBER,
                  check_queues=lambda: {'production_user_idle': True, 'learning_queue_idle': True},
                  validate_trace=lambda trace: json.loads(trace))
    fresh.run_fresh_extraction(**kwargs, invoke=lambda _: {'trace': '{"themes": []}', 'themes': []})
    state = json.loads(path.read_text())
    if damage == 'source_hash':
        state['batches'][0]['source_hashes'][sources[0]['id']] = '0' * 64
    elif damage == 'prompt_hash':
        state['batches'][0]['prompt_sha256'] = '0' * 64
    else:
        state['results']['0']['trace'] = '{"themes": [{"label": "forged"}]}'
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        fresh.run_fresh_extraction(**kwargs, invoke=lambda _: pytest.fail('invalid checkpoint reused'))
    assert path.read_bytes() == before


@pytest.mark.parametrize('attribution', ['direct_request', 'interpretation', 'third_party_paraphrase'])
def test_exact_source_references_and_attribution_are_retained(attribution):
    text = 'Начало. Проверить затраты. Конец.'
    fragment = 'Проверить затраты'
    start = text.index(fragment)
    chunk = {'source_id': 'synthetic', 'start': 0, 'end': len(text), 'sha256': sha(text)}
    entry = {'start': start, 'end': start + len(fragment), 'fragment': fragment,
             'attribution': attribution}
    def parse(evidence):
        return parse_chunk_extraction({'themes': [{'requirement': 'need', 'evidence': [evidence]}]},
                                      chunk=chunk, source_text=text)
    result = parse(entry)['themes'][0]['evidence'][0]
    assert result['source_id'] == 'synthetic'
    assert result['locator'] == f'chars:{start}-{start + len(fragment)}'
    assert result['fragment'] == fragment and result['attribution'] == attribution
    with pytest.raises(ValueError):
        parse({**entry, 'start': start + 1})
    with pytest.raises(ValueError):
        parse({**entry, 'attribution': 'invented'})

@pytest.mark.parametrize('damage', ['missing', 'changed'])
def test_duplicate_file_is_verified_before_reuse(inventory, tmp_path, monkeypatch, damage):
    directory, sources, reconciled, _, texts, save = inventory
    duplicate_path = directory / 'duplicate.txt'
    duplicate_path.write_text(texts[sources[0]['id']], encoding='utf-8')
    reconciled.append({**reconciled[0], 'id': 'duplicate-source',
                       'path': str(duplicate_path), 'original_path': str(duplicate_path),
                       'duplicate_of': sources[0]['id']})
    save()
    if damage == 'missing':
        duplicate_path.unlink()
    else:
        duplicate_path.write_text('different content', encoding='utf-8')
    monkeypatch.setattr(cli, 'production_idle', lambda: pytest.fail('model queue checked before validation'))
    monkeypatch.setattr(cli, 'invoke_factory', lambda *_: pytest.fail('model setup before validation'))
    monkeypatch.setattr(sys, 'argv', ['fresh', '--index-dir', str(directory), '--output',
                                     str(tmp_path / 'out'), '--cutoff', FINAL.isoformat()])
    with pytest.raises((ValueError, OSError)):
        cli.main()

@pytest.mark.parametrize('day', ['2026-10-26', '2026-10-28', '2026-10-30'])
def test_dimension_projection_requires_explicit_bound_final_snapshot(tmp_path, day):
    from tof_reliability.vitaly_bot.hypotheses_topic_projection import project_semantic_dimensions
    path = tmp_path / 'source.txt'
    path.write_text('synthetic source', encoding='utf-8')
    index = {'version': 'final-snapshot', 'snapshot_cutoff': FINAL.isoformat(), 'sources': [
        {'id': 'source', 'path': str(path), 'sha256': sha('synthetic source'), 'date': day}]}
    profile = {'version': 'final-profile', 'corpus_version': index['version'], 'interests': []}
    classifier = lambda _: []
    verifier = lambda _: {}
    with pytest.raises(ValueError):
        project_semantic_dimensions(profile, index, classifier=classifier, verifier=verifier)
    result = project_semantic_dimensions(profile, index, classifier=classifier, verifier=verifier,
                                         final_cutoff=FINAL.isoformat())
    assert result['corpus_version'] == index['version']
    index['snapshot_cutoff'] = '2026-10-26'
    with pytest.raises(ValueError):
        project_semantic_dimensions(profile, index, classifier=classifier, verifier=verifier,
                                    final_cutoff=FINAL.isoformat())
    index['snapshot_cutoff'] = FINAL.isoformat()
    index['sources'][0]['date'] = FINAL.isoformat()
    with pytest.raises(ValueError):
        project_semantic_dimensions(profile, index, classifier=classifier, verifier=verifier,
                                    final_cutoff=FINAL.isoformat())


def test_final_generator_broad_patterns_require_two_independent_meetings():
    from tof_reliability.vitaly_bot.hypotheses_topic_projection import (
        DIMENSION_VERSION, dimensions_for_generator,
    )
    projection = {'version': DIMENSION_VERSION, 'dimensions': [
        {'dimension_id': 'repair_techcard_equipment_operations_coverage',
         'topic_id': 'techcard_coverage_verification', 'interest_ids': ['interest-a'],
         'source_ids': ['source-a'], 'independent_meeting_count': 1},
        {'dimension_id': 'repair_labor_norm_actual',
         'topic_id': 'labor_norm_observed_measurement', 'interest_ids': ['interest-b'],
         'source_ids': ['source-b', 'source-c'], 'independent_meeting_count': 2},
    ]}
    result = dimensions_for_generator(projection, min_independent_meetings=2)
    assert [r['dimension_id'] for r in result['dimensions']] == ['repair_labor_norm_actual']
    del projection['dimensions'][0]['independent_meeting_count']
    with pytest.raises(ValueError):
        dimensions_for_generator(projection, min_independent_meetings=2)


def test_dimension_projection_preserves_chronological_boundary_after_historical_dates(tmp_path):
    from tof_reliability.vitaly_bot.hypotheses_topic_projection import project_semantic_dimensions
    path = tmp_path / 'source.txt'
    path.write_text('synthetic source', encoding='utf-8')
    index = {'version': 'historical', 'sources': [
        {'id': 'source', 'path': str(path), 'sha256': sha('synthetic source'), 'date': '2026-11-01'}]}
    profile = {'version': 'historical-profile', 'corpus_version': 'historical', 'interests': []}
    calls = []
    with pytest.raises(ValueError):
        project_semantic_dimensions(profile, index, classifier=lambda _: calls.append('classifier'),
                                    verifier=lambda _: calls.append('verifier'),
                                    development_cutoff=OCTOBER.isoformat())
    assert calls == []
