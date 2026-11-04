"""Synthetic contracts for private training extraction; no corpus data is read."""
import hashlib
import importlib

import pytest


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@pytest.fixture
def extraction():
    name = 'tof_reliability.vitaly_bot.hypotheses_interest_extraction'
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name != name:
            raise
        class MissingExtraction:
            def __getattr__(self, name):
                pytest.fail(f'training extraction API {name} is not implemented')
        return MissingExtraction()


@pytest.fixture
def corpus():
    training = 'Вступление. Проверить повторные ремонты. Заключение.'
    holdout = 'HOLDOUT_SECRET must never reach training.'
    sources = {'version': 'synthetic-v1', 'sources': [
        {'id': 'training', 'date': '2026-10-01', 'kind': 'transcript', 'sha256': digest(training)},
        {'id': 'holdout', 'date': '2026-10-26', 'kind': 'transcript', 'sha256': digest(holdout)},
    ]}
    cut = training.index(' Заключение')
    chunks = [
        {'source_id': 'training', 'start': 0, 'end': cut, 'sha256': digest(training[:cut])},
        {'source_id': 'training', 'start': cut, 'end': len(training), 'sha256': digest(training[cut:])},
        {'source_id': 'holdout', 'start': 0, 'end': len(holdout), 'sha256': digest(holdout)},
    ]
    return sources, chunks, training, holdout


def test_training_selection_excludes_holdout_before_context_construction(extraction, corpus):
    sources, chunks, training, holdout = corpus
    selected = extraction.select_training_chunks(sources, chunks, holdout_dates=['2026-10-26'])
    assert selected == chunks[:2]
    assert 'HOLDOUT_SECRET' not in str(selected)
    assert all(c['source_id'] != 'holdout' for c in selected)


def test_exact_unicode_excerpt_offsets_and_attribution_are_verified(extraction, corpus):
    _, chunks, training, _ = corpus
    start = training.index('Проверить')
    fragment = 'Проверить повторные ремонты'
    record = {'themes': [{'requirement': 'Анализ повторных ремонтов', 'evidence': [{
        'start': start, 'end': start + len(fragment), 'fragment': fragment,
        'attribution': 'interpretation', 'speaker': 'unknown',
    }]}]}
    result = extraction.parse_chunk_extraction(record, chunk=chunks[0], source_text=training)
    evidence = result['themes'][0]['evidence'][0]
    assert evidence['source_id'] == 'training'
    assert evidence['locator'] == f'chars:{start}-{start + len(fragment)}'
    assert evidence['fragment'] == fragment
    assert evidence['attribution'] == 'interpretation'
    assert evidence.get('speaker') != 'Vitaly'


@pytest.mark.parametrize('mutation', ['invented', 'shifted', 'outside', 'bad_attribution', 'wrong_hash'])
def test_unverifiable_evidence_is_rejected(extraction, corpus, mutation):
    _, chunks, text, _ = corpus
    start = text.index('Проверить')
    fragment = 'Проверить повторные ремонты'
    entry = {'start': start, 'end': start + len(fragment), 'fragment': fragment,
             'attribution': 'direct_request'}
    chunk = dict(chunks[0])
    if mutation == 'invented':
        entry['fragment'] = 'Несуществующая цитата'
    elif mutation == 'shifted':
        entry['start'] += 1
    elif mutation == 'outside':
        entry.update(start=chunks[1]['start'], end=len(text), fragment=text[chunks[1]['start']:])
    elif mutation == 'bad_attribution':
        entry['attribution'] = 'certain'
    else:
        chunk['sha256'] = '0' * 64
    response = {'themes': [{'requirement': 'Тема', 'evidence': [entry]}]}
    with pytest.raises(ValueError):
        extraction.parse_chunk_extraction(response, chunk=chunk, source_text=text)


def test_every_training_chunk_requires_successful_hash_bound_manifest(extraction, corpus):
    sources, chunks, _, _ = corpus
    selected = extraction.select_training_chunks(sources, chunks, holdout_dates=['2026-10-26'])
    manifests = [{**chunk, 'status': 'processed', 'attempts': 1} for chunk in selected]
    assert extraction.validate_training_coverage(selected, manifests) is None
    with pytest.raises(ValueError, match='(?i)missing|coverage|chunk'):
        extraction.validate_training_coverage(selected, manifests[:1])
    for replacement in ({**manifests[1], 'status': 'failed'},
                        {**manifests[1], 'sha256': '0' * 64},
                        {**manifests[1], 'attempts': 0}):
        with pytest.raises(ValueError):
            extraction.validate_training_coverage(selected, [manifests[0], replacement])
    with pytest.raises(ValueError):
        extraction.validate_training_coverage(selected, manifests + [manifests[0]])
