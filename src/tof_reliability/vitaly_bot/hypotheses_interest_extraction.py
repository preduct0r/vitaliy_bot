"""Verify training-only chunk selection, extraction evidence and coverage."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from .hypotheses_interests import ATTRIBUTIONS, HOLDOUT_DATES


def select_training_chunks(index: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]], *,
                           holdout_dates: Sequence[str] = HOLDOUT_DATES) -> list[dict]:
    sources = {s['id']: s for s in index['sources']}
    if len(sources) != len(index['sources']):
        raise ValueError('duplicate source id')
    dates = set(holdout_dates)
    return [dict(c) for c in chunks if c.get('source_id') in sources
            and sources[c['source_id']]['date'] not in dates]


def parse_chunk_extraction(record: Mapping[str, Any], *, chunk: Mapping[str, Any],
                           source_text: str) -> dict:
    start, end = chunk['start'], chunk['end']
    if (not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int)
            or isinstance(end, bool) or not 0 <= start < end <= len(source_text)
            or hashlib.sha256(source_text[start:end].encode('utf-8')).hexdigest() != chunk['sha256']):
        raise ValueError('invalid chunk bounds or hash')
    themes = record.get('themes')
    if not isinstance(themes, list):
        raise ValueError('themes must be a list')
    result = []
    for theme in themes:
        if not isinstance(theme, dict) or not isinstance(theme.get('requirement'), str) or not theme['requirement'].strip():
            raise ValueError('theme requirement is required')
        entries = theme.get('evidence')
        if not isinstance(entries, list) or not entries:
            raise ValueError('theme evidence is required')
        evidence = []
        for entry in entries:
            a, b, fragment = entry.get('start'), entry.get('end'), entry.get('fragment')
            if (not isinstance(a, int) or isinstance(a, bool) or not isinstance(b, int)
                    or isinstance(b, bool) or not start <= a < b <= end
                    or not isinstance(fragment, str) or not fragment
                    or source_text[a:b] != fragment or entry.get('attribution') not in ATTRIBUTIONS):
                raise ValueError('unverifiable chunk evidence')
            item = {'source_id': chunk['source_id'], 'locator': f'chars:{a}-{b}',
                    'fragment': fragment, 'attribution': entry['attribution']}
            if entry.get('speaker'):
                item['speaker'] = entry['speaker']
            evidence.append(item)
        result.append({'requirement': theme['requirement'].strip(), 'evidence': evidence})
    return {'themes': result}


def validate_training_coverage(chunks: Sequence[Mapping[str, Any]],
                               manifests: Sequence[Mapping[str, Any]]) -> None:
    def key(row: Mapping[str, Any]) -> tuple:
        return row['source_id'], row['start'], row['end'], row['sha256']
    expected = [key(row) for row in chunks]
    actual = [key(row) for row in manifests]
    if (len(expected) != len(set(expected)) or len(actual) != len(set(actual))
            or set(expected) != set(actual) or any(m.get('status') != 'processed'
            or not isinstance(m.get('attempts'), int) or isinstance(m.get('attempts'), bool)
            or m['attempts'] < 1 for m in manifests)):
        raise ValueError('missing, duplicate or invalid chunk coverage')
