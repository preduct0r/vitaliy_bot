"""Revalidate source-bound observations before an independent chronological grouping."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

from .hypotheses_interests import _date

TRAINING_CUTOFF = '2026-10-02'


def reconstruct_observations(*, sources, source_texts, observations):
    by_id = {}
    for source in sources:
        sid = source.get('id')
        if not isinstance(sid, str) or not sid or sid in by_id or _date(source.get('date')) >= TRAINING_CUTOFF:
            raise ValueError('invalid or non-training source')
        text = source_texts.get(sid)
        if not isinstance(text, str) or hashlib.sha256(text.encode('utf-8')).hexdigest() != source.get('sha256'):
            raise ValueError('source hash mismatch')
        by_id[sid] = text
    result = []
    for obs in observations:
        if not isinstance(obs, Mapping):
            raise ValueError('invalid observation')
        text = by_id.get(obs.get('source_id'))
        start, end, fragment = obs.get('start'), obs.get('end'), obs.get('fragment')
        if (not isinstance(text, str) or type(start) is not int or type(end) is not int
                or not isinstance(fragment, str) or not fragment or not 0 <= start < end <= len(text)
                or text[start:end] != fragment or not isinstance(obs.get('requirement'), str)
                or not obs['requirement'].strip()):
            raise ValueError('unverifiable observation')
        result.append(dict(obs))
    if not result:
        raise ValueError('observations required')
    return result


def build_training_map(*, sources, source_texts, observations, grouper):
    checked = reconstruct_observations(sources=sources, source_texts=source_texts, observations=observations)
    return grouper(checked)
