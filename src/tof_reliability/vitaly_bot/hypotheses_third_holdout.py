"""Fail-closed provenance and scoring primitives for an additional trial."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from .hypotheses_interests import chronological_holdout


def filter_training_chunks(*, index, chunks, holdout_dates):
    split = chronological_holdout(index, dates=holdout_dates)
    sources = index['sources']
    by_id = {s['id']: s for s in sources}
    if len(by_id) != len(sources):
        raise ValueError('duplicate source provenance')
    training_ids = {s['id'] for s in split['training']}
    seen = set()
    tainted = set()
    for chunk in chunks:
        sid, context, cid = chunk.get('source_id'), chunk.get('context_id'), chunk.get('id')
        if sid not in by_id or not isinstance(context, str) or not context.strip():
            raise ValueError('missing source or context provenance')
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError('missing or duplicate chunk provenance')
        seen.add(cid)
        if sid not in training_ids:
            tainted.add(context)
    training, exclusions = [], []
    for chunk in chunks:
        sid, context = chunk['source_id'], chunk['context_id']
        if sid not in training_ids or context in tainted:
            reason = ('source is at or after chronological cutoff' if sid not in training_ids
                      else 'context shared with source at or after chronological cutoff')
            exclusions.append({'chunk_id': chunk['id'], 'source_id': sid,
                               'context_id': context, 'reason': reason})
        else:
            training.append(dict(chunk))
    return {'training': training, 'exclusions': exclusions}


def build_grouping_prompt(*, observations):
    # Explicit allowlist: never transmit arbitrary fields from prior stages.
    allowed = ('id', 'description', 'source_id', 'start', 'end', 'fragment', 'sha256')
    safe = []
    for obs in observations:
        item = {key: obs[key] for key in allowed if key in obs}
        if 'evidence' in obs:
            item['evidence'] = [{key: ev[key] for key in allowed if key in ev}
                                for ev in obs['evidence']]
        safe.append(item)
    return ('Group the observed themes and classify each as analytical or organizational, '
            'giving a reason and retaining source evidence. Account for every observation. '
            'Observations: ' + json.dumps(safe, ensure_ascii=False))


def score_third_holdout(*, observed_group_ids, groups, source_texts, predictions, mappings):
    if not isinstance(predictions, list) or not 5 <= len(predictions) <= 7 or any(
            not isinstance(p, str) or not p.strip() for p in predictions):
        raise ValueError('five to seven nonempty prediction cards required')
    if not isinstance(observed_group_ids, list) or not observed_group_ids or len(set(observed_group_ids)) != len(observed_group_ids):
        raise ValueError('unique observed groups required')
    by_id = {}
    for group in groups:
        ident = group.get('id')
        if not isinstance(ident, str) or not ident or ident in by_id:
            raise ValueError('invalid or duplicate group')
        if group.get('disposition') not in ('analytical', 'organizational') or any(
                not isinstance(group.get(k), str) or not group[k].strip() for k in ('reason', 'description')):
            raise ValueError('group disposition and reason required')
        evidence = group.get('evidence')
        if not isinstance(evidence, list) or not evidence:
            raise ValueError('group evidence required')
        for ev in evidence:
            text = source_texts.get(ev.get('source_id'))
            start, end, fragment = ev.get('start'), ev.get('end'), ev.get('fragment')
            if (not isinstance(text, str) or type(start) is not int or type(end) is not int
                    or not isinstance(fragment, str) or not fragment or not 0 <= start < end <= len(text)
                    or text[start:end] != fragment
                    or hashlib.sha256(text.encode('utf-8')).hexdigest() != ev.get('sha256')):
                raise ValueError('invalid source evidence, offset or hash')
        by_id[ident] = group
    if set(by_id) != set(observed_group_ids):
        raise ValueError('every observed group needs a disposition')
    relevant = [ident for ident in observed_group_ids if by_id[ident]['disposition'] == 'analytical']
    if not relevant:
        raise ValueError('analytical groups required')
    strong, broad = set(), set()
    for mapping in mappings:
        ident, idx = mapping.get('group_id'), mapping.get('prediction_index')
        if ident not in relevant or type(idx) is not int or not 0 <= idx < len(predictions):
            raise ValueError('invalid mapping group or prediction')
        if mapping.get('support') not in ('strong', 'partial') or not isinstance(mapping.get('rationale'), str) or not mapping['rationale'].strip():
            raise ValueError('mapping support and rationale required')
        sq, pq = mapping.get('source_quote'), mapping.get('prediction_quote')
        if (not isinstance(sq, str) or not sq or not any(sq in ev['fragment'] for ev in by_id[ident]['evidence'])
                or not isinstance(pq, str) or not pq or pq not in predictions[idx]):
            raise ValueError('mapping quotes must be grounded in evidence and prediction')
        broad.add(ident)
        if mapping['support'] == 'strong':
            strong.add(ident)
    n = len(relevant)
    return {'denominator': n, 'strict_matched_count': len(strong), 'broad_matched_count': len(broad),
            'strict_coverage': len(strong) / n, 'broad_coverage': len(broad) / n,
            'strict_matched_group_ids': [i for i in relevant if i in strong],
            'broad_matched_group_ids': [i for i in relevant if i in broad],
            'passing': len(strong) * 10 >= n * 7, 'original_gate_met': False}
