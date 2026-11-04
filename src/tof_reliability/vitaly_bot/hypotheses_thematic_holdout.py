"""Source-bound mechanics for an additional thematic holdout assessment.

Semantic judgments and an independent review remain necessary; validated quotes
and calculated coverage alone do not establish that a group is coherent.
"""
from __future__ import annotations

import json


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _evidence(evidence, source_texts):
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('source evidence required')
    for ev in evidence:
        if not isinstance(ev, dict):
            raise ValueError('invalid evidence')
        source = source_texts.get(ev.get('source_id'))
        start, end, fragment = ev.get('start'), ev.get('end'), ev.get('fragment')
        if (not isinstance(source, str) or type(start) is not int or type(end) is not int
                or not _text(fragment) or not 0 <= start < end <= len(source)
                or source[start:end] != fragment):
            raise ValueError('invalid source evidence')


def validate_partition(*, topics, groups, exclusions, source_texts):
    if not isinstance(topics, list) or not topics or not isinstance(source_texts, dict):
        raise ValueError('nonempty topics and sources required')
    if not isinstance(groups, list) or not groups or not isinstance(exclusions, list):
        raise ValueError('groups and exclusions required')
    by_id = {}
    for topic in topics:
        if not isinstance(topic, dict) or not _text(topic.get('id')) or topic['id'] in by_id:
            raise ValueError('unique topic id required')
        if not _text(topic.get('description')):
            raise ValueError('topic description required')
        _evidence(topic.get('evidence'), source_texts)
        by_id[topic['id']] = topic
    seen, group_ids = set(), set()
    for group in groups:
        if not isinstance(group, dict) or not _text(group.get('id')) or group['id'] in group_ids:
            raise ValueError('unique group id required')
        group_ids.add(group['id'])
        if not _text(group.get('description')):
            raise ValueError('group description required')
        children = group.get('child_ids')
        if not isinstance(children, list) or not children:
            raise ValueError('nonempty group required')
        for child in children:
            if not isinstance(child, str) or child not in by_id or child in seen:
                raise ValueError('unknown or duplicate child')
            seen.add(child)
        _evidence(group.get('evidence'), source_texts)
        child_evidence = [ev for child in children for ev in by_id[child]['evidence']]
        if any(ev not in child_evidence for ev in group['evidence']):
            raise ValueError('group evidence must come from its children')
    for exclusion in exclusions:
        if not isinstance(exclusion, dict) or not _text(exclusion.get('rationale')):
            raise ValueError('exclusion rationale required')
        child = exclusion.get('topic_id')
        if not isinstance(child, str) or child not in by_id or child in seen:
            raise ValueError('unknown or duplicate exclusion')
        seen.add(child)
    if seen != set(by_id):
        raise ValueError('every topic must appear exactly once')


def passes_threshold(matched, denominator):
    if (type(matched) is not int or type(denominator) is not int or denominator <= 0
            or not 0 <= matched <= denominator):
        raise ValueError('invalid score')
    return matched * 10 >= denominator * 7


def score_groups(*, topics, groups, exclusions, source_texts, predictions, mappings):
    validate_partition(topics=topics, groups=groups, exclusions=exclusions, source_texts=source_texts)
    if not isinstance(predictions, list) or len(predictions) < 7 or any(not _text(p) for p in predictions[:7]):
        raise ValueError('first seven predictions required')
    if not isinstance(mappings, list):
        raise ValueError('mappings required')
    group_by_id = {g['id']: g for g in groups}
    topic_by_id = {t['id']: t for t in topics}
    strict, broad = set(), set()
    for mapping in mappings:
        if not isinstance(mapping, dict):
            raise ValueError('invalid mapping')
        group_id, index = mapping.get('group_id'), mapping.get('prediction_index')
        if not isinstance(group_id, str) or group_id not in group_by_id or type(index) is not int or not 0 <= index < 7:
            raise ValueError('unknown group or prediction index')
        support = mapping.get('support')
        if support not in ('strong', 'partial', 'weak', 'unsupported') or not _text(mapping.get('rationale')):
            raise ValueError('support and rationale required')
        source_quote, prediction_quote = mapping.get('source_quote'), mapping.get('prediction_quote')
        child_evidence = [ev['fragment'] for child in group_by_id[group_id]['child_ids']
                          for ev in topic_by_id[child]['evidence']]
        if (not _text(source_quote) or not any(source_quote in fragment for fragment in child_evidence)
                or not _text(prediction_quote) or prediction_quote not in predictions[index]):
            raise ValueError('mapping quotes must be grounded in group evidence and prediction')
        if support == 'strong':
            strict.add(group_id)
            broad.add(group_id)
        elif support == 'partial':
            broad.add(group_id)
    denominator = len(groups)
    return {'denominator': denominator, 'strict_matched_count': len(strict),
            'broad_matched_count': len(broad), 'strict_coverage': len(strict) / denominator,
            'broad_coverage': len(broad) / denominator,
            'passing': passes_threshold(len(strict), denominator),
            'strict_matched_group_ids': [g['id'] for g in groups if g['id'] in strict],
            'broad_matched_group_ids': [g['id'] for g in groups if g['id'] in broad]}


def build_grouping_prompt(*, topics):
    # Explicit projection prevents unrelated fields from leaking into the prompt.
    clean = [{'id': t['id'], 'description': t['description'],
              'evidence': [{k: ev[k] for k in ('source_id', 'start', 'end', 'fragment')}
                           for ev in t['evidence']]}
             for t in topics]
    return ('Independently partition the analytical topics below into coherent main thematic groups '
            'by conceptual breadth. Preserve every topic ID exactly once, including any non-analytical '
            'exclusion with a reason. Give each group an id, description, child_ids, and source evidence '
            'copied from its children. Return JSON with groups and exclusions. Topics: '
            + json.dumps(clean, ensure_ascii=False))
