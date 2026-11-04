"""Deterministic validation for a source-bound, independently mapped holdout."""
from __future__ import annotations

import json


def build_generator_prompt(*, training_map, provenance_verified):
    if provenance_verified is not True:
        raise ValueError('training provenance must be verified')
    interests = training_map.get('interests')
    if not isinstance(interests, list) or not interests:
        raise ValueError('validated training interests required')
    requirements = []
    for interest in interests:
        requirement = interest.get('requirement')
        if not isinstance(requirement, str) or not requirement.strip():
            raise ValueError('invalid training requirement')
        requirements.append(requirement)
    return ('На основе только перечисленных аналитических интересов сформулируй семь конкретных '
            'аналитических гипотез или вопросов, которые могли бы заинтересовать руководителя. '
            'Укажи предмет, сравнение или результат, избегай общих лозунгов. '
            'Верни только JSON {"predictions":["...", ...]} с ровно семью элементами. '
            'Интересы: ' + json.dumps(requirements, ensure_ascii=False))


def validate_generator_trace(trace):
    message = None
    completed = False
    for line in trace.splitlines():
        event = json.loads(line)
        kind = event.get('type')
        if kind in ('thread.started', 'turn.started', 'turn.completed'):
            completed |= kind == 'turn.completed'
        elif kind in ('item.started', 'item.updated', 'item.completed'):
            item = event.get('item', {})
            typ = item.get('type')
            # A startup diagnostic from the disabled host is not a tool call.
            if typ == 'error' and 'Code Mode is unavailable because code-mode host is disabled' in item.get('message', ''):
                continue
            if typ not in ('agent_message', 'reasoning'):
                raise ValueError('tool event violates trace isolation')
            if kind == 'item.completed' and typ == 'agent_message':
                if message is not None:
                    raise ValueError('multiple messages in isolated trace')
                message = item.get('text')
        else:
            raise ValueError('unknown event violates trace isolation')
    if not completed or not isinstance(message, str):
        raise ValueError('incomplete isolated trace')
    if message.startswith('```'):
        message = message.split('\n', 1)[1].rsplit('```', 1)[0]
    result = json.loads(message)
    predictions = result.get('predictions')
    if not isinstance(predictions, list) or len(predictions) < 7 or any(
            not isinstance(p, str) or not p.strip() for p in predictions[:7]):
        raise ValueError('trace requires first seven predictions')
    return result


def score_holdout(*, topics, predictions, mappings, source_texts):
    if not isinstance(topics, list) or not topics:
        raise ValueError('nonempty expected topics required')
    if len(predictions) < 7 or any(not isinstance(p, str) or not p.strip() for p in predictions[:7]):
        raise ValueError('first seven predictions required')
    ids = []
    for topic in topics:
        topic_id = topic.get('id')
        if not isinstance(topic_id, str) or not topic_id or topic_id in ids:
            raise ValueError('topic ids must be unique')
        if not isinstance(topic.get('description'), str) or not topic['description'].strip():
            raise ValueError('topic description required')
        ids.append(topic_id)
        evidence = topic.get('evidence')
        if not isinstance(evidence, list) or not evidence:
            raise ValueError('source evidence required for every topic')
        for ev in evidence:
            source = source_texts.get(ev.get('source_id'))
            start, end, fragment = ev.get('start'), ev.get('end'), ev.get('fragment')
            if (not isinstance(source, str) or type(start) is not int or type(end) is not int
                or not isinstance(fragment, str) or not fragment or not 0 <= start < end <= len(source)
                or source[start:end] != fragment):
                raise ValueError('invalid source evidence fragment or position')
    matched = set()
    for mapping in mappings:
        index = mapping.get('prediction_index')
        topic_id = mapping.get('topic_id')
        if topic_id not in ids or type(index) is not int or not 0 <= index < 7:
            raise ValueError('unknown topic or prediction index outside first seven')
        rationale = mapping.get('rationale')
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError('mapping rationale required')
        matched.add(topic_id)
    hits = [topic_id for topic_id in ids if topic_id in matched]
    return {'denominator': len(ids), 'matched_count': len(hits),
            'coverage': len(hits) / len(ids), 'passing': len(hits) * 10 >= len(ids) * 7,
            'matched_topic_ids': hits, 'missed_topic_ids': [i for i in ids if i not in matched]}
