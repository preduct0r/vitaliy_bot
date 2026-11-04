#!/usr/bin/env python3
"""Group verified training observations into a provenance-checked private map."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from scripts.hypotheses_extract_training import invoke_factory
from tof_reliability.vitaly_bot.hypotheses_extract_runner import _write, parse_trace
from tof_reliability.vitaly_bot.hypotheses_interest_extraction import (parse_chunk_extraction,
    select_training_chunks, validate_training_coverage)
from tof_reliability.vitaly_bot.hypotheses_interests import validate_interest_map, assess_provenance


def ground_attribution(evidence, source_text):
    """Do not ascribe a request to Vitaly without an explicit speaker label."""
    result = dict(evidence)
    if result['attribution'] != 'direct_request':
        return result
    start = int(result['locator'].split(':', 1)[1].split('-', 1)[0])
    context = source_text[max(0, start - 1000):start]
    labels = list(re.finditer(r'(?m)^\s*(?:\[[^\]\n]{1,30}\]\s*)?([^:\n]{1,70}):\s*', context))
    if not labels or not re.fullmatch(r'\s*(?i:виталий)(?:\s+[А-ЯЁ][а-яё.-]+){0,2}\s*', labels[-1].group(1)):
        result['reported_attribution'] = 'direct_request'
        result['attribution'] = 'interpretation'
        result['attribution_note'] = 'speaker not independently established; model label retained separately'
    return result


def group_observations(requirements, checkpoint, invoke, *, batch_size=100):
    if batch_size < 2 or not requirements:
        raise ValueError('nonempty requirements and batch size >= 2 required')
    digest = hashlib.sha256(json.dumps(requirements, ensure_ascii=False).encode()).hexdigest()
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state['digest'] != digest or state['batch_size'] != batch_size:
            raise ValueError('group checkpoint does not match observations')
    else:
        state = {'digest': digest, 'batch_size': batch_size, 'cursor': 0, 'next': [],
                 'current': [{'requirement': r, 'members': [i]} for i, r in enumerate(requirements)]}
        _write(checkpoint, state)
    while not state.get('complete'):
        current, cursor = state['current'], state['cursor']
        batch = current[cursor:cursor + batch_size]
        if len(batch) == 1:
            grouped = [{'requirement': batch[0]['requirement'], 'members': [0]}]
        else:
            grouped = invoke([entry['requirement'] for entry in batch])
        if not isinstance(grouped, list) or not grouped or any(
            not isinstance(g, dict) or not isinstance(g.get('requirement'), str)
            or not g['requirement'].strip() or not isinstance(g.get('members'), list)
            or not g['members'] or any(type(i) is not int for i in g['members']) for g in grouped
        ):
            raise ValueError('invalid grouping structure')
        members = [i for group in grouped for i in group['members']]
        if sorted(members) != list(range(len(batch))):
            raise ValueError('grouping omitted or duplicated observations')
        state['next'].extend({'requirement': group['requirement'],
                              'members': [original for i in group['members'] for original in batch[i]['members']]}
                             for group in grouped)
        state['cursor'] += len(batch)
        if state['cursor'] == len(current):
            state['complete'] = len(current) <= batch_size or len(state['next']) >= len(current)
            state['current'], state['next'], state['cursor'] = state['next'], [], 0
        _write(checkpoint, state)
    return state['current']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--chunks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    index = json.loads(args.index.read_text())
    chunks = json.loads(args.chunks.read_text())
    state = json.loads((args.output / 'extraction-checkpoint.json').read_text())
    training = select_training_chunks(index, chunks)
    validate_training_coverage(training, state['manifest'])
    ids = {row['source_id'] for row in training}
    sources = [s for s in index['sources'] if s['id'] in ids]
    byid = {s['id']: s for s in sources}
    texts = {s['id']: Path(s['path']).read_text(encoding='utf-8') for s in sources}
    observations = []
    for i, row in enumerate(state['manifest']):
        parsed = parse_chunk_extraction(state['results'][str(i)], chunk=row, source_text=texts[row['source_id']])
        observations.extend(parsed['themes'])
    if not observations:
        raise ValueError('no observations to group')
    # Only opaque observation numbers and claims are sent to the grouper.
    invoke = invoke_factory(args.output, sources)
    def group_batch(requirements):
        request = [{'n': i, 'requirement': r} for i, r in enumerate(requirements)]
        prompt = ('Сгруппируй связанные аналитические потребности в предметные темы. '
                  'Близкие задачи объединяй в общую тему; различия исходных наблюдений сохраняются отдельно. '
                  'Не смешивай несвязанные темы и не делай чрезмерно широких обобщений. '
                  'Верни только JSON {"groups":[{"requirement":"точное обобщение", "members":[номера]}]}. '
                  'Ответ должен быть единственным сообщением без вступлений и промежуточных сообщений. '
                  'Каждый номер должен встретиться ровно один раз. Не делай выводов о частоте. Данные: '
                  + json.dumps(request, ensure_ascii=False))
        try:
            return parse_trace(invoke(prompt))['groups']
        except Exception as exc:
            _write(args.output / 'group-error.json', {'type': type(exc).__name__, 'message': str(exc)})
            raise
    groups = group_observations([o['requirement'] for o in observations],
                                args.output / 'group-checkpoint.json', group_batch)
    if sorted(i for group in groups for i in group['members']) != list(range(len(observations))):
        raise ValueError('group checkpoint omitted or duplicated observations')
    interests = []
    for number, group in enumerate(groups, 1):
        details = [{'requirement': observations[i]['requirement'],
                    'evidence': [ground_attribution(e, texts[e['source_id']])
                                 for e in observations[i]['evidence']]}
                   for i in group['members']]
        evidence = [e for detail in details for e in detail['evidence']]
        meeting_dates = {byid[e['source_id']]['date'] for e in evidence
                         if byid[e['source_id']]['kind'] == 'transcript'}
        interests.append({'id': f'interest-{number:03}', 'requirement': group['requirement'],
                          'evidence': evidence, 'observations': details,
                          'meeting_count_lower_bound': len(meeting_dates),
                          'support': 'recurring' if len(meeting_dates) >= 2 else 'isolated',
                          'attributions': sorted({e['attribution'] for e in evidence})})
    training_index = {'version': index['version'], 'sources': sources}
    profile = {'version': 'training-interest-map-v1', 'corpus_version': index['version'], 'interests': interests}
    validate_interest_map(profile, training_index)
    provenance = assess_provenance(profile, training_index)
    if not provenance['verified']:
        _write(args.output / 'provenance-failed.json', provenance)
        raise ValueError('provenance verification failed')
    _write(args.output / 'training-interest-map.json', profile)
    _write(args.output / 'training-provenance.json', provenance)
    print(json.dumps({'interests': len(interests), 'observations': len(observations), 'verified': True}))


if __name__ == '__main__':
    main()
