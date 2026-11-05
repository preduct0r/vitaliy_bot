#!/usr/bin/env python3
"""Run fresh training extraction privately against the verified corpus indexes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from scripts.hypotheses_extract_training import invoke_factory, production_idle
from tof_reliability.vitaly_bot.hypotheses_extract_runner import _write, parse_trace
from tof_reliability.vitaly_bot.hypotheses_fresh_training import CUTOFF, prepare_fresh_batches, run_fresh_extraction
from datetime import date


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.output, 0o700)
    check = subprocess.run(['git', '-C', str(args.output), 'rev-parse', '--is-inside-work-tree'],
                           capture_output=True, text=True, check=False)
    if check.returncode == 0 and check.stdout.strip() == 'true':
        raise ValueError('private output must be outside Git')
    index = json.loads((args.index_dir / 'source-index.json').read_text())
    chunks = json.loads((args.index_dir / 'chunk-index-v2.json').read_text())
    reconciled = json.loads((args.index_dir / 'reconciled-sources-v2.json').read_text())
    indexed = {}
    for row in index['sources']:
        if row['id'] in indexed:
            raise ValueError('duplicate source identity in index')
        indexed[row['id']] = row
    groups, seen_ids = {}, set()
    for row in reconciled:
        sid = row['id']
        if sid in seen_ids:
            raise ValueError('duplicate source identity in reconciled inventory')
        seen_ids.add(sid)
        if (row.get('status') != 'accepted' or row.get('kind') not in ('baseline', 'transcript')
                or date.fromisoformat(row['date']) >= CUTOFF):
            continue
        digest = row.get('sha256')
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError('missing source hash: ' + sid)
        groups.setdefault(digest, []).append(row)
    training, originals, missing_sources = [], {}, []
    for digest, group in groups.items():
        matches = [row for row in group if row['id'] in indexed and not row.get('duplicate_of')]
        if len(matches) != 1:
            missing_sources.append([row['id'] for row in group])
            continue
        original = matches[0]
        row = indexed[original['id']]
        if any(row.get(key) != original.get(key) for key in ('sha256', 'date', 'kind')):
            raise ValueError('verified source index mismatch: ' + row['id'])
        training.append(row)
        originals[row['id']] = original
    ids = {row['id'] for row in training}
    selected = [chunk for chunk in chunks if chunk['source_id'] in ids]
    manifest = {'expected': {'sources': len(groups), 'chunks': len(selected)},
                'found': {'sources': 0, 'chunks': 0},
                'missing': {'sources': missing_sources + sorted(ids),
                            'chunks': [c.get('id') or f"{c['source_id']}:{c['start']}:{c['end']}:{c['sha256']}"
                                       for c in selected]},
                'source_ids': sorted(ids),
                'chunk_ids': [c.get('id') or f"{c['source_id']}:{c['start']}:{c['end']}:{c['sha256']}"
                              for c in selected]}
    def save_manifest():
        _write(args.output / 'coverage-manifest.json', manifest)
    save_manifest()
    if missing_sources:
        raise ValueError('missing or duplicate source in index')
    texts, provenance = {}, []
    for row in training:
        original = originals.get(row['id'])
        if not original or original['sha256'] != row['sha256']:
            raise ValueError('verified source index mismatch: ' + row['id'])
        data = Path(row['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('source hash mismatch: ' + row['id'])
        original_path = Path(original.get('original_path', row['path']))
        original_data = original_path.read_bytes()
        original_hash = hashlib.sha256(original_data).hexdigest()
        if original_hash != original.get('original_sha256', row['sha256']):
            raise ValueError('original source hash mismatch: ' + row['id'])
        texts[row['id']] = data.decode('utf-8')
        encoding = original.get('original_encoding', 'utf-16' if original_data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8')
        if original_data.decode(encoding) != texts[row['id']]:
            raise ValueError('converted source origin mismatch: ' + row['id'])
        provenance.append({'source_id': row['id'], 'source_sha256': row['sha256'],
                           'original_sha256': original_hash, 'original_path': str(original_path),
                           'original_encoding': encoding})
        manifest['missing']['sources'].remove(row['id'])
        manifest['found']['sources'] += 1
        save_manifest()
    gaps = []
    for sid in ids:
        spans = sorted((c['start'], c['end']) for c in selected if c['source_id'] == sid)
        cursor = 0
        for start, end in spans:
            if start != cursor or end <= start:
                gaps.append({'source_id': sid, 'offset': cursor})
                break
            cursor = end
        if cursor != len(texts[sid]):
            gaps.append({'source_id': sid, 'offset': cursor})
    if gaps:
        manifest['missing']['chunks'].extend(gaps)
        save_manifest()
        raise ValueError('source chunk coverage missing or overlapping')
    prepare_fresh_batches(index={'sources': training}, chunks=selected, source_texts=texts, batch_size=1)
    manifest['missing']['chunks'] = []
    manifest['found'] = {'sources': len(training), 'chunks': len(selected)}
    save_manifest()
    _write(args.output / 'source-provenance.json', provenance)
    invoke_raw = invoke_factory(args.output, training)
    def queues():
        production_idle()
        return {'production_user_idle': True, 'learning_queue_idle': True}
    def invoke(prompt):
        trace = invoke_raw(prompt)
        return {'trace': trace, 'themes': parse_trace(trace)['themes']}
    result = run_fresh_extraction(index={'sources': training}, chunks=selected, source_texts=texts,
                                  checkpoint_path=args.output / 'fresh-checkpoint.json', batch_size=1,
                                  check_queues=queues, invoke=invoke, validate_trace=parse_trace)
    manifest['processed_chunks'] = len(result['coverage'])
    save_manifest()
    _write(args.output / 'fresh-extraction.json', result)
    print(json.dumps({'processed_chunks': len(result['coverage']), 'evidence': len(result['evidence'])}))


if __name__ == '__main__':
    main()
