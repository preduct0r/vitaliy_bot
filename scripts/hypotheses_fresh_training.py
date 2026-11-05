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
from tof_reliability.vitaly_bot.hypotheses_fresh_training import CUTOFF, run_fresh_extraction
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
    originals = {row['id']: row for row in reconciled if row.get('status') == 'accepted'}
    training = [row for row in index['sources'] if row['kind'] == 'transcript'
                and date.fromisoformat(row['date']) < CUTOFF]
    texts, provenance = {}, []
    for row in training:
        original = originals.get(row['id'])
        if not original or original['sha256'] != row['sha256']:
            raise ValueError('verified source index mismatch: ' + row['id'])
        data = Path(row['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']:
            raise ValueError('source hash mismatch: ' + row['id'])
        original_path = Path(original.get('original_path', row['path']))
        original_hash = hashlib.sha256(original_path.read_bytes()).hexdigest()
        if original_hash != original.get('original_sha256', row['sha256']):
            raise ValueError('original source hash mismatch: ' + row['id'])
        texts[row['id']] = data.decode('utf-8')
        provenance.append({'source_id': row['id'], 'source_sha256': row['sha256'],
                           'original_sha256': original_hash, 'original_path': str(original_path)})
    ids = set(texts)
    selected = [chunk for chunk in chunks if chunk['source_id'] in ids]
    if ids != {chunk['source_id'] for chunk in selected}:
        raise ValueError('training source missing from chunk index')
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
    _write(args.output / 'fresh-extraction.json', result)
    print(json.dumps({'processed_chunks': len(result['coverage']), 'evidence': len(result['evidence'])}))


if __name__ == '__main__':
    main()
