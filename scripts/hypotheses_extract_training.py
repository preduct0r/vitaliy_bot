#!/usr/bin/env python3
"""Run private extraction; all inputs, traces and checkpoints stay outside Git."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

from scripts.hypotheses_private_corpus import _check_production_idle
from tof_reliability.vitaly_bot.config import load_config
from tof_reliability.vitaly_bot.hypotheses_extract_runner import _write, run_batches
from tof_reliability.vitaly_bot.hypotheses_interest_extraction import select_training_chunks


def production_idle():
    config = load_config(require_token=False)
    service = subprocess.run(['launchctl', 'print', f'gui/{os.getuid()}/ru.nornickel.tof.vitaly-bot'],
                             capture_output=True, text=True, check=True)
    if 'state = running' not in service.stdout:
        raise RuntimeError('production bot is not running')
    dbpath = config.state_dir / 'vitaly_bot.sqlite3'
    with sqlite3.connect(dbpath.as_uri() + '?mode=ro', uri=True) as db:
        for table in ('jobs', 'learning_jobs'):
            if db.execute(f"SELECT COUNT(*) FROM {table} WHERE status IN ('queued','running')").fetchone()[0]:
                raise RuntimeError(f'production {table} queue is active')
    _check_production_idle()


def invoke_factory(output, sources):
    def invoke(prompt):
        for source in sources:
            if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
                raise ValueError('source hash changed: ' + source['id'])
        while True:
            try:
                production_idle()
                break
            except RuntimeError as exc:
                _write(output / 'admission.json', {'status': 'waiting', 'reason': str(exc)})
                time.sleep(20)
        n = len(list(output.glob('trace-*.jsonl'))) + 1
        args = [str(load_config(require_token=False).codex_bin), 'exec', '--json', '--ephemeral',
                '--ignore-user-config', '--skip-git-repo-check', '--sandbox', 'read-only',
                '-m', 'gpt-6.1-sol', '-c', 'model_reasoning_effort="high"',
                '-c', 'skills.bundled.enabled=false', '-c', 'skills.include_instructions=false',
                '-c', 'mcp_servers={}', '-c', 'project_doc_max_bytes=0', '-c', 'web_search="disabled"']
        for feature in ('shell_tool', 'unified_exec', 'apps', 'plugins', 'memories', 'multi_agent',
                        'code_mode_host', 'computer_use', 'browser_use', 'view_image'):
            args += ['--disable', feature]
        args.append('-')
        try:
            result = subprocess.run(args, input=prompt, capture_output=True, text=True,
                                    cwd=output, timeout=1200, check=False)
        except subprocess.TimeoutExpired as exc:
            for suffix, data in (('jsonl', exc.stdout), ('stderr', exc.stderr)):
                raw = data if isinstance(data, bytes) else (data or '').encode('utf-8')
                path = output / f'trace-{n:04}.{suffix}'
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(raw)
            _write(output / f'diagnostic-{n:04}.json', {'error': 'timeout', 'seconds': exc.timeout})
            raise RuntimeError('model invocation timed out') from exc
        for suffix, data in (('jsonl', result.stdout), ('stderr', result.stderr)):
            path = output / f'trace-{n:04}.{suffix}'
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write(data)
        if result.returncode:
            raise RuntimeError(f'model exit {result.returncode}; see trace-{n:04}.stderr')
        return result.stdout
    return invoke


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--chunks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--previous-checkpoint', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=3)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    git_check = subprocess.run(['git', '-C', str(args.output), 'rev-parse', '--is-inside-work-tree'],
                               capture_output=True, text=True, check=False)
    if git_check.returncode == 0 and git_check.stdout.strip() == 'true':
        raise ValueError('private output must be outside a Git working tree')
    os.chmod(args.output, 0o700)
    index = json.loads(args.index.read_text())
    chunks = json.loads(args.chunks.read_text())
    training = select_training_chunks(index, chunks)
    previous = json.loads(args.previous_checkpoint.read_text())
    keys = ('source_id', 'start', 'end', 'sha256')
    if previous['corpus_version'] != index['version'] or [tuple(c[k] for k in keys) for c in previous['chunks']] != [tuple(c[k] for k in keys) for c in training]:
        raise ValueError('previous checkpoint does not match training index')
    ids = {c['source_id'] for c in training}
    sources = [s for s in index['sources'] if s['id'] in ids]
    texts = {}
    for source in sources:
        data = Path(source['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != source['sha256']:
            raise ValueError('source hash mismatch: ' + source['id'])
        texts[source['id']] = data.decode('utf-8')
    invoke = invoke_factory(args.output, sources)
    for attempt in range(3):
        try:
            result = run_batches(index=index, chunks=chunks, source_texts=texts,
                                 checkpoint_path=args.output / 'extraction-checkpoint.json',
                                 batch_size=args.batch_size, invoke=invoke)
            print(json.dumps({'processed': len(result['manifest'])}))
            return
        except (ValueError, KeyError, RuntimeError) as exc:
            _write(args.output / 'last-run-error.json', {'attempt': attempt + 1,
                                                         'type': type(exc).__name__, 'message': str(exc)})
            if attempt == 2:
                raise
            time.sleep(10)


if __name__ == '__main__':
    main()
