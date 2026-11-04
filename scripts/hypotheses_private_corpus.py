#!/usr/bin/env python3
"""Assess a manually curated private inventory, training map and holdout topics."""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from tof_reliability.vitaly_bot.hypotheses_private_corpus import build_and_assess_private_corpus


_ALLOWED_ITEMS = {'agent_message', 'reasoning'}


def _check_production_idle():
    """Reject model children of the production poller, not the idle poller itself."""
    lines = subprocess.run(['ps', '-ax', '-o', 'pid=,ppid=,command='],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    processes = {}
    for line in lines:
        match = re.match(r'^\s*(\d+)\s+(\d+)\s+(.+)$', line)
        if match:
            pid, parent, command = match.groups()
            processes[int(pid)] = (int(parent), command)
    bots = {pid for pid, (_, command) in processes.items()
            if re.search(r'(?:^|/)python(?:\d+(?:\.\d+)?)?\s+\S*/scripts/run_vitaly_bot\.py(?:\s|$)', command)}
    for pid, (_, command) in processes.items():
        parent = processes[pid][0]
        ancestors = set()
        while parent in processes and parent not in ancestors:
            if parent in bots and re.search(r'(?:^|/)codex(?:\s|$)', command):
                raise RuntimeError('active production model job is running')
            ancestors.add(parent)
            parent = processes[parent][0]


def _model(request, output):
    _check_production_idle()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    prompt = ('Return only a JSON object with a predictions array of at least seven concise topic labels. '
              'Use only the following training context.\n' + json.dumps(request, ensure_ascii=False))
    args = ['codex', 'exec', '--json', '--ephemeral', '--ignore-user-config', '--skip-git-repo-check',
            '--sandbox', 'read-only', '-m', 'gpt-6.1-sol', '-c', 'model_reasoning_effort="high"',
            '-c', 'skills.bundled.enabled=false', '-c', 'skills.include_instructions=false',
            '-c', 'mcp_servers={}', '-c', 'project_doc_max_bytes=0']
    for feature in ('shell_tool', 'unified_exec', 'apps', 'plugins', 'memories', 'multi_agent'):
        args.extend(['--disable', feature])
    args.append('-')
    result = subprocess.run(args, input=prompt, capture_output=True, text=True, cwd=str(output), check=False)
    # Save the trace before validation, including rejected runs; private, never in Git.
    trace = output / 'model-trace.jsonl'
    fd = os.open(trace, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        stream.write(result.stdout)
    if result.returncode:
        raise RuntimeError('model invocation failed: ' + result.stderr[-1000:])
    message, completed = None, False
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError as exc:
            raise RuntimeError('invalid model trace') from exc
        event_type = event.get('type', '')
        if event_type.startswith('item.'):
            item = event.get('item', {})
            if item.get('type') not in _ALLOWED_ITEMS:
                raise RuntimeError('tool or unknown item in model trace; evaluation discarded')
            if event_type == 'item.completed' and item.get('type') == 'agent_message':
                message = item.get('text')
        elif event_type == 'turn.completed':
            completed = True
        elif event_type not in {'thread.started', 'turn.started'}:
            raise RuntimeError('unexpected event in model trace; evaluation discarded')
    if not completed or not isinstance(message, str):
        raise RuntimeError('incomplete model trace')
    try:
        return json.loads(message)
    except ValueError as exc:
        raise RuntimeError('model did not return JSON') from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'training-map', 'holdout-topics', 'output-dir', 'corpus-version'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def model(request):
        return _model(request, Path(args.output_dir))

    result = build_and_assess_private_corpus(sources=read(args.sources),
        training_map=read(args.training_map), holdout_topics=read(args.holdout_topics),
        output_dir=args.output_dir, corpus_version=args.corpus_version, model=model)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result['status'] == 'assessed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
