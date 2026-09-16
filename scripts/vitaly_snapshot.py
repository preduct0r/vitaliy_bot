#!/usr/bin/env python3
"""Private recoverable snapshot; never writes to the live bot or its repository."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tarfile
from datetime import datetime, timezone


def snapshot(root, state, destination):
    destination.mkdir(parents=True, exist_ok=False)
    destination.chmod(0o700)
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    (destination / 'head.txt').write_bytes(git('rev-parse', 'HEAD'))
    (destination / 'working.patch').write_bytes(git('diff', '--binary', 'HEAD'))
    (destination / 'index.patch').write_bytes(git('diff', '--cached', '--binary'))
    files = git('ls-files', '-z', '--cached', '--others', '--exclude-standard').decode().split('\0')
    with tarfile.open(destination / 'working-tree.tar.gz', 'w:gz') as archive:
        for name in files:
            if name and (root / name).exists():
                archive.add(root / name, arcname=name, recursive=False)
    with sqlite3.connect(state.joinpath('vitaly_bot.sqlite3').as_uri()+'?mode=ro', uri=True) as source:
        with sqlite3.connect(destination / 'vitaly_bot.sqlite3') as target:
            source.backup(target)
            assert target.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    shutil.copy2(state / 'config.env', destination / 'config.env')
    shutil.copy2(Path.home() / 'Library/LaunchAgents/ru.nornickel.tof.vitaly-bot.plist', destination / 'launchd.plist')
    with tarfile.open(destination / 'cases.tar.gz', 'w:gz') as archive:
        archive.add(state / 'cases', arcname='cases')
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'root': str(root),
                'note': 'SQLite is transaction-consistent; case file copy is captured while bot stays live.',
                'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.iterdir() if p.is_file()}}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    for path in destination.iterdir():
        path.chmod(0o600)
    print(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    snapshot(args.root, args.state, args.destination)
