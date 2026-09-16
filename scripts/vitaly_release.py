#!/usr/bin/env python3
"""Quiescent launchd switch with rollback; never clears Telegram updates."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import signal
import sqlite3
import subprocess
import time

LABEL = 'ru.nornickel.tof.vitaly-bot'


def idle(state):
    with sqlite3.connect((state / 'vitaly_bot.sqlite3').as_uri()+'?mode=ro', uri=True) as db:
        jobs = db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
        cases = db.execute("SELECT COUNT(*) FROM cases WHERE activity <> 'idle' OR mode IN ('grilling','stopping')").fetchone()[0]
    return not jobs and not cases


def pid():
    r = subprocess.run(['launchctl', 'print', f'gui/{os.getuid()}/{LABEL}'], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.strip().startswith('pid = '):
            return int(line.split('=')[1])
    return None


def switch(plist_path, candidate, state, backup):
    if not idle(state):
        raise RuntimeError('Бот занят: выпуск отложен, старый процесс продолжает работать.')
    before = plist_path.read_bytes()
    old_pid = pid()
    if not old_pid:
        raise RuntimeError('Исходный процесс не найден; автоматическое переключение отменено.')
    os.kill(old_pid, signal.SIGSTOP)
    try:
        if not idle(state):
            raise RuntimeError('Обнаружена новая работа: переключение отменено.')
        backup.mkdir(parents=True, exist_ok=False)
        backup.chmod(0o700)
        (backup / 'launchd.plist').write_bytes(before)
        with sqlite3.connect((state / 'vitaly_bot.sqlite3').as_uri()+'?mode=ro', uri=True) as src:
            with sqlite3.connect(backup / 'vitaly_bot.sqlite3') as dst:
                src.backup(dst)
                assert dst.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        # Deliver graceful stop before resuming the loop; no second poller exists.
        os.kill(old_pid, signal.SIGTERM)
    finally:
        os.kill(old_pid, signal.SIGCONT)
    domain = f'gui/{os.getuid()}'
    for _ in range(50):
        if pid() is None:
            break
        time.sleep(.1)
    if pid() is not None:
        raise RuntimeError('Старый процесс ещё завершает работу; новая версия не запущена.')
    subprocess.run(['launchctl', 'bootout', domain, str(plist_path)], check=False, capture_output=True)
    plist_path.write_bytes(candidate.read_bytes())
    try:
        subprocess.run(['launchctl', 'bootstrap', domain, str(plist_path)], check=True)
        for _ in range(20):
            if pid():
                break
            time.sleep(.25)
        started = pid()
        time.sleep(2)
        if not started or pid() != started:
            raise RuntimeError('Новая версия не удержала процесс')
        with sqlite3.connect((state / 'vitaly_bot.sqlite3').as_uri()+'?mode=ro', uri=True) as db:
            assert db.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        print(json.dumps({'running_pid': started, 'backup': str(backup)}))
    except BaseException:
        subprocess.run(['launchctl', 'bootout', domain, str(plist_path)], check=False, capture_output=True)
        plist_path.write_bytes(before)
        subprocess.run(['launchctl', 'bootstrap', domain, str(plist_path)], check=True)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-plist', type=Path, required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--backup', type=Path, required=True)
    p.add_argument('--plist', type=Path, default=Path.home() / 'Library/LaunchAgents' / (LABEL+'.plist'))
    a = p.parse_args()
    switch(a.plist, a.candidate_plist, a.state, a.backup)
