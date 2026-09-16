#!/usr/bin/env python3
"""Run the local Telegram avatar of Vitaly on the current Mac."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import fcntl
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tof_reliability.vitaly_bot.config import load_config
from tof_reliability.vitaly_bot.storage import VitalyStore
from tof_reliability.vitaly_bot.telegram import run_bot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-auth", action="store_true", help="only check the saved ChatGPT Codex login")
    parser.add_argument("--cleanup-only", action="store_true", help="remove expired local case files and exit")
    args = parser.parse_args()
    config = load_config(require_token=not (args.check_auth or args.cleanup_only))
    if args.check_auth:
        result = subprocess.run([str(config.codex_bin), "login", "status"], check=False)
        return result.returncode
    if args.cleanup_only:
        removed = VitalyStore(config.state_dir).cleanup_expired_files(config.cleanup_days)
        print("removed_cases=" + str(len(removed)))
        return 0
    lock_dir = Path.home() / "Library" / "Application Support" / "TOF Vitaly Poller Locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / (hashlib.sha256(config.token.encode()).hexdigest() + ".lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Этот Telegram-токен уже обслуживается другим polling-процессом")
        asyncio.run(run_bot(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
