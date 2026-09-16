#!/usr/bin/env python3
"""Run the local Telegram avatar of Vitaly on the current Mac."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    asyncio.run(run_bot(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
