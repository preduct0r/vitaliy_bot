#!/usr/bin/env python3
"""Install or preview a local launchd agent without placing a token in Git."""

from __future__ import annotations

import argparse
import os
import plistlib
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict


LABEL = "ru.nornickel.tof.vitaly-bot"
LAUNCHD_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def build_plist(repository: Path, env_file: Path, logs_dir: Path) -> Dict[str, Any]:
    python = repository / ".venv" / "bin" / "python"
    return {
        "Label": LABEL,
        "ProgramArguments": [str(python), str(repository / "scripts" / "run_vitaly_bot.py")],
        "WorkingDirectory": str(repository),
        "EnvironmentVariables": {
            "VITALY_BOT_ENV_FILE": str(env_file),
            # launchd does not inherit the interactive shell's Homebrew PATH.
            "PATH": os.environ.get("PATH", LAUNCHD_PATH) or LAUNCHD_PATH,
            **({"VITALY_BOT_TOF_ROOT": os.environ["VITALY_BOT_TOF_ROOT"]} if os.environ.get("VITALY_BOT_TOF_ROOT") else {}),
        },
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "StandardOutPath": str(logs_dir / "vitaly-bot.out.log"),
        "StandardErrorPath": str(logs_dir / "vitaly-bot.err.log"),
        "ProcessType": "Background",
    }


def reload_launch_agent(
    target: Path, runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run
) -> int:
    """Reload an existing user agent so updated environment variables take effect.

    ``launchctl load`` leaves an already registered job untouched. Booting it
    out first replaces its launchd-side environment with the plist just written.
    The legacy fallback keeps installation recoverable if bootstrap rejects a
    recently booted-out job on a particular macOS version.
    """

    domain = "gui/" + str(os.getuid())
    runner(["launchctl", "bootout", domain, str(target)], check=False)
    bootstrap = runner(["launchctl", "bootstrap", domain, str(target)], check=False)
    if bootstrap.returncode == 0:
        return 0
    return runner(["launchctl", "load", "-w", str(target)], check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True, help="local config.env containing VITALY_BOT_TOKEN")
    parser.add_argument("--target", type=Path, default=Path.home() / "Library" / "LaunchAgents" / (LABEL + ".plist"))
    parser.add_argument("--logs-dir", type=Path, default=Path.home() / "Library" / "Logs" / "TOF-Vitaly-Bot")
    parser.add_argument("--load", action="store_true", help="load the agent after writing the plist")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    env_file = args.env_file.expanduser().resolve()
    if not env_file.exists() or "VITALY_BOT_TOKEN=" not in env_file.read_text(encoding="utf-8"):
        parser.error("--env-file must exist locally and contain VITALY_BOT_TOKEN=")
    repository = Path(__file__).resolve().parents[1]
    payload = build_plist(repository, env_file, args.logs_dir.expanduser())
    if args.dry_run:
        print(plistlib.dumps(payload).decode("utf-8"))
        return 0
    args.logs_dir.expanduser().mkdir(parents=True, exist_ok=True)
    args.target.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with args.target.expanduser().open("wb") as target:
        plistlib.dump(payload, target)
    print("installed=" + str(args.target.expanduser()))
    if args.load:
        return reload_launch_agent(args.target.expanduser())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
