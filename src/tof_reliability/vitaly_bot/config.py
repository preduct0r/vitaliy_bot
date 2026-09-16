"""Read the local-only Mac configuration without ever committing a token."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from .codex import codex_binary_from_environment
from .models import BotConfig, InputError


DEFAULT_STATE_DIR = Path.home() / "Library" / "Application Support" / "TOF Vitaly Bot"
DEFAULT_ENV_FILE = DEFAULT_STATE_DIR / "config.env"


def _read_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise InputError("Некорректная строка " + str(number) + " в локальном config.env")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise InputError("Пустое имя переменной в локальном config.env")
        values[key] = value.strip().strip("\"'")
    return values


def load_config(environ: Optional[Dict[str, str]] = None, require_token: bool = True) -> BotConfig:
    """Load environment variables, preferring exported values over config.env."""

    environment = dict(os.environ if environ is None else environ)
    env_file = Path(environment.get("VITALY_BOT_ENV_FILE", str(DEFAULT_ENV_FILE))).expanduser()
    from_file = _read_env_file(env_file)
    values = dict(from_file)
    values.update({key: value for key, value in environment.items() if key.startswith("VITALY_BOT_")})
    token = values.get("VITALY_BOT_TOKEN", "").strip()
    if not token and require_token:
        raise InputError(
            "Не задан VITALY_BOT_TOKEN. Создайте локальный " + str(env_file) + " вне Git."
        )
    state_dir = Path(values.get("VITALY_BOT_STATE_DIR", str(DEFAULT_STATE_DIR))).expanduser()
    return BotConfig(
        token=token,
        state_dir=state_dir,
        codex_bin=codex_binary_from_environment(values.get("VITALY_BOT_CODEX_BIN")),
        model=values.get("VITALY_BOT_MODEL", "gpt-6-astra"),
        reasoning_effort=values.get("VITALY_BOT_REASONING_EFFORT", "xhigh"),
        poll_timeout=int(values.get("VITALY_BOT_POLL_TIMEOUT", "30")),
        model_timeout_seconds=int(values.get("VITALY_BOT_MODEL_TIMEOUT_SECONDS", "480")),
        evidence_timeout_seconds=int(values.get("VITALY_BOT_EVIDENCE_TIMEOUT_SECONDS", "180")),
        cleanup_days=int(values.get("VITALY_BOT_CLEANUP_DAYS", "30")),
        context_ttl_days=int(values.get("VITALY_BOT_CONTEXT_TTL_DAYS", "4")),
    )
