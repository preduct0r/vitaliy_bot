from pathlib import Path

from tof_reliability.vitaly_bot.config import load_config
from tof_reliability.vitaly_bot.models import BotConfig


def _environment(tmp_path: Path, **overrides: str) -> dict[str, str]:
    return {
        "VITALY_BOT_TOKEN": "test-token",
        "VITALY_BOT_STATE_DIR": str(tmp_path / "state"),
        "VITALY_BOT_ENV_FILE": str(tmp_path / "missing.env"),
        "VITALY_BOT_CODEX_BIN": str(tmp_path / "codex"),
        **overrides,
    }


def test_bot_config_defaults_to_hypotheses_model(tmp_path: Path):
    config = BotConfig(
        token="test-token",
        state_dir=tmp_path / "state",
        codex_bin=tmp_path / "codex",
    )

    assert config.model == "gpt-6.1-sol"
    assert config.reasoning_effort == "high"


def test_load_config_defaults_to_hypotheses_model(tmp_path: Path):
    config = load_config(_environment(tmp_path))

    assert config.model == "gpt-6.1-sol"
    assert config.reasoning_effort == "high"


def test_load_config_honors_explicit_model_environment_overrides(tmp_path: Path):
    config = load_config(
        _environment(
            tmp_path,
            VITALY_BOT_MODEL="gpt-6-luna",
            VITALY_BOT_REASONING_EFFORT="xhigh",
        )
    )

    assert config.model == "gpt-6-luna"
    assert config.reasoning_effort == "xhigh"
