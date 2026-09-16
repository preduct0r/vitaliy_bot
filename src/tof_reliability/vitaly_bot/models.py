"""Small, Python-3.9-compatible value objects used by the Vitaly bot."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BotConfig:
    """Local-only configuration for the Mac-resident bot."""

    token: str
    state_dir: Path
    codex_bin: Path
    model: str = "gpt-6-astra"
    reasoning_effort: str = "xhigh"
    poll_timeout: int = 30
    model_timeout_seconds: int = 480
    evidence_timeout_seconds: int = 180
    cleanup_days: int = 30
    context_ttl_days: int = 4


@dataclass(frozen=True)
class ArtifactInput:
    """A file that has already been downloaded from Telegram."""

    telegram_file_id: str
    original_name: str
    path: Path
    size_bytes: int


@dataclass(frozen=True)
class PreparedArtifact:
    """Text and visual material prepared from one case artifact."""

    artifact_id: str
    name: str
    text: str
    text_reference: str
    image_paths: List[Path]
    unit_count: int
    warnings: List[str]


@dataclass(frozen=True)
class ModelResult:
    """A validated structured response from a non-interactive Codex run."""

    value: Dict[str, Any]
    stdout: str
    stderr: str


class VitalyBotError(RuntimeError):
    """Base error whose text is safe to show to a bot user."""


class InputError(VitalyBotError):
    """The uploaded material does not fit the public bot contract."""


class ModelError(VitalyBotError):
    """Codex could not produce a usable answer."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


class EvidenceError(VitalyBotError):
    """The current published TOF ledger cannot provide a safe evidence pack."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


class StateError(VitalyBotError):
    """The requested bot action is invalid for the current case state."""


JsonDict = Dict[str, Any]
JsonList = List[JsonDict]
OptionalText = Optional[str]
