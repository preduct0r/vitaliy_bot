"""Bounded material hints and a read-only lookup on the current TOF server."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence

from .models import BotConfig, EvidenceError
from .server_evidence import EVIDENCE_VERSION, MAX_CANDIDATES
from .server_evidence import build_evidence_pack as _server_build_evidence_pack


ROOT = Path(__file__).resolve().parents[3]
SERVER_PYTHON = "/srv/tof/repos/tof_v1/.venv/bin/python"


class EvidenceProvider(Protocol):
    async def build(self, material_text: str, artifact_hashes: Sequence[str]) -> Dict[str, Any]:
        ...


def _date_identifier(value: str) -> bool:
    if len(value) != 8 or not value.startswith(("19", "20")):
        return False
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        return False
    return True


def extract_hints(material_text: str) -> Dict[str, List[str]]:
    """Extract exact identifiers without treating prose or filenames as IDs."""

    explicit: List[str] = []
    patterns = (
        r"(?i)\b(?:заказ|order(?:_id)?)\s*[№#:=]?\s*([A-ZА-Я0-9][A-ZА-Я0-9_-]{3,})",
        r"(?i)\b(?:ео|equipment(?:[_ ]id)?|оборудовани[ея])\s*[№#:=]?\s*([A-ZА-Я0-9][A-ZА-Я0-9_-]{3,})",
        r"(?i)\b(?:мтр|material(?:[_ ]id)?)\s*[№#:=]?\s*([A-ZА-Я0-9][A-ZА-Я0-9_-]{3,})",
    )
    for pattern in patterns:
        explicit.extend(
            match.group(1).strip()
            for match in re.finditer(pattern, material_text)
            if any(character.isdigit() for character in match.group(1))
        )
    bare_numbers = [
        match.group(1)
        for match in re.finditer(r"(?<!\w)(\d{6,12})(?!\w)", material_text)
        if not _date_identifier(match.group(1))
    ]
    candidate_ids = list(dict.fromkeys([*explicit, *bare_numbers]))[:MAX_CANDIDATES]
    periods = list(
        dict.fromkeys(re.findall(r"\b20\d{2}(?:[-./](?:0[1-9]|1[0-2]))?\b", material_text))
    )[:MAX_CANDIDATES]
    numbers = list(
        dict.fromkeys(re.findall(r"(?<!\w)\d{1,3}(?:[ \u00a0]\d{3})*(?:[.,]\d+)?\s*(?:%|₽|руб\.?|ч(?:ас(?:а|ов)?)?)?", material_text))
    )[:80]
    claims = [" ".join(value.split()) for value in re.split(r"[\n.!?]+", material_text) if len(value.strip()) >= 12]
    return {
        "candidate_ids": candidate_ids,
        "periods": periods,
        "numbers": numbers,
        "claims": list(dict.fromkeys(claims))[:80],
    }


def build_evidence_pack(
    request: Dict[str, Any],
    *,
    generation_state_path: Path = Path("/srv/tof/state/current_generation.json"),
    root: Path = ROOT,
) -> Dict[str, Any]:
    """Local test entry point for the same fixed server-side lookup."""

    return _server_build_evidence_pack(request, generation_state_path=generation_state_path, root=root)


def _remote_program(request: Dict[str, Any]) -> bytes:
    """Send code and data over stdin; no user material enters the SSH command."""

    source = Path(__file__).with_name("server_evidence.py").read_bytes()
    encoded = base64.b64encode(json.dumps(request, ensure_ascii=False).encode("utf-8")).decode("ascii")
    invocation = (
        "\nimport base64 as _b64, json as _json\n"
        "print(_json.dumps(build_evidence_pack(_json.loads(_b64.b64decode("
        + repr(encoded)
        + "))), ensure_ascii=False, sort_keys=True))\n"
    )
    return source + invocation.encode("utf-8")


class RemoteLedgerEvidenceProvider:
    """Ask the TOF server to query its current published ledger."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config

    async def build(self, material_text: str, artifact_hashes: Sequence[str]) -> Dict[str, Any]:
        request = {
            "version": EVIDENCE_VERSION,
            "artifact_hashes": list(artifact_hashes),
            **extract_hints(material_text),
        }
        try:
            program = _remote_program(request)
        except OSError as error:
            raise EvidenceError("tof_evidence_invalid", str(error)) from error
        command = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            os.environ.get("TOF_SSH_HOST", "tof"),
            SERVER_PYTHON + " -",
        ]
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(program), timeout=self.config.evidence_timeout_seconds
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
                raise
        except asyncio.TimeoutError as error:
            raise EvidenceError("tof_unavailable", "timeout") from error
        except OSError as error:
            raise EvidenceError("tof_unavailable", str(error)) from error
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise EvidenceError("tof_unavailable", (stderr + "\n" + stdout)[-4000:])
        for line in reversed(stdout.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("evidence_version") == EVIDENCE_VERSION:
                return value
        raise EvidenceError("tof_evidence_invalid", (stderr + "\n" + stdout)[-4000:])
