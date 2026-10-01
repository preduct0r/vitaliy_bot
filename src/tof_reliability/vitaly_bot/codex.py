"""Adapter for a saved ChatGPT login exposed by the local Codex CLI."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .models import BotConfig, ModelError, ModelResult


TRANSIENT_CODES = {"model_timeout", "codex_network", "codex_transient"}
PROCESS_STOP_GRACE_SECONDS = 5


def _classify_failure(detail: str) -> str:
    lowered = detail.casefold()
    if "not logged" in lowered or "login" in lowered or "authentication" in lowered:
        return "codex_not_logged_in"
    if "rate limit" in lowered or "usage limit" in lowered or "quota" in lowered:
        return "subscription_limit_reached"
    if "timed out" in lowered or "timeout" in lowered:
        return "model_timeout"
    if "network" in lowered or "connection" in lowered or "temporarily" in lowered:
        return "codex_network"
    return "codex_transient"


async def _stop_process_group(process: asyncio.subprocess.Process) -> None:
    """Stop and reap a timed-out Codex CLI process and anything it spawned."""

    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=PROCESS_STOP_GRACE_SECONDS)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()


async def _reap_communicate(task: "asyncio.Task[tuple[bytes, bytes]]") -> None:
    """Drain stdio after killing a process so its temporary directory is safe to remove."""

    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


class CodexRunner:
    """Run an isolated, schema-constrained non-interactive Codex invocation."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config

    def command(
        self,
        work_dir: Path,
        schema_path: Path,
        output_path: Path,
        image_paths: Sequence[Path],
    ) -> List[str]:
        command = [
            str(self.config.codex_bin),
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-rules",
            "--sandbox",
            "read-only",
            "--cd",
            str(work_dir),
            "--model",
            self.config.model,
            "--config",
            'model_reasoning_effort="' + self.config.reasoning_effort + '"',
            "--output-schema",
            str(schema_path),
            "--output-last-message",
            str(output_path),
            "--json",
        ]
        for image_path in image_paths:
            command.extend(["--image", str(image_path)])
        command.append("-")
        return command

    async def run(
        self,
        prompt: str,
        schema: Dict[str, Any],
        image_paths: Optional[Sequence[Path]] = None,
    ) -> ModelResult:
        if not self.config.codex_bin.exists() and shutil.which(str(self.config.codex_bin)) is None:
            raise ModelError("codex_not_found", str(self.config.codex_bin))
        images = [path for path in image_paths or [] if path.exists()]
        with tempfile.TemporaryDirectory(prefix="tof-vitaly-codex-") as temporary:
            work_dir = Path(temporary)
            schema_path = work_dir / "response.schema.json"
            output_path = work_dir / "last-message.json"
            schema_path.write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
            command = self.command(work_dir, schema_path, output_path, images)
            process: Optional[asyncio.subprocess.Process] = None
            communicate_task: Optional["asyncio.Task[tuple[bytes, bytes]]"] = None
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
                communicate_task = asyncio.create_task(process.communicate(prompt.encode("utf-8")))
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    asyncio.shield(communicate_task), timeout=self.config.model_timeout_seconds
                )
            except asyncio.TimeoutError as error:
                if process is not None:
                    await _stop_process_group(process)
                if communicate_task is not None:
                    await _reap_communicate(communicate_task)
                raise ModelError("model_timeout", "limit_seconds=" + str(self.config.model_timeout_seconds)) from error
            except asyncio.CancelledError:
                if process is not None:
                    await _stop_process_group(process)
                if communicate_task is not None:
                    await _reap_communicate(communicate_task)
                raise
            except OSError as error:
                raise ModelError("codex_not_found", str(error)) from error
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
            if process.returncode != 0:
                detail = (stderr + "\n" + stdout)[-4000:]
                raise ModelError(_classify_failure(detail), detail)
            if not output_path.exists():
                raise ModelError("model_output_missing", stdout[-1000:])
            raw = output_path.read_text(encoding="utf-8", errors="replace").strip()
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ModelError("model_output_invalid_json", raw[-1000:]) from error
            if not isinstance(value, dict):
                raise ModelError("model_output_invalid_json", "Ответ не является JSON-объектом")
            return ModelResult(value=value, stdout=stdout, stderr=stderr)


def codex_binary_from_environment(value: Optional[str]) -> Path:
    """Resolve the configured CLI, current desktop bundle, then PATH."""

    configured = Path(value) if value else None
    if configured is not None and configured.exists():
        return configured
    resources = Path("/Applications/ChatGPT.app/Contents/Resources")
    # ChatGPT has shipped the CLI at both the resources root and inside its
    # bundled CodexCLI app. A stale saved path should not disable the bot.
    desktop_candidates = [
        resources / "codex",
        resources / "codex-cli/CodexCLI.app/Contents/MacOS/codex",
        *sorted(resources.glob("codex-cli/*/Contents/MacOS/codex")),
    ]
    for desktop in desktop_candidates:
        if desktop.is_file():
            return desktop
    discovered = shutil.which("codex")
    return Path(discovered) if discovered else configured or Path("codex")
