"""Remote client for the fixed published-generation coverage operation."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from .models import EvidenceError

SERVER_PYTHON = '/srv/tof/repos/tof_v1/.venv/bin/python'


def _remote_program(request: dict[str, Any]) -> bytes:
    source = Path(__file__).with_name('server_coverage.py').read_bytes()
    payload = base64.b64encode(json.dumps(request, ensure_ascii=False).encode()).decode('ascii')
    invocation = ("\nimport base64 as _b64, json as _json\n"
                  + 'print(_json.dumps(build_coverage(_json.loads(_b64.b64decode('
                  + repr(payload) + '))), ensure_ascii=False))\n')
    return source + invocation.encode()


class RemoteCoverageProvider:
    def __init__(self, *, timeout_seconds: float = 120) -> None:
        self.timeout_seconds = timeout_seconds

    async def build(self, coverage_request: dict[str, Any]) -> dict[str, Any]:
        try:
            program = _remote_program(coverage_request)
            process = await asyncio.create_subprocess_exec(
                'ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                os.environ.get('TOF_SSH_HOST', 'tof'), SERVER_PYTHON + ' -',
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(program), self.timeout_seconds)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
                await process.wait()
                raise
        except (OSError, asyncio.TimeoutError) as exc:
            raise EvidenceError('tof_unavailable', str(exc)) from exc
        if process.returncode:
            raise EvidenceError('tof_unavailable', stderr.decode('utf-8', errors='replace')[-4000:])
        try:
            value = json.loads(stdout)
            if not isinstance(value, dict) or value.get('coverage_version') != 1 or not isinstance(value.get('generation'), dict) or not value['generation'].get('run_id') or not isinstance(value.get('routes'), list):
                raise ValueError('invalid coverage response')
        except (ValueError, UnicodeError) as exc:
            raise EvidenceError('tof_coverage_invalid', str(exc)) from exc
        return value
