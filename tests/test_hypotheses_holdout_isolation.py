"""Synthetic contracts for the private holdout operator; no live services are used."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


@pytest.fixture
def operator(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/hypotheses_private_corpus.py"
    spec = importlib.util.spec_from_file_location("holdout_operator_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(dir=os.environ.get("WORKFLOW_OUTPUT_DIR")) as directory:
        root = Path(directory)
        inputs = []
        for name, data in (("sources", []), ("training-map", {}), ("holdout-topics", [])):
            path = root / (name + ".json")
            path.write_text(json.dumps(data), encoding="utf-8")
            inputs.extend(("--" + name, str(path)))
        monkeypatch.setattr(sys, "argv", ["operator", *inputs, "--output-dir", str(root / "out"),
                                         "--corpus-version", "synthetic-v1"])
        yield module


def exercise_model(monkeypatch, operator, *, active=False, tool_event=None):
    """Run the real CLI wiring with a fake assessment and fake process table/CLI."""
    calls = []
    bot = "123 1 python /release/scripts/run_vitaly_bot.py"
    process_table = bot + ("\n124 123 codex exec --json synthetic-production-job" if active else "")
    events = [{"type": "thread.started", "thread_id": "synthetic"},
              {"type": "turn.started"}]
    if tool_event is not None:
        events.append(tool_event)
    events.extend([
        {"type": "item.completed", "item": {"id": "message", "type": "agent_message",
         "text": json.dumps({"predictions": ["a", "b", "c", "d", "e", "f", "g"]})}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ])

    def fake_run(args, *pos, **kwargs):
        args = list(map(str, args))
        calls.append(args)
        executable = Path(args[0]).name
        if executable == "ps":
            return subprocess.CompletedProcess(args, 0, stdout=process_table, stderr="")
        if executable == "codex":
            return subprocess.CompletedProcess(args, 0, stdout="\n".join(map(json.dumps, events)), stderr="")
        pytest.fail("unexpected external command in synthetic test: " + executable)

    def fake_assessment(**kwargs):
        response = kwargs["model"]({"training": "SYNTHETIC_TRAINING_ONLY"})
        assert response["predictions"] == ["a", "b", "c", "d", "e", "f", "g"]
        return {"status": "assessed", "artifacts": {}}

    monkeypatch.setattr(operator.subprocess, "run", fake_run)
    monkeypatch.setattr(operator, "build_and_assess_private_corpus", fake_assessment)
    return calls


def test_idle_polling_allows_isolated_tool_free_model_call(monkeypatch, operator):
    calls = exercise_model(monkeypatch, operator)
    assert operator.main() == 0
    launches = [call for call in calls if Path(call[0]).name == "codex"]
    assert len(launches) == 1
    args = launches[0]
    assert "--json" in args and "--ignore-user-config" in args
    for feature in ("shell_tool", "unified_exec", "apps", "plugins", "memories", "multi_agent"):
        assert any(args[i:i + 2] == ["--disable", feature] for i in range(len(args) - 1))
    # Both skills and inherited MCPs must be disabled in the invocation itself.
    options = " ".join(args)
    assert "skills" in options and "mcp_servers" in options


def test_active_production_model_job_blocks_before_launch(monkeypatch, operator):
    calls = exercise_model(monkeypatch, operator, active=True)
    with pytest.raises(RuntimeError, match="(?i)(active|running|busy).*(model|job)|(model|job).*(active|running|busy)"):
        operator.main()
    assert not any(Path(call[0]).name == "codex" for call in calls)


@pytest.mark.parametrize("item", [
    {"id": "cmd", "type": "command_execution", "command": "cat heldout.txt", "status": "completed"},
    {"id": "mcp", "type": "mcp_tool_call", "server": "synthetic", "tool": "read_file", "status": "completed"},
])
def test_any_tool_event_invalidates_evaluation(monkeypatch, operator, item):
    calls = exercise_model(monkeypatch, operator, tool_event={"type": "item.completed", "item": item})
    with pytest.raises(RuntimeError, match="(?i)(tool|command|isolation|trace)"):
        operator.main()
    assert any(Path(call[0]).name == "codex" for call in calls)
