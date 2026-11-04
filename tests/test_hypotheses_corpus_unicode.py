"""Regression contracts for synthetic Unicode paths and opted-out sources."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unicodedata

import pytest

from tof_reliability.vitaly_bot.hypotheses_private_corpus import build_and_assess_private_corpus


@pytest.fixture
def private_root():
    root = Path(os.environ["WORKFLOW_OUTPUT_DIR"]) if os.environ.get("WORKFLOW_OUTPUT_DIR") else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unicode-corpus-test-", dir=root) as directory:
        yield Path(directory)


def _record(ident, path, content, **extra):
    return {"id": ident, "date": "2026-06-03", "kind": "transcript",
            "status": "accepted", "path": str(path),
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(), **extra}


def _inventory_and_corpus(root, sources):
    def unexpected_model(_request):
        pytest.fail("model must not run without holdout evidence")

    result = build_and_assess_private_corpus(
        sources=sources, training_map={}, holdout_topics=[], model=unexpected_model,
        output_dir=root / "artifacts", corpus_version="synthetic-v1",
    )
    artifacts = {name: json.loads(Path(path).read_text(encoding="utf-8"))
                 for name, path in result["artifacts"].items()}
    return artifacts


@pytest.mark.parametrize("day", ["06_03", "06_08", "06_18"])
def test_confirmed_nfc_inventory_path_resolves_nfd_filename(private_root, monkeypatch, day):
    nfc_name = f"{day}_виталий.txt"
    nfd_name = unicodedata.normalize("NFD", nfc_name)
    assert nfc_name != nfd_name
    actual = private_root / nfd_name
    actual.write_text("synthetic source " + day, encoding="utf-8")
    requested = private_root / nfc_name
    source = _record(day, requested, "synthetic source " + day)

    # Make the regression reproducible on normalization-insensitive macOS too:
    # a literal NFC lookup behaves as it would on a byte-sensitive filesystem.
    original_read_bytes = Path.read_bytes
    def byte_sensitive_read_bytes(path):
        if str(path) == str(requested):
            raise FileNotFoundError(str(path))
        return original_read_bytes(path)
    monkeypatch.setattr(Path, "read_bytes", byte_sensitive_read_bytes)

    artifacts = _inventory_and_corpus(private_root, [source])
    row = artifacts["inventory"]["sources"][0]
    assert row["status"] == "accepted"
    assert Path(row["path"]).name == nfd_name
    assert [item["id"] for item in artifacts["corpus"]["sources"]] == [day]


def test_opted_out_records_are_tracked_but_not_required_or_in_corpus(private_root):
    included = private_root / "included.txt"
    included.write_text("synthetic included", encoding="utf-8")
    excluded = private_root / "excluded.txt"
    excluded.write_text("synthetic opted out", encoding="utf-8")
    sources = [
        _record("confirmed", included, "synthetic included"),
        _record("july-probable", excluded, "synthetic opted out", use_for_simulation=False),
        _record("may-opted-out", private_root / "missing-excluded.txt", "missing synthetic",
                use_for_simulation=False),
    ]
    artifacts = _inventory_and_corpus(private_root, sources)
    rows = {row["id"]: row for row in artifacts["inventory"]["sources"]}
    assert all(rows[ident]["status"] == "excluded" for ident in ("july-probable", "may-opted-out"))
    assert [source["id"] for source in artifacts["corpus"]["sources"]] == ["confirmed"]
    assert "may-opted-out" not in json.dumps(artifacts["report"]["blockers"])
