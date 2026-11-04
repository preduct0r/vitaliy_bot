#!/usr/bin/env python3
"""Prepare private corpus indexes, verify interest evidence and score holdout."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tof_reliability.vitaly_bot.hypotheses_interests import (
    assess_provenance, chronological_holdout, prepare_private_corpus,
    score_interest_holdout,
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_private(path: Path, value) -> None:
    path = path.resolve()
    repo = Path(__file__).resolve().parent.parent
    if path == repo or repo in path.parents:
        raise ValueError("private output must be outside the Git checkout")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Never overwrite evidence or follow an existing symlink.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, sort_keys=True)
        out.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--receipt", required=True, type=Path)
    prepare.add_argument("--baseline", required=True, type=Path, help="JSON list of accepted baseline source records")
    prepare.add_argument("--version", required=True)
    prepare.add_argument("--output", required=True, type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--index", required=True, type=Path)
    verify.add_argument("--profile", required=True, type=Path)
    verify.add_argument("--output", required=True, type=Path)
    score = sub.add_parser("score")
    score.add_argument("--index", required=True, type=Path)
    score.add_argument("--expected", required=True, type=Path, help="verifier-only JSON list")
    score.add_argument("--predicted", required=True, type=Path, help="independently generated ordered JSON list")
    score.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            receipt = read(args.receipt)
            meetings = receipt["accepted_meetings"]
            accepted = [m for m in meetings if m.get("status") == "accepted"]
            if receipt.get("missing") != [] or receipt.get("expected") != len(accepted) or receipt.get("found") != len(accepted) or receipt.get("source_bytes_verified") is not True or receipt.get("dates_verified") is not True:
                raise ValueError("receipt does not establish complete verified accepted meetings")
            records = [{"id": m["meeting_submission_id"], "date": m["date"], "status": "accepted", "kind": "transcript", "sha256": m["source_sha256"], "path": m["source_path"]} for m in accepted]
            baseline = read(args.baseline)
            if not isinstance(baseline, list) or not baseline or any(r.get("kind") != "baseline" or r.get("status") != "accepted" for r in baseline):
                raise ValueError("accepted baseline source records required")
            result = prepare_private_corpus(records + baseline, version=args.version)
            chronological_holdout(result)
        elif args.command == "verify":
            result = assess_provenance(read(args.profile), read(args.index))
        else:
            index = read(args.index)
            split = chronological_holdout(index)
            result = score_interest_holdout(read(args.expected), read(args.predicted))
            result.update({"corpus_version": index["version"], "holdout_sources": [{"id": s["id"], "date": s["date"], "sha256": s["sha256"]} for s in split["holdout"]]})
        write_private(args.output, result)
        return 0 if result.get("passing", result.get("verified", True)) else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
