#!/usr/bin/env python3
"""Score top-ten predicted Vitaly question topics against the held-out meetings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

from tof_reliability.vitaly_bot.profile import load_default_profile, score_holdout_coverage


def _topics(value: Any) -> List[str]:
    if isinstance(value, dict):
        questions = value.get("questions", [])
        if isinstance(questions, list):
            return [str(question["topic"]) for question in questions if isinstance(question, dict) and "topic" in question]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("expected a review JSON object or a JSON list of topic ids")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", type=Path, required=True, help="review JSON or topic-id JSON list")
    args = parser.parse_args()
    try:
        topics = _topics(json.loads(args.predicted.read_text(encoding="utf-8")))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    result = score_holdout_coverage(load_default_profile(), topics)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
