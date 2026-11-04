#!/usr/bin/env python3
"""Run a private offline pilot against the current published server generation."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path

from tof_reliability.vitaly_bot.codex import CodexRunner
from tof_reliability.vitaly_bot.config import load_config
from tof_reliability.vitaly_bot.coverage import RemoteCoverageProvider
from tof_reliability.vitaly_bot.hypotheses_interests import assess_provenance, generator_context, pilot_training_index
from tof_reliability.vitaly_bot.hypotheses_pilot import TrustedRequestStore, codex_generator, export_pilot, generate_pilot, subprocess_verifier


def read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True, help='private S1 corpus index')
    parser.add_argument('--profile', type=Path, required=True, help='private verified interest profile')
    parser.add_argument('--verified-labels', type=Path, required=True)
    parser.add_argument('--verifier-audit', type=Path, required=True)
    parser.add_argument('--private-verifier', type=Path, required=True, help='separate executable JSON verifier')
    parser.add_argument('--knowledge-version', required=True)
    parser.add_argument('--scope', default='')
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--user-id', required=True)
    parser.add_argument('--trusted-store', type=Path, required=True, help='private SQLite file outside checkout')
    parser.add_argument('--coverage-request', type=Path, required=True, help='S2 request JSON')
    parser.add_argument('--output', type=Path, required=True, help='new private directory outside checkout')
    args = parser.parse_args()
    try:
        store_path = args.trusted_store.resolve()
        repo = Path(__file__).resolve().parents[1]
        if store_path == repo or repo in store_path.parents:
            raise ValueError('trusted store must be outside the checkout')
        store = TrustedRequestStore(store_path)
        trusted = store.create({'run_id': uuid.uuid4().hex, 'case_id': args.case_id,
                                'user_id': args.user_id, 'scope': args.scope})
        index, profile = read(args.index), read(args.profile)
        training = pilot_training_index(index)
        verification = assess_provenance(profile, training)
        if not verification['verified']:
            raise ValueError('source provenance verification failed')
        context = generator_context({'version': training['version'], 'training': training['sources']}, profile)
        request = read(args.coverage_request)
        provider = RemoteCoverageProvider()
        config = load_config(environ={**os.environ, 'VITALY_BOT_ENV_FILE': '/dev/null'}, require_token=False)
        run = generate_pilot(scope=args.scope, knowledge_version=args.knowledge_version,
                             interest_context=context, verified_labels=read(args.verified_labels),
                             verifier_audit=read(args.verifier_audit),
                             private_verifier=subprocess_verifier(args.private_verifier),
                             coverage_provider=lambda: asyncio.run(provider.build(request)),
                             generator=codex_generator(CodexRunner(config)),
                             trusted_context=trusted, trusted_store=store)
        export_pilot(run, output_dir=args.output, source_verification=verification,
                     trusted_context=trusted, trusted_store=store)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f'pilot blocked: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
