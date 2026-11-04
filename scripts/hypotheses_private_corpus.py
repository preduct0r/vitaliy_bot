#!/usr/bin/env python3
"""Assess a manually curated private inventory, training map and holdout topics."""
import argparse
import json
import subprocess
from pathlib import Path

from tof_reliability.vitaly_bot.hypotheses_private_corpus import build_and_assess_private_corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('sources', 'training-map', 'holdout-topics', 'output-dir', 'corpus-version'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()

    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8'))

    def model(request):
        processes = subprocess.run(['ps', '-ax', '-o', 'command='], capture_output=True,
                                   text=True, check=True).stdout.splitlines()
        if any('python' in line and '/scripts/run_vitaly_bot.py' in line for line in processes):
            raise RuntimeError('production bot is active; model execution prohibited')
        # CodexRunner's read-only sandbox still permits reads outside its working
        # directory. Until an OS-level boundary excludes holdout files, running it
        # would not establish the required blindness of the generator.
        raise RuntimeError('generator filesystem isolation from holdout is not verified')

    result = build_and_assess_private_corpus(sources=read(args.sources),
        training_map=read(args.training_map), holdout_topics=read(args.holdout_topics),
        output_dir=args.output_dir, corpus_version=args.corpus_version, model=model)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result['status'] == 'assessed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
