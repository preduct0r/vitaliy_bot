"""Explicit private development knowledge source with replayed provenance."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .hypotheses_extract_runner import parse_trace
from .hypotheses_interests import generator_context
from .hypotheses_pilot import bind_expanded_projection


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


class VerifiedKnowledgeLoader:
    """Load one explicitly bound development corpus and replay all its audits.

    The caller supplies the case profile/package and the SHA of the private
    audit manifest. Nothing here selects a production package automatically.
    """
    def __init__(self, *, corpus_dir, original_dir, adjudication_dir, profile_version,
                 knowledge_package_id, manifest_sha256):
        self.corpus_dir = Path(corpus_dir)
        self.original_dir = Path(original_dir)
        self.adjudication_dir = Path(adjudication_dir)
        self.profile_version = profile_version
        self.knowledge_package_id = knowledge_package_id
        self.manifest_sha256 = manifest_sha256
        if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
            raise ValueError('private audit manifest hash required')

    def load(self, profile_version, knowledge_package_id):
        if (profile_version, knowledge_package_id) != (self.profile_version, self.knowledge_package_id):
            raise ValueError('case profile or knowledge package mismatch')
        try:
            manifest_path = self.adjudication_dir / 'audit-manifest.json'
            if _sha(manifest_path) != self.manifest_sha256:
                raise ValueError('private audit manifest changed')
            manifest = _read(manifest_path)
            if manifest.get('version') != 'conflict-adjudication-audit.v1' or manifest.get('developmental_only') is not True:
                raise ValueError('unsupported private development audit')
            input_hashes = manifest['input_sha256']
            outputs = manifest['output_sha256']
            if not isinstance(input_hashes, dict) or not input_hashes or not isinstance(outputs, dict):
                raise ValueError('incomplete audit inventory')
            for name, digest in input_hashes.items():
                if not Path(name).is_absolute() or _sha(name) != digest:
                    raise ValueError('private audit input changed')
            for name, digest in outputs.items():
                if Path(name).name != name or _sha(self.adjudication_dir / name) != digest:
                    raise ValueError('private audit output changed')
            required_inputs = [self.corpus_dir / 'training-interest-map.json', self.corpus_dir / 'training-index.json',
                               self.original_dir / 'repair-dimensions-private.json']
            classifier_paths = sorted(self.original_dir.glob('classifier-bounded-*.json'))
            verifier_paths = sorted(self.original_dir.glob('verifier-bounded-*.json'))
            if not classifier_paths or not verifier_paths:
                raise ValueError('original classifier and verifier audit required')
            if any(str(p) not in input_hashes for p in [*required_inputs, *classifier_paths, *verifier_paths]):
                raise ValueError('audit inventory omits required input')
            required_outputs = {'original-candidates.json', 'original-verdicts-normalized.json',
                                'adjudications-private.json', 'revised-projection-private.json'}
            if not required_outputs <= set(outputs):
                raise ValueError('audit inventory omits required output')
            classifiers = [_read(p)['answer'] for p in classifier_paths]
            verifiers = [_read(p)['answer'] for p in verifier_paths]
            candidates = [row for answer in classifiers for row in answer['candidates']]
            raw_verdicts = [row for answer in verifiers for row in answer['verdicts']]
            verdicts = [{**{k: row[k] for k in ('interest_id', 'dimension_id', 'source_id', 'locator')},
                         'verdict': {k: row[k] for k in ('supported', 'confidence', 'conflicting')}} for row in raw_verdicts]
            if (candidates != _read(self.adjudication_dir / 'original-candidates.json') or
                    verdicts != _read(self.adjudication_dir / 'original-verdicts-normalized.json')):
                raise ValueError('semantic audit differs from original batches')
            traces = sorted(self.original_dir.glob('trace-*.jsonl'))
            if len(traces) != len(classifiers) + len(verifiers):
                raise ValueError('incomplete independent semantic traces')
            answers, threads = [], set()
            for trace in traces:
                text = trace.read_text(encoding='utf-8')
                events = [json.loads(line) for line in text.splitlines() if line.strip()]
                identities = [e['thread_id'] for e in events if e.get('type') == 'thread.started']
                if len(identities) != 1 or identities[0] in threads:
                    raise ValueError('semantic audit model identities are not independent')
                threads.add(identities[0])
                answers.append(parse_trace(text))
            if Counter(_digest(answer) for answer in answers) != Counter(_digest(answer) for answer in classifiers + verifiers):
                raise ValueError('semantic audit trace mismatch')
            profile, index = (_read(p) for p in required_inputs[:2])
            original = _read(required_inputs[2])
            projection = bind_expanded_projection(profile=profile, source_index=index,
                projection=_read(self.adjudication_dir / 'revised-projection-private.json'),
                original_projection=original, original_verdicts=verdicts, original_candidates=candidates,
                adjudications=_read(self.adjudication_dir / 'adjudications-private.json'))
            context = generator_context({'version': index['version'], 'training': index['sources']}, profile)
            return {'knowledge_version': 'development-audit:' + self.manifest_sha256,
                    'interest_context': context, 'dimension_projection': projection}
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError('private audit unavailable or invalid') from error
