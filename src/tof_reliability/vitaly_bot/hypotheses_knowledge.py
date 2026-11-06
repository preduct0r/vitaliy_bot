"""Explicit private development knowledge source with replayed provenance."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
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


class VerifiedReleaseStore:
    """Private, content-addressed releases. The supplied manifest hash is the trust anchor.

    Hashes detect changes; they do not establish who approved a release or prove
    corpus completeness. An operator must independently verify the inventory.
    """
    def __init__(self, root):
        self.root = Path(root)

    @staticmethod
    def _version(value):
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value) or value in ('.', '..'):
            raise ValueError('invalid release version')
        return value

    @staticmethod
    def _validate(directory, expected):
        directory = Path(directory)
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError('manifest SHA-256 required')
        manifest_path = directory / 'manifest.json'
        if manifest_path.is_symlink() or _sha(manifest_path) != expected:
            raise ValueError('manifest hash mismatch')
        m = _read(manifest_path)
        if m.get('format') != 'hypotheses-knowledge-release.v1' or m.get('purpose') != 'production' or m.get('developmental_only') is not False:
            raise ValueError('not a production release')
        version = VerifiedReleaseStore._version(m.get('version'))
        inventory = m.get('files')
        if not isinstance(inventory, dict) or not inventory:
            raise ValueError('empty release inventory')
        for name, digest in inventory.items():
            if not isinstance(name, str) or Path(name).name != name or name in ('.', '..', 'manifest.json') or not re.fullmatch(r'[0-9a-f]{64}', str(digest)):
                raise ValueError('invalid inventory entry')
            path = directory / name
            if path.is_symlink() or not path.is_file() or _sha(path) != digest:
                raise ValueError('release file changed')
        roles = ('interest_map', 'source_index', 'dimension_projection', 'classifier', 'verifier', 'classifier_trace', 'verifier_trace')
        if any(not isinstance(m.get(k), str) or m[k] not in inventory for k in roles) or len({m[k] for k in roles}) != len(roles):
            raise ValueError('release roles missing or overlapping')
        values = {k: _read(directory / m[k]) for k in roles}
        index, interests = values['source_index'], values['interest_map']
        if index.get('version') != version or interests.get('version') != version:
            raise ValueError('release version mismatch')
        sources = index.get('sources')
        items = interests.get('interests')
        if not isinstance(sources, list) or not sources or not isinstance(items, list) or not items:
            raise ValueError('source inventory or interest map missing')
        source_ids = [row['id'] for row in sources]
        if len(set(source_ids)) != len(source_ids) or set(m.get('source_ids', [])) != set(source_ids) or len(m.get('source_ids', [])) != len(source_ids):
            raise ValueError('source inventory mismatch')
        for item in items:
            references = [item['source_id']] if 'source_id' in item else [e['source_id'] for e in item.get('evidence', [])]
            if not references or not set(references) <= set(source_ids):
                raise ValueError('interest source mismatch')
        a, b = values['classifier_trace'], values['verifier_trace']
        if not a.get('thread_id') or not b.get('thread_id') or a['thread_id'] == b['thread_id'] or a.get('answer') != values['classifier'].get('answer') or b.get('answer') != values['verifier'].get('answer'):
            raise ValueError('independent trace mismatch')
        candidates = values['classifier'].get('answer', {}).get('candidates')
        verdicts = values['verifier'].get('answer', {}).get('verdicts')
        if not isinstance(candidates, list) or not candidates or not isinstance(verdicts, list):
            raise ValueError('classifier or verifier results missing')
        def audit_key(row):
            if not isinstance(row, dict) or not row.get('interest_id') or not row.get('source_id'):
                raise ValueError('invalid audit row')
            return tuple(row.get(k) for k in ('interest_id', 'source_id', 'dimension_id', 'locator'))
        candidate_keys = [audit_key(row) for row in candidates]
        verdict_keys = [audit_key(row) for row in verdicts]
        if len(set(candidate_keys)) != len(candidate_keys) or Counter(candidate_keys) != Counter(verdict_keys):
            raise ValueError('classifier candidates and verifier verdicts differ')
        known_interests = {item['id'] for item in items}
        if any(row['interest_id'] not in known_interests or row['source_id'] not in source_ids for row in candidates):
            raise ValueError('audit cites unknown interest or source')
        if any(type(row.get('supported')) is not bool or row.get('confidence') not in ('high', 'low') for row in verdicts):
            raise ValueError('invalid verifier verdict')
        approved = {audit_key(row) for row in verdicts if row['supported'] and row['confidence'] == 'high'
                    and not row.get('conflicting', False)}
        dimensions = values['dimension_projection'].get('dimensions')
        if not isinstance(dimensions, list):
            raise ValueError('invalid dimension projection')
        for dimension in dimensions:
            for evidence in dimension.get('evidence', []):
                key = (evidence.get('interest_id'), evidence.get('source_id'),
                       dimension.get('dimension_id'), evidence.get('locator'))
                if key not in approved:
                    raise ValueError('dimension lacks an independent supported verdict')
        return m, values

    def _atomic_json(self, path, value):
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.release-', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(value, stream, ensure_ascii=False, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def install(self, source_dir, *, manifest_sha256):
        target = self.root.resolve()
        if any((parent / '.git').exists() for parent in (target, *target.parents)):
            raise ValueError('private releases must be outside Git')
        manifest, _ = self._validate(source_dir, manifest_sha256)
        version = manifest['version']
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        releases = self.root / 'releases'
        releases.mkdir(mode=0o700, exist_ok=True)
        destination = releases / version
        if destination.exists():
            existing = _read(self.root / 'records' / (version + '.json'))
            if existing['manifest_sha256'] != manifest_sha256:
                raise ValueError('release version already exists with different content')
            self._validate(destination, manifest_sha256)
        else:
            stage = Path(tempfile.mkdtemp(prefix='.staged-', dir=releases))
            try:
                for name in ['manifest.json', *manifest['files']]:
                    shutil.copyfile(Path(source_dir) / name, stage / name)
                    os.chmod(stage / name, 0o600)
                self._validate(stage, manifest_sha256)
                os.rename(stage, destination)
                self._atomic_json(self.root / 'records' / (version + '.json'), {'manifest_sha256': manifest_sha256})
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        return self.activate(version)

    def activate(self, version):
        version = self._version(version)
        self.load(version)
        record = _read(self.root / 'records' / (version + '.json'))
        current = {'version': version, 'manifest_sha256': record['manifest_sha256']}
        self._atomic_json(self.root / 'current.json', current)
        return current

    def current(self):
        path = self.root / 'current.json'
        if not path.exists():
            return None
        result = _read(path)
        version = self._version(result['version'])
        record = _read(self.root / 'records' / (version + '.json'))
        if record['manifest_sha256'] != result['manifest_sha256']:
            raise ValueError('current release hash mismatch')
        self.load(version)
        return result

    def load(self, version):
        version = self._version(version)
        record = _read(self.root / 'records' / (version + '.json'))
        _, values = self._validate(self.root / 'releases' / version, record['manifest_sha256'])
        context = values['interest_map']
        index = values['source_index']
        # Production maps may contain private fragments; expose only the
        # explicitly selected generalized fields to the generator.
        if all('evidence' in item for item in context['interests']):
            sources = {s['id']: s for s in index['sources']}
            projected = []
            for item in context['interests']:
                evidence = []
                for entry in item['evidence']:
                    source = sources[entry['source_id']]
                    evidence.append({'source_id': source['id'], 'date': source['date'], 'locator': entry['locator'],
                                     'attribution': entry['attribution'], 'sha256': source['sha256'],
                                     'reference': source['id'] + ':' + entry['locator']})
                projected.append({'id': item['id'], 'requirement': item['requirement'],
                                  'attributions': sorted({e['attribution'] for e in evidence}), 'evidence': evidence})
            context = {'version': context['version'], 'corpus_version': index['version'], 'interests': projected}
        return {'knowledge_version': version, 'interest_context': context, 'dimension_projection': values['dimension_projection']}

    def bind(self, *, profile_version, knowledge_package_id, case_id, run_id):
        current = self.current()
        if current is None:
            raise ValueError('no active release')
        binding = {'profile_version': profile_version, 'knowledge_package_id': knowledge_package_id,
                   'case_id': case_id, 'run_id': run_id, 'knowledge_version': current['version'],
                   'manifest_sha256': current['manifest_sha256']}
        return {**binding, 'knowledge_hash': _digest(binding)}

    def load_bound(self, binding):
        binding = dict(binding)
        digest = binding.pop('knowledge_hash', None)
        if digest != _digest(binding):
            raise ValueError('binding changed')
        version = self._version(binding['knowledge_version'])
        record = _read(self.root / 'records' / (version + '.json'))
        if binding['manifest_sha256'] != record['manifest_sha256']:
            raise ValueError('bound release changed')
        return self.load(version)


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
