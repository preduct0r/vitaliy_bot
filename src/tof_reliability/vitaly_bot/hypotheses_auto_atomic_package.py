"""Offline, authenticated release materialization for a trusted installation.

Source hashes, separate trace IDs and a separately supplied inventory check
consistency; they do not by themselves prove completeness, independence,
authorship, or semantic correctness. No runtime activation is wired here.
"""
from __future__ import annotations

import contextlib
import copy
import fcntl
import hashlib
import hmac
import json
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

from .hypotheses_knowledge import VerifiedReleaseStore
from .hypotheses_pilot import ATTRIBUTIONS


def _bytes(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing original or source')
    data = path.read_bytes()
    if not isinstance(expected, str) or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('source hash mismatch')
    return data


def validate_source_evidence(*, release_store, inventory, candidate, proposal,
                             classifier, verifier, classifier_trace, verifier_trace):
    """Check an offline candidate against its installed baseline and originals.

    The returned mapping contains original bytes for packaging; callers must
    separately capture normalized source bytes and repeat checks immediately
    before committing an authenticated promotion. No publication occurs here.
    """
    current = release_store.current()
    if not current or inventory.get('complete') is not True or inventory.get('baseline_version') != current['version']:
        raise ValueError('baseline or inventory mismatch')
    _, old = VerifiedReleaseStore._validate(
        release_store.root / 'releases' / current['version'], current['manifest_sha256'])
    old_sources = {row['id']: row for row in old['source_index']['sources']}
    historical = inventory.get('historical_sources')
    meetings = inventory.get('meetings')
    ids = inventory.get('historical_source_ids')
    if not isinstance(historical, list) or not isinstance(meetings, list) or not isinstance(ids, list):
        raise ValueError('incomplete inventory')
    if len(ids) != len(set(ids)) or set(ids) != set(old_sources):
        raise ValueError('incomplete historical inventory')
    rows = historical + meetings
    row_ids = [row.get('id') for row in rows]
    if any(not isinstance(value, str) or not value for value in row_ids) or len(row_ids) != len(set(row_ids)):
        raise ValueError('duplicate source')
    if {row.get('id') for row in historical} != set(old_sources):
        raise ValueError('historical source missing')
    for row in historical:
        old_row = old_sources[row['id']]
        if any(row.get(key) != old_row.get(key) for key in ('id', 'sha256', 'date', 'meeting_id')):
            raise ValueError('historical source changed')
    originals, texts, hashes = {}, {}, set()
    for row in rows:
        if row.get('status') != 'accepted' or row.get('conflicting'):
            raise ValueError('unconfirmed or conflicting source')
        data = _bytes(row['path'], row['sha256'])
        originals[row['id']] = _bytes(row['original_path'], row['original_sha256'])
        if row['sha256'] in hashes:
            raise ValueError('duplicate source content')
        hashes.add(row['sha256'])
        texts[row['id']] = data.decode('utf-8')
    if any(row.get('kind') != 'meeting' or not row.get('meeting_id') for row in meetings):
        raise ValueError('meeting identity required')
    meeting_ids = [row['meeting_id'] for row in meetings]
    if len(set(meeting_ids)) != len(meeting_ids):
        raise ValueError('meetings are not distinct')
    historical_meetings = {row.get('meeting_id') for row in historical if row.get('meeting_id')}
    if historical_meetings.intersection(meeting_ids):
        raise ValueError('meeting already present in historical inventory')
    by_id = {row['id']: row for row in meetings}
    evidence = proposal.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('missing proposal evidence')
    if len({e.get('source_id') for e in evidence}) != len(evidence):
        raise ValueError('duplicate evidence')
    if proposal.get('scope') == 'broad' and len(evidence) < 2:
        raise ValueError('insufficient meetings')
    for entry in evidence:
        ident = entry.get('source_id')
        match = re.fullmatch(r'chars:(\d+)-(\d+)', str(entry.get('locator', '')))
        if (ident not in by_id or entry.get('attribution') not in ATTRIBUTIONS or not match
                or int(match[1]) >= int(match[2]) or int(match[2]) > len(texts[ident])):
            raise ValueError('invalid source evidence')
        if any(not old.get('meeting_id') and old.get('date') == by_id[ident].get('date') for old in historical):
            raise ValueError('historical meeting identity is ambiguous')
    version = candidate.get('knowledge_version')
    context = candidate.get('interest_context', {})
    if not version or version == current['version'] or context.get('version') != version:
        raise ValueError('candidate version mismatch')
    items = context.get('interests', [])
    old_items = old['interest_map']['interests']
    if not isinstance(items, list) or items[:-1] != old_items or len(items) != len(old_items) + 1:
        raise ValueError('candidate changed historical interests')
    added = items[-1]
    if added.get('id') != proposal.get('id') or added.get('requirement') != proposal.get('requirement'):
        raise ValueError('candidate differs from proposal')
    if [{k: row.get(k) for k in ('source_id', 'locator', 'attribution')} for row in added.get('evidence', [])] != evidence:
        raise ValueError('candidate evidence differs from proposal')
    for entry in added['evidence']:
        source = by_id[entry['source_id']]
        if (entry.get('date') != source['date'] or entry.get('sha256') != source['sha256']
                or entry.get('reference') != source['id'] + ':' + entry['locator']):
            raise ValueError('candidate source metadata differs')
    if added.get('attributions') != sorted({entry['attribution'] for entry in evidence}):
        raise ValueError('candidate attribution differs')
    dimensions_list = candidate.get('dimension_projection', {}).get('dimensions', [])
    if not isinstance(dimensions_list, list):
        raise ValueError('invalid dimension projection')
    dimensions_by_id = {row['dimension_id']: row for row in dimensions_list}
    if len(dimensions_by_id) != len(dimensions_list):
        raise ValueError('duplicate dimension')
    for previous in old['dimension_projection']['dimensions']:
        current_dimension = dimensions_by_id.get(previous['dimension_id'], {})
        derived = {'evidence', 'interest_ids', 'source_ids', 'independent_meeting_count', 'attribution_meeting_counts'}
        if any(current_dimension.get(key) != value for key, value in previous.items() if key not in derived):
            raise ValueError('historical dimension changed')
        if any(entry not in current_dimension.get('evidence', []) for entry in previous.get('evidence', [])):
            raise ValueError('historical dimension evidence removed')
    # Runtime classifiers emit one proposal, whereas the release format stores
    # one row per cited source. Preserve the actual trace and verify the exact
    # deterministic normalization rather than inventing a model response.
    raw_classifier = classifier_trace.get('answer')
    if raw_classifier != classifier.get('answer'):
        if (not isinstance(raw_classifier, dict)
                or set(raw_classifier) != {'decision', 'interest_id', 'dimension_id', 'requirement', 'scope', 'evidence'}
                or raw_classifier['decision'] != 'candidate'
                or raw_classifier['interest_id'] != proposal.get('id')
                or raw_classifier['requirement'] != proposal.get('requirement')
                or raw_classifier['scope'] != proposal.get('scope')
                or raw_classifier['evidence'] != evidence):
            raise ValueError('classifier trace mismatch')
        expected_rows = [{**entry, 'interest_id': raw_classifier['interest_id'],
                          'dimension_id': raw_classifier['dimension_id']} for entry in evidence]
        if classifier.get('answer') != {'candidates': expected_rows}:
            raise ValueError('classifier trace normalization mismatch')
    if (not classifier_trace.get('thread_id') or not verifier_trace.get('thread_id')
            or verifier_trace.get('answer') != verifier.get('answer')):
        raise ValueError('trace mismatch')
    if classifier_trace['thread_id'] == verifier_trace['thread_id']:
        raise ValueError('same trace thread')
    candidates = classifier.get('answer', {}).get('candidates')
    verdicts = verifier.get('answer', {}).get('verdicts')
    if not isinstance(candidates, list) or not isinstance(verdicts, list) or len(candidates) != len(evidence) or len(verdicts) != len(evidence):
        raise ValueError('incomplete verification')
    keys = ('interest_id', 'source_id', 'dimension_id', 'locator', 'attribution')
    def key(row):
        return tuple(row.get(k) for k in keys)
    if len({key(row) for row in candidates}) != len(evidence) or {key(row) for row in candidates} != {key(row) for row in verdicts}:
        raise ValueError('verification does not match classifier')
    dimensions = set(dimensions_by_id)
    for row in candidates:
        if (row.get('interest_id') != proposal.get('id') or row.get('dimension_id') not in dimensions
                or {k: row.get(k) for k in ('source_id', 'locator', 'attribution')} not in evidence):
            raise ValueError('classifier does not match evidence')
    if any(row.get('supported') is not True or row.get('confidence') != 'high' or row.get('conflicting') is not False for row in verdicts):
        raise ValueError('unconfirmed or conflicting verdict')
    prior_dimensions = {d['dimension_id']: d for d in old['dimension_projection']['dimensions']}
    linked = set()
    for dimension in dimensions_list:
        prior = prior_dimensions.get(dimension['dimension_id'], {}).get('evidence', [])
        entries = dimension.get('evidence', [])
        if (set(dimension.get('interest_ids', [])) != {e.get('interest_id') for e in entries}
                or set(dimension.get('source_ids', [])) != {e.get('source_id') for e in entries}):
            raise ValueError('dimension identifiers do not match evidence')
        previous = prior_dimensions.get(dimension['dimension_id'])
        new_entries = [e for e in entries if e not in prior]
        if previous:
            new_meetings = {by_id[e['source_id']]['meeting_id'] for e in new_entries if e.get('source_id') in by_id}
            if dimension.get('independent_meeting_count') != previous.get('independent_meeting_count') + len(new_meetings):
                raise ValueError('dimension meeting count mismatch')
            prior_counts = previous.get('attribution_meeting_counts')
            if prior_counts is None:
                expected_counts = None
            elif isinstance(prior_counts, dict):
                expected_counts = dict(prior_counts)
                additions = {(by_id[e['source_id']]['meeting_id'], e['attribution'])
                             for e in new_entries if e.get('source_id') in by_id}
                for _, attribution in additions:
                    expected_counts[attribution] = expected_counts.get(attribution, 0) + 1
            else:
                raise ValueError('invalid prior attribution aggregates')
            if dimension.get('attribution_meeting_counts') != expected_counts:
                raise ValueError('unverified attribution aggregates')
        for item in dimension.get('evidence', []):
            if item not in prior and not any(
                row.get('dimension_id') == dimension['dimension_id'] and
                all(row.get(k) == item.get(k) for k in ('interest_id', 'source_id', 'locator'))
                for row in verdicts
            ):
                raise ValueError('new dimension evidence lacks verified verdict')
            if item not in prior:
                linked.add((item.get('interest_id'), item.get('source_id'), dimension['dimension_id'],
                            item.get('locator'), item.get('attribution')))
    if {key(row) for row in verdicts} != linked:
        raise ValueError('verified interest is not actively bound to its dimension')
    return originals

# The authority and SQLite state are owner-controlled installation configuration.
# They do not protect a host where the bot account itself is compromised.
from .hypotheses_scenario_comparison import (AssessmentAuthority, _digest, _observation, fixed_scenarios,
    _fixed_request, _script_fingerprint, _target_binding_changed)
from . import hypotheses_pilot as pilot


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _verify_assessment(audit, baseline, candidate, proposal):
    scenarios = fixed_scenarios()
    records = audit.get('scenarios')
    if not isinstance(records, list) or len(records) != len(scenarios) or not audit.get('generation_id'):
        raise ValueError('assessment scenarios incomplete')
    coverage = audit.get('coverage')
    if (not isinstance(coverage, dict) or _digest(coverage) != audit.get('coverage_digest')
            or audit.get('scripts_sha256') != _script_fingerprint()
            or set(audit.get('components_sha256', {})) != {'generator', 'verifier'}):
        raise ValueError('assessment execution context mismatch')
    improved = False
    common = None
    for scenario, record in zip(scenarios, records):
        if any(record.get(key) != scenario[key] for key in ('id', 'kind', 'input', 'expected', 'checks')):
            raise ValueError('assessment scenario mismatch')
        observations = []
        requests = []
        for side, payload in (('baseline', baseline), ('candidate', candidate)):
            run = record[side]
            request = run['input']
            if any(run.get(field + '_sha256') != _digest(run.get(field))
                   for field in ('input', 'output', 'verified_cards')):
                raise ValueError('assessment execution fingerprint mismatch')
            if ('knowledge_version' in request or
                request.get('coverage', {}).get('generation_id') != audit['generation_id'] or
                request.get('scenario') != {k: scenario[k] for k in ('id', 'kind', 'input')} or
                not run.get('output') or not run.get('verified_cards')):
                raise ValueError('assessment execution mismatch')
            fixed = {k: v for k, v in _fixed_request(request).items() if k != 'scenario'}
            if common is None:
                common = fixed
            elif common != fixed:
                raise ValueError('assessment fixed inputs changed')
            reviews = iter(run.get('reviews', []))
            def replay_generator(actual):
                actual.pop('knowledge_version', None)
                actual['scenario'] = {k: scenario[k] for k in ('id', 'kind', 'input')}
                if actual != request:
                    raise ValueError('assessment prompt does not match knowledge')
                return copy.deepcopy(run['output'])
            def replay_verifier(actual):
                review = next(reviews, None)
                if not review or review.get('input_sha256') != _digest(actual):
                    raise ValueError('assessment verifier input mismatch')
                return copy.deepcopy(review['output'])
            replay = pilot.generate_dimension_pilot(
                scope=request['scope'], knowledge_version=payload['knowledge_version'],
                interest_context=copy.deepcopy(payload['interest_context']),
                dimension_projection=copy.deepcopy(payload['dimension_projection']),
                coverage_provider=lambda: copy.deepcopy(coverage), generator=replay_generator,
                private_verifier=replay_verifier)
            if next(reviews, None) is not None or replay.get('rejected_cards') or replay['cards'] != run['verified_cards']:
                raise ValueError('assessment output does not match verified cards')
            actual = {check['kind']: _observation(check['kind'], run['verified_cards'], request['scope'])
                      for check in scenario['checks']}
            if actual != run['observations']:
                raise ValueError('assessment observations mismatch')
            observations.append(actual)
            requests.append(request)
        if _fixed_request(requests[0]) != _fixed_request(requests[1]):
            raise ValueError('assessment inputs differ')
        old, new = observations
        checks = scenario['checks']
        comparison = {'mandatory_pass': all(new[c['kind']] for c in checks if c['mandatory']),
                      'degraded': any(old[c['kind']] and not new[c['kind']] for c in checks),
                      'new_critical_error': any(c['critical'] and old[c['kind']] and not new[c['kind']] for c in checks),
                      'improved': (any(c['target'] and not old[c['kind']] and new[c['kind']] for c in checks)
                                   and _target_binding_changed(requests[0], requests[1], record['baseline']['verified_cards'], record['candidate']['verified_cards']))}
        if (comparison != record['comparison'] or not comparison['mandatory_pass'] or
            comparison['degraded'] or comparison['new_critical_error']):
            raise ValueError('assessment comparison mismatch')
        if comparison['improved']:
            if not any(c.get('meeting_interest', {}).get('interest_id') == proposal['id']
                       and c['meeting_interest'].get('dimension_id') == 'repair_repeat_interval'
                       for c in record['candidate']['verified_cards']):
                raise ValueError('improvement is unrelated to the proposed interest')
            improved = True
    if not improved:
        raise ValueError('assessment shows no improvement')


class AtomicReleaseManager:
    """Authenticated offline release chain; caller supplies separately trusted config."""
    def __init__(self, *, release_store, trusted_manifest_sha256, authority_path,
                 authority_fingerprint, state_path, inventory_path=None):
        self.release_store = release_store
        self.anchor = trusted_manifest_sha256
        if not isinstance(self.anchor, str) or not re.fullmatch(r'[0-9a-f]{64}', self.anchor):
            raise ValueError('invalid initial trust anchor')
        self.authority = AssessmentAuthority(authority_path, authority_fingerprint)
        self.state_path = Path(state_path)
        self.marker_path = self.state_path.with_suffix(self.state_path.suffix + '.initialized')
        self.inventory_path = Path(inventory_path) if inventory_path is not None else None
        if any((p / '.git').exists() for p in (self.state_path.resolve().parent, *self.state_path.resolve().parents)):
            raise ValueError('trusted state must be outside Git')
        self.state_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with self._locked() as db:
            self._initialize(db)
            if not self.marker_path.exists():
                fd = os.open(self.marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    os.write(fd, b'initialized\n')
                    os.fsync(fd)
                finally:
                    os.close(fd)
                _fsync_dir(self.marker_path.parent)
            self._recover(db)

    @contextlib.contextmanager
    def _locked(self):
        lock_path = self.state_path.with_suffix(self.state_path.suffix + '.lock')
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if self.marker_path.exists() and not self.state_path.is_file():
                raise ValueError('trusted state missing')
            db = sqlite3.connect(self.state_path)
            try:
                os.chmod(self.state_path, 0o600)
                db.execute('PRAGMA synchronous=FULL')
                yield db
            finally:
                db.close()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _initialize(self, db):
        db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), anchor TEXT, fingerprint TEXT, version TEXT, hash TEXT, pending_version TEXT, pending_hash TEXT, chain_head TEXT, resume_version TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS chain (seq INTEGER PRIMARY KEY, version TEXT UNIQUE, hash TEXT, previous TEXT, mac TEXT)')
        row = db.execute('SELECT anchor, fingerprint FROM state WHERE id=1').fetchone()
        fingerprint = hashlib.sha256(self.authority._key).hexdigest()
        if row is None:
            if self.marker_path.exists():
                raise ValueError('trusted state was reset')
            current = self.release_store.current()
            if not current or current['manifest_sha256'] != self.anchor:
                raise ValueError('initial pointer must match configured trust anchor')
            record = {'version': current['version'], 'hash': self.anchor, 'previous': ''}
            db.execute('INSERT INTO chain VALUES (1,?,?,?,?)', (*record.values(), self.authority._mac(record)))
            db.execute('INSERT INTO state VALUES (1,?,?,?,?,NULL,NULL,?,NULL)', (self.anchor, fingerprint, current['version'], self.anchor, self.authority._mac(record)))
            db.commit()
        elif row != (self.anchor, fingerprint):
            raise ValueError('trusted configuration changed')

    def _chain(self, db):
        previous = ''
        releases = {}
        for seq, version, digest, prior, mac in db.execute('SELECT seq, version, hash, previous, mac FROM chain ORDER BY seq'):
            record = {'version': version, 'hash': digest, 'previous': prior}
            if seq != len(releases) + 1 or prior != previous or not hmac.compare_digest(mac, self.authority._mac(record)):
                raise ValueError('release chain invalid')
            if seq == 1 and digest != self.anchor:
                raise ValueError('release anchor mismatch')
            self.release_store._validate(self.release_store.root / 'releases' / version, digest)
            releases[version] = digest
            previous = mac
        if not releases:
            raise ValueError('empty release chain')
        if db.execute('SELECT chain_head FROM state WHERE id=1').fetchone()[0] != previous:
            raise ValueError('release chain truncated')
        return releases, previous

    def _recover(self, db):
        releases, _ = self._chain(db)
        version, digest, pending_version, pending_hash = db.execute('SELECT version,hash,pending_version,pending_hash FROM state WHERE id=1').fetchone()
        if releases.get(version) != digest:
            raise ValueError('trusted active release invalid')
        expected = {'version': version, 'manifest_sha256': digest}
        current = self.release_store.current()
        if pending_version is not None:
            # Restore the committed pointer after any interruption before final commit.
            allowed = {'version': pending_version, 'manifest_sha256': pending_hash}
            if current != expected and current != allowed:
                raise ValueError('unexpected pointer during recovery')
            if current != expected:
                self.release_store._atomic_json(self.release_store.root / 'current.json', expected)
                _fsync_dir(self.release_store.root)
            db.execute('UPDATE state SET pending_version=NULL,pending_hash=NULL WHERE id=1')
            db.commit()
        elif current != expected:
            raise ValueError('active pointer does not match trusted state')
        return expected

    def current(self):
        with self._locked() as db:
            return self._recover(db)

    def bind(self, *, profile_version, knowledge_package_id, case_id, run_id):
        with self._locked() as db:
            self._recover(db)
            return self.release_store.bind(profile_version=profile_version,
                knowledge_package_id=knowledge_package_id, case_id=case_id, run_id=run_id)

    def load_bound(self, binding):
        with self._locked() as db:
            self._recover(db)
            releases, _ = self._chain(db)
            if releases.get(binding.get('knowledge_version')) != binding.get('manifest_sha256'):
                raise ValueError('binding is not in the trusted chain')
            return self.release_store.load_bound(binding)

    def load_version(self, version):
        """Load a historical or active release only from the authenticated chain."""
        with self._locked() as db:
            self._recover(db)
            releases, _ = self._chain(db)
            digest = releases.get(version)
            if not digest:
                raise ValueError('release version is not trusted')
            record = json.loads((self.release_store.root / 'records' / (version + '.json')).read_text(encoding='utf-8'))
            if record.get('manifest_sha256') != digest:
                raise ValueError('release record does not match trusted chain')
            return self.release_store.load(version)

    def _advance(self, db, version, digest):
        db.execute('UPDATE state SET pending_version=?,pending_hash=? WHERE id=1', (version, digest))
        db.commit()
        self.release_store._atomic_json(self.release_store.root / 'current.json', {'version': version, 'manifest_sha256': digest})
        _fsync_dir(self.release_store.root)
        db.execute('UPDATE state SET version=?,hash=?,pending_version=NULL,pending_hash=NULL,resume_version=NULL WHERE id=1', (version, digest))
        db.commit()

    def rollback(self, version):
        with self._locked() as db:
            self._recover(db)
            releases, _ = self._chain(db)
            if version not in releases:
                raise ValueError('rollback target not in trusted chain')
            self._advance(db, version, releases[version])
            return {'version': version, 'manifest_sha256': releases[version]}

    def promote(self, *, inventory, candidate, proposal, classifier, verifier,
                classifier_trace, verifier_trace, assessment_path):
        with self._locked() as db:
            current = self._recover(db)
            releases, previous = self._chain(db)
            version = VerifiedReleaseStore._version(candidate.get('knowledge_version'))
            resume = db.execute('SELECT resume_version FROM state WHERE id=1').fetchone()[0]
            if resume is not None and resume != version:
                raise ValueError('another release is awaiting resumption')
            if version in releases and resume != version:
                raise ValueError('release version already exists')
            if self.inventory_path is None or self.inventory_path.is_symlink() or not self.inventory_path.is_file():
                raise ValueError('independent inventory unavailable')
            if any((p / '.git').exists() for p in (self.inventory_path.resolve().parent, *self.inventory_path.resolve().parents)):
                raise ValueError('independent inventory must be outside Git')
            inventory_bytes = self.inventory_path.read_bytes()
            if json.loads(inventory_bytes) != inventory:
                raise ValueError('inventory differs from configured source')
            originals = validate_source_evidence(release_store=self.release_store, inventory=inventory,
                candidate=candidate, proposal=proposal, classifier=classifier, verifier=verifier,
                classifier_trace=classifier_trace, verifier_trace=verifier_trace)
            if not isinstance(assessment_path, (str, os.PathLike)):
                raise ValueError('assessment path required')
            path = Path(assessment_path)
            if path.is_symlink() or not path.is_file():
                raise ValueError('assessment unavailable')
            assessment_bytes = path.read_bytes()
            audit = json.loads(assessment_bytes)
            if not self.authority.verify(audit) or audit.get('activation_allowed') is not True:
                raise ValueError('assessment is not authenticated and successful')
            _, old = self.release_store._validate(self.release_store.root / 'releases' / current['version'], current['manifest_sha256'])
            baseline = {'knowledge_version': current['version'], 'interest_context': old['interest_map'],
                        'dimension_projection': old['dimension_projection']}
            if audit.get('knowledge_payload_sha256') != {'baseline': _digest(baseline), 'candidate': _digest(candidate)}:
                raise ValueError('assessment payload mismatch')
            _verify_assessment(audit, baseline, candidate, proposal)
            # Combine baseline and new audit rows so every retained dimension
            # continues to have a corresponding independently recorded verdict.
            combined_classifier = {'answer': {'candidates': old['classifier']['answer']['candidates'] + classifier['answer']['candidates']}}
            combined_verifier = {'answer': {'verdicts': old['verifier']['answer']['verdicts'] + verifier['answer']['verdicts']}}
            combined_ct = {'thread_id': classifier_trace['thread_id'], 'answer': combined_classifier['answer'], 'prior_trace': old['classifier_trace'], 'new_trace': classifier_trace}
            combined_vt = {'thread_id': verifier_trace['thread_id'], 'answer': combined_verifier['answer'], 'prior_trace': old['verifier_trace'], 'new_trace': verifier_trace}
            context = copy.deepcopy(candidate['interest_context'])
            context['interests'][-1]['evidence'] = copy.deepcopy(proposal['evidence'])
            rows = inventory['historical_sources'] + inventory['meetings']
            payloads = {'interest-map.json': _json_bytes(context), 'source-index.json': _json_bytes({'version': version, 'sources': rows}),
                'dimension-projection.json': _json_bytes(candidate['dimension_projection']), 'classifier.json': _json_bytes(combined_classifier),
                'verifier.json': _json_bytes(combined_verifier), 'classifier-trace.json': _json_bytes(combined_ct),
                'verifier-trace.json': _json_bytes(combined_vt), 'assessment.json': assessment_bytes,
                'baseline-payload.json': _json_bytes(baseline), 'candidate-payload.json': _json_bytes(candidate),
                'inventory.json': _json_bytes(inventory), 'proposal.json': _json_bytes(proposal),
                'new-classifier.json': _json_bytes(classifier), 'new-verifier.json': _json_bytes(verifier),
                'new-classifier-trace.json': _json_bytes(classifier_trace), 'new-verifier-trace.json': _json_bytes(verifier_trace)}
            for i, row in enumerate(rows):
                payloads[f'original-{i}.bin'] = originals[row['id']]
                payloads[f'normalized-{i}.bin'] = _bytes(row['path'], row['sha256'])
            roles = dict(interest_map='interest-map.json', source_index='source-index.json', dimension_projection='dimension-projection.json',
                classifier='classifier.json', verifier='verifier.json', classifier_trace='classifier-trace.json', verifier_trace='verifier-trace.json')
            manifest = {'format': 'hypotheses-knowledge-release.v1', 'purpose': 'production', 'developmental_only': False,
                'version': version, 'source_ids': [r['id'] for r in rows], **roles,
                'files': {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}}
            manifest_bytes = _json_bytes(manifest)
            digest = hashlib.sha256(manifest_bytes).hexdigest()
            if resume is not None and releases.get(version) != digest:
                raise ValueError('resumed release content differs')
            root = self.release_store.root
            stage = Path(tempfile.mkdtemp(prefix='.atomic-', dir=root / 'releases'))
            try:
                for name, data in {**payloads, 'manifest.json': manifest_bytes}.items():
                    fd = os.open(stage / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                    with os.fdopen(fd, 'wb') as output:
                        output.write(data)
                        output.flush()
                        os.fsync(output.fileno())
                _fsync_dir(stage)
                VerifiedReleaseStore._validate(stage, digest)
                # Revalidate original input paths before publication, detecting drift.
                validate_source_evidence(release_store=self.release_store, inventory=inventory, candidate=candidate,
                    proposal=proposal, classifier=classifier, verifier=verifier, classifier_trace=classifier_trace, verifier_trace=verifier_trace)
                if self.inventory_path.read_bytes() != inventory_bytes:
                    raise ValueError('independent inventory changed during staging')
                if path.read_bytes() != assessment_bytes:
                    raise ValueError('assessment changed during staging')
                destination = root / 'releases' / version
                if destination.exists():
                    # A previous process may have stopped after the atomic rename.
                    # Resume only if the complete staged bytes match exactly.
                    VerifiedReleaseStore._validate(destination, digest)
                else:
                    os.replace(stage, destination)
                    _fsync_dir(root / 'releases')
                self.release_store._atomic_json(root / 'records' / (version + '.json'), {'manifest_sha256': digest})
                _fsync_dir(root / 'records')
                if resume is None:
                    record = {'version': version, 'hash': digest, 'previous': previous}
                    mac = self.authority._mac(record)
                    db.execute('INSERT INTO chain (version,hash,previous,mac) VALUES (?,?,?,?)', (*record.values(), mac))
                    db.execute('UPDATE state SET chain_head=?,resume_version=? WHERE id=1', (mac, version))
                    db.commit()
                self._advance(db, version, digest)
                return {'published': True, 'version': version, 'manifest_sha256': digest}
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
