"""Historical submission identity is pinned to independently captured lineage.

Synthetic contract: operator inventory carries ``historical_submission_lineage``
keyed by the exact indexed source ID. Each entry pins the release version and
indexed digest plus the accepted DB record's ID, date, source digest, normalized
DB digest, and *separately measured* normalized file digest. Those last two can
legitimately differ in the audited snapshot; neither is a substitute for the
other. A deployment must authenticate this metadata as trusted operator input.
The implementation may use a different representation with the same guarantees.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from tof_reliability.vitaly_bot.hypotheses_learning_cycle import AutomaticInterestCycle


def digest(data):
    return hashlib.sha256(data).hexdigest()


class FakeStore:
    def __init__(self, path):
        self.path = path
        with self._connect() as db:
            db.execute('CREATE TABLE meeting_submissions (id TEXT PRIMARY KEY, status TEXT, '
                       'meeting_date TEXT, source_path TEXT, normalized_path TEXT, '
                       'source_sha256 TEXT, normalized_sha256 TEXT)')

    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def get_meeting(self, meeting_id):
        with self._connect() as db:
            row = db.execute('SELECT * FROM meeting_submissions WHERE id=?', (meeting_id,)).fetchone()
            return dict(row) if row else None

    def add(self, record):
        with self._connect() as db:
            db.execute('INSERT INTO meeting_submissions VALUES (:id, :status, :meeting_date, '
                       ':source_path, :normalized_path, :source_sha256, :normalized_sha256)', record)


class FakeReleases:
    def __init__(self, root, sources):
        self.root = root
        self.sources = sources

    def _validate(self, directory, expected):
        return {}, {'source_index': {'sources': copy.deepcopy(self.sources)},
                    'interest_map': {}, 'dimension_projection': {}}

    def _atomic_json(self, path, value):
        path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')


class FakeManager:
    def __init__(self, releases):
        self.release_store = releases
        self.active = {'version': 'baseline', 'manifest_sha256': 'pinned-manifest'}

    def current(self):
        return copy.deepcopy(self.active)


@pytest.fixture
def lab():
    base = Path(os.environ.get('WORKFLOW_OUTPUT_DIR', tempfile.gettempdir()))
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='historical-dedup-', dir=base) as name:
        root = Path(name)
        def file(name, data):
            path = root / name
            path.write_bytes(data)
            return str(path)
        indexed = b'indexed transformed content of historical submission'
        normalized = b'original normalized snapshot content'
        original = b'original accepted submission'
        historical = {'id': 'submission-abc123', 'date': '2026-07-14', 'kind': 'transcript',
                      'status': 'accepted', 'path': file('indexed.txt', indexed),
                      'original_path': file('indexed-original.txt', indexed),
                      'sha256': digest(indexed), 'original_sha256': digest(indexed)}
        record = {'id': 'abc123', 'status': 'accepted', 'meeting_date': '2026-07-14',
                  'source_path': file('db-original.txt', original),
                  'normalized_path': file('db-normalized.txt', normalized),
                  'source_sha256': digest(original), 'normalized_sha256': digest(normalized)}
        new_data = b'genuinely new accepted meeting'
        new = {'id': 'abc1234', 'status': 'accepted', 'meeting_date': '2026-08-03',
               'source_path': file('new-original.txt', new_data),
               'normalized_path': file('new-normalized.txt', new_data),
               'source_sha256': digest(new_data), 'normalized_sha256': digest(new_data)}
        store = FakeStore(root / 'test.sqlite')
        store.add(record)
        store.add(new)
        releases = FakeReleases(root / 'releases', [historical])
        manager = FakeManager(releases)
        inventory = {'complete': True, 'baseline_version': 'baseline',
                     'historical_source_ids': [historical['id']], 'historical_sources': [historical],
                     'meetings': [], 'historical_submission_lineage': {
                         historical['id']: {'baseline_version': 'baseline',
                         'indexed_sha256': historical['sha256'], 'meeting_id': record['id'],
                         'meeting_date': record['meeting_date'], 'source_sha256': record['source_sha256'],
                         'normalized_sha256': record['normalized_sha256'],
                         'normalized_file_sha256': digest(normalized)}}}
        path = root / 'inventory.json'
        path.write_text(json.dumps(inventory), encoding='utf-8')
        engine = object.__new__(AutomaticInterestCycle)
        engine.store, engine.releases, engine.manager, engine.inventory_path = store, releases, manager, path
        yield {'root': root, 'store': store, 'engine': engine, 'path': path, 'seed': inventory,
               'historical': historical, 'record': record, 'new': new, 'releases': releases, 'manager': manager}


def meeting_row(record):
    return {'id': record['id'], 'meeting_id': record['id'], 'date': record['meeting_date'],
            'kind': 'meeting', 'status': 'accepted', 'path': record['normalized_path'],
            'original_path': record['source_path'], 'sha256': record['normalized_sha256'],
            'original_sha256': record['source_sha256']}


def write_inventory(lab, inv):
    lab['path'].write_text(json.dumps(inv), encoding='utf-8')


def test_refresh_uses_exact_verified_historical_submission_and_preserves_lineage(lab):
    result = lab['engine'].refresh_inventory()
    assert result['historical_sources'] == [lab['historical']]
    assert [row['meeting_id'] for row in result['meetings']] == [lab['new']['id']]
    assert result['historical_submission_lineage'] == lab['seed']['historical_submission_lineage']


def test_inputs_accepts_verified_historical_identity_and_keeps_new_meeting(lab):
    inventory = copy.deepcopy(lab['seed'])
    inventory['meetings'] = [meeting_row(lab['new'])]
    write_inventory(lab, inventory)
    result, texts, _, _, _ = lab['engine']._inputs()
    assert set(texts) == {lab['new']['id']}
    assert result['historical_sources'] == [lab['historical']]


@pytest.mark.parametrize('mutation', ['date', 'db_normalized_hash', 'db_source_hash', 'normalized_bytes', 'source_bytes', 'missing_lineage', 'wrong_release', 'wrong_index_hash'])
@pytest.mark.parametrize('entrypoint', ['refresh_inventory', '_inputs'])
def test_changed_or_untrusted_historical_record_fails_closed(lab, mutation, entrypoint):
    inventory = copy.deepcopy(lab['seed'])
    inventory['meetings'] = [meeting_row(lab['new'])]
    lineage = inventory['historical_submission_lineage']['submission-abc123']
    if mutation in {'date', 'db_normalized_hash', 'db_source_hash'}:
        column = {'date': 'meeting_date', 'db_normalized_hash': 'normalized_sha256',
                  'db_source_hash': 'source_sha256'}[mutation]
        with lab['store']._connect() as db:
            db.execute(f'UPDATE meeting_submissions SET {column}=? WHERE id=?',
                       ('2030-01-01' if mutation == 'date' else '0' * 64, 'abc123'))
    elif mutation in {'normalized_bytes', 'source_bytes'}:
        Path(lab['record']['normalized_path' if mutation == 'normalized_bytes' else 'source_path']).write_bytes(b'tampered')
    elif mutation == 'missing_lineage':
        inventory.pop('historical_submission_lineage')
    elif mutation == 'wrong_release':
        lineage['baseline_version'] = 'wrong-version'
    else:
        lineage['indexed_sha256'] = '0' * 64
    write_inventory(lab, inventory)
    before = lab['path'].read_bytes()
    with pytest.raises((ValueError, OSError)):
        getattr(lab['engine'], entrypoint)()
    assert lab['path'].read_bytes() == before


def test_independently_pinned_existing_normalized_path_db_digest_mismatch(lab):
    # Reproduce the two audited records: the immutable snapshot records both
    # actual file hash and a different historical DB digest, without inventing a hash.
    db_digest = digest(b'original DB normalized value represented by different bytes')
    with lab['store']._connect() as db:
        db.execute('UPDATE meeting_submissions SET normalized_sha256=? WHERE id=?', (db_digest, 'abc123'))
    seed = copy.deepcopy(lab['seed'])
    seed['historical_submission_lineage']['submission-abc123']['normalized_sha256'] = db_digest
    write_inventory(lab, seed)
    result = lab['engine'].refresh_inventory()
    assert [row['meeting_id'] for row in result['meetings']] == [lab['new']['id']]
    assert set(lab['engine']._inputs()[1]) == {lab['new']['id']}


def test_new_submission_with_historical_content_collision_is_not_silently_swallowed(lab):
    old = lab['historical']
    with lab['store']._connect() as db:
        db.execute('UPDATE meeting_submissions SET normalized_sha256=?, normalized_path=? WHERE id=?',
                   (old['sha256'], old['path'], lab['new']['id']))
    before = lab['path'].read_bytes()
    with pytest.raises((ValueError, OSError)):
        lab['engine'].refresh_inventory()
    assert lab['path'].read_bytes() == before


def test_older_index_without_submission_ids_still_recognizes_content_and_includes_new(lab):
    old = copy.deepcopy(lab['historical'])
    old.update(id='legacy-source', path=lab['record']['normalized_path'],
               sha256=lab['record']['normalized_sha256'])
    lab['releases'].sources = [old]
    inv = copy.deepcopy(lab['seed'])
    inv.update(historical_source_ids=['legacy-source'], historical_sources=[old])
    inv.pop('historical_submission_lineage')
    write_inventory(lab, inv)
    result = lab['engine'].refresh_inventory()
    assert [row['meeting_id'] for row in result['meetings']] == [lab['new']['id']]
    assert set(lab['engine']._inputs()[1]) == {lab['new']['id']}


def test_identity_remains_correct_after_promoting_then_rolling_back(lab):
    first = lab['engine'].refresh_inventory()
    new_historical = meeting_row(lab['new'])
    lab['releases'].sources = [lab['historical'], new_historical]
    lab['manager'].active = {'version': 'promoted', 'manifest_sha256': 'promoted-manifest'}
    promoted = lab['engine'].refresh_inventory()
    assert promoted['meetings'] == []
    assert {row['id'] for row in promoted['historical_sources']} == {'submission-abc123', 'abc1234'}
    lab['engine']._inputs()
    lab['releases'].sources = [lab['historical']]
    lab['manager'].active = {'version': 'baseline', 'manifest_sha256': 'pinned-manifest'}
    rolled_back = lab['engine'].refresh_inventory()
    assert [row['meeting_id'] for row in rolled_back['meetings']] == ['abc1234']
    assert rolled_back['historical_submission_lineage'] == first['historical_submission_lineage']
    assert set(lab['engine']._inputs()[1]) == {'abc1234'}


def test_inputs_rejects_new_content_collision(lab):
    old = lab['historical']
    with lab['store']._connect() as db:
        db.execute('UPDATE meeting_submissions SET normalized_sha256=?, normalized_path=? WHERE id=?',
                   (old['sha256'], old['path'], lab['new']['id']))
    inv = copy.deepcopy(lab['seed'])
    inv['meetings'] = []
    write_inventory(lab, inv)
    with pytest.raises((ValueError, OSError)):
        lab['engine']._inputs()


def test_missing_indexed_submission_fails_closed(lab):
    with lab['store']._connect() as db:
        db.execute('DELETE FROM meeting_submissions WHERE id=?', ('abc123',))
    with pytest.raises((ValueError, OSError)):
        lab['engine'].refresh_inventory()


def test_new_submission_matching_historical_snapshot_content_fails_closed(lab):
    old = lab['record']
    with lab['store']._connect() as db:
        db.execute('UPDATE meeting_submissions SET normalized_sha256=?, normalized_path=? WHERE id=?',
                   (old['normalized_sha256'], old['normalized_path'], lab['new']['id']))
    with pytest.raises((ValueError, OSError)):
        lab['engine'].refresh_inventory()


def test_promoted_meeting_source_cannot_change_with_database_hash(lab):
    lab['engine'].refresh_inventory()
    promoted = meeting_row(lab['new'])
    lab['releases'].sources = [lab['historical'], promoted]
    lab['manager'].active = {'version': 'promoted', 'manifest_sha256': 'promoted-manifest'}
    changed = lab['root'] / 'changed-source.txt'
    changed.write_bytes(b'changed original')
    with lab['store']._connect() as db:
        db.execute('UPDATE meeting_submissions SET source_sha256=?, source_path=? WHERE id=?',
                   (digest(b'changed original'), str(changed), lab['new']['id']))
    with pytest.raises((ValueError, OSError)):
        lab['engine'].refresh_inventory()


@pytest.mark.parametrize('entrypoint', ['refresh_inventory', '_inputs'])
def test_historical_original_digest_cannot_be_replaced_in_inventory(lab, entrypoint):
    inv = copy.deepcopy(lab['seed'])
    original = lab['root'] / 'replacement-original.txt'
    original.write_bytes(b'replaced original')
    inv['historical_sources'][0]['original_path'] = str(original)
    inv['historical_sources'][0]['original_sha256'] = digest(b'replaced original')
    inv['meetings'] = [meeting_row(lab['new'])]
    write_inventory(lab, inv)
    with pytest.raises((ValueError, OSError)):
        getattr(lab['engine'], entrypoint)()


def test_minimal_pinned_index_does_not_require_original_digest(lab):
    lab['releases'].sources = [{key: lab['historical'][key]
                                for key in ('id', 'date', 'kind', 'sha256')}]
    result = lab['engine'].refresh_inventory()
    assert [row['meeting_id'] for row in result['meetings']] == [lab['new']['id']]
    assert set(lab['engine']._inputs()[1]) == {lab['new']['id']}


@pytest.mark.parametrize('entrypoint', ['refresh_inventory', '_inputs'])
def test_minimal_index_rejects_replacement_of_submission_original(lab, entrypoint):
    lab['releases'].sources = [{key: lab['historical'][key]
                                for key in ('id', 'date', 'kind', 'sha256')}]
    inv = copy.deepcopy(lab['seed'])
    replacement = lab['root'] / 'replacement.txt'
    replacement.write_bytes(b'changed archived original')
    inv['historical_sources'][0].update(original_path=str(replacement),
                                         original_sha256=digest(b'changed archived original'))
    inv['meetings'] = [meeting_row(lab['new'])]
    write_inventory(lab, inv)
    with pytest.raises((ValueError, OSError)):
        getattr(lab['engine'], entrypoint)()
