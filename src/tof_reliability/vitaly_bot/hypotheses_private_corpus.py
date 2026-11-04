"""Private, fail-closed operator workflow for chronological interest assessment."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import unicodedata
from pathlib import Path

from .hypotheses_interests import (
    HOLDOUT_DATES, assess_provenance, chronological_holdout, generator_context,
    prepare_private_corpus, score_interest_holdout,
)


def _outside_git(path: Path) -> None:
    probe = path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    result = subprocess.run(['git', '-C', str(probe), 'rev-parse', '--is-inside-work-tree'],
                            capture_output=True, text=True, check=False)
    if result.returncode == 0 and result.stdout.strip() == 'true':
        raise ValueError('private artifacts must be outside every Git worktree')


def _resolve_unicode_path(path: Path) -> Path:
    """Resolve each filename canonically, rejecting ambiguous equivalents."""
    if not path.is_absolute():
        raise ValueError('source path must be absolute')
    current = Path(path.anchor)
    for part in path.parts[1:]:
        matches = [entry for entry in current.iterdir()
                   if unicodedata.normalize('NFC', entry.name) == unicodedata.normalize('NFC', part)]
        if len(matches) != 1:
            raise ValueError('source path missing or Unicode-ambiguous: ' + str(current / part))
        current = matches[0]
    return current


def build_and_assess_private_corpus(*, sources, training_map, holdout_topics, model,
                                    output_dir, corpus_version):
    output = Path(output_dir).resolve()
    _outside_git(output)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output, 0o700)
    artifacts, blockers = {}, []

    def save(name, value):
        path = output / (name + '.json')
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write('\n')
        artifacts[name] = str(path)

    rows, accepted, hashes, identifiers = [], [], {}, set()
    for source in sources:
        row = dict(source)
        ident = row.get('id')
        if not isinstance(ident, str) or not ident or ident in identifiers:
            blockers.append('missing or repeated source id: ' + str(ident))
        identifiers.add(ident)
        if row.get('use_for_simulation') is False:
            row['status'] = 'excluded'
        elif row.get('status') == 'accepted':
            try:
                path = _resolve_unicode_path(Path(row['path']))
                row['path'] = str(path)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != str(row.get('sha256', '')).lower():
                    row['status'] = 'hash_mismatch'
                    blockers.append('sha256 mismatch: ' + str(ident))
                elif digest in hashes:
                    prior = hashes[digest]
                    row['duplicate_of'] = prior['id']
                    if (row.get('date'), row.get('kind')) != (prior.get('date'), prior.get('kind')):
                        blockers.append('duplicate metadata conflict: ' + str(ident))
                else:
                    hashes[digest] = row
                    accepted.append(row)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                row['status'] = 'missing'
                blockers.append('missing or unreadable source ' + str(ident) + ': ' + str(exc))
        elif row.get('status') != 'rejected':
            blockers.append('unresolved source status: ' + str(ident))
        rows.append(row)
    save('inventory', {'sources': rows})
    split = None
    try:
        corpus = prepare_private_corpus(accepted, version=corpus_version)
        save('corpus', corpus)
        present = {s['date'] for s in corpus['sources']}
        for day in HOLDOUT_DATES:
            if day not in present:
                blockers.append('missing holdout date: ' + day)
        split = chronological_holdout(corpus)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        blockers.append('corpus: ' + str(exc))

    # Persist the training map before constructing any generator request.
    save('training_map', training_map)
    save('holdout_topics', holdout_topics)
    if split is not None:
        try:
            train_index = {'version': corpus_version, 'sources': split['training']}
            provenance = assess_provenance(training_map, train_index)
            save('training_provenance', provenance)
            if not provenance['verified']:
                blockers.append('training provenance failed')
            if not isinstance(holdout_topics, list) or not holdout_topics:
                raise ValueError('holdout topics required')
            ids = [t['id'] for t in holdout_topics]
            if any(not isinstance(i, str) or not i.strip() for i in ids) or len(set(ids)) != len(ids):
                raise ValueError('unique holdout topic ids required')
            holdout_profile = {'version': 'holdout-evidence', 'corpus_version': corpus_version,
                'interests': [{'id': t['id'], 'requirement': t['id'], 'evidence': [
                    dict(e, attribution='interpretation') for e in t['evidence']]} for t in holdout_topics]}
            holdout_index = {'version': corpus_version, 'sources': split['holdout']}
            checked = assess_provenance(holdout_profile, holdout_index)
            save('holdout_provenance', checked)
            if not checked['verified']:
                blockers.append('holdout provenance failed')
            covered = {e['source_id'] for t in holdout_topics for e in t['evidence']}
            covered_dates = {s['date'] for s in split['holdout'] if s['id'] in covered}
            for day in HOLDOUT_DATES:
                if day not in covered_dates:
                    blockers.append('holdout topics lack evidence for ' + day)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            blockers.append('provenance: ' + str(exc))
    if not blockers:
        request = generator_context(split, training_map)
        save('model_input', request)
        try:
            response = model(request)
            save('model_output', response)
            predictions = response['predictions']
            if not isinstance(predictions, list) or len(predictions) < 7 or any(not isinstance(x, str) or not x.strip() for x in predictions):
                raise ValueError('model must return at least seven nonempty predictions')
            score = score_interest_holdout([t['id'] for t in holdout_topics], predictions)
            save('score', score)
            if not score['passing']:
                blockers.append('holdout coverage below 0.70')
        except Exception as exc:
            blockers.append('model or scoring failed: ' + str(exc))
    report = {'status': 'blocked' if blockers else 'assessed', 'blockers': blockers,
              'independent_verifier_required': True,
              'limitations': ['Exact topic-label matching is conservative; model output is not proof of interest.',
                              'Independent verification of attribution, topic selection and meeting identity is required.',
                              'Corpus completeness depends on the supplied inventory.']}
    save('report', report)
    return {'status': report['status'], 'artifacts': artifacts}
