"""Offline scenario comparison; a positive result is evidence, not a release.

The fixed inputs are synthetic.  The injected generator and verifier must be
separate components.  No provider fallback, scheduler or activation is supplied.
The textual predicates are deliberately narrow, inspectable checks, not a
semantic assessment or a measurement of general model quality.
"""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
import subprocess
from pathlib import Path
from uuid import uuid4

from . import hypotheses_pilot as pilot
from . import hypotheses_interests, hypotheses_topic_projection, models


class AssessmentAuthority:
    """Configured owner-controlled authority, separate from the release store."""
    def __init__(self, key_path, fingerprint):
        path = Path(key_path)
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
            raise ValueError('private authority key required')
        if any((p / '.git').exists() for p in (path.resolve().parent, *path.resolve().parents)):
            raise ValueError('authority must be outside Git')
        self._key = path.read_bytes()
        if len(self._key) < 32 or not hmac.compare_digest(hashlib.sha256(self._key).hexdigest(), str(fingerprint)):
            raise ValueError('authority fingerprint mismatch')

    def _mac(self, value):
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        return hmac.new(self._key, b'hypotheses-assessment-v1\0' + data, hashlib.sha256).hexdigest()

    def verify(self, audit):
        body = dict(audit)
        signature = body.pop('authority_hmac_sha256', '')
        return isinstance(signature, str) and hmac.compare_digest(signature, self._mac(body))


def _case(ident, kind, data, expected, check, *, target=False):
    checks = [{'kind': check, 'mandatory': not target, 'critical': not target,
               'target': target}]
    if target:
        checks.append({'kind': 'grounded', 'mandatory': True, 'critical': True, 'target': False})
    return {'id': ident, 'kind': kind, 'input': data, 'expected': expected, 'checks': checks}


def fixed_scenarios():
    """Return the pinned synthetic cases; callers cannot redefine the oracle."""
    return [
        _case('contradiction-true', 'true_contradiction', {'same_measure': True, 'same_period': True, 'values': [4, 7]}, {'contradiction': True}, 'true_contradiction'),
        _case('contradiction-false', 'false_contradiction', {'same_measure': True, 'same_period': False, 'values': [4, 7]}, {'contradiction': False}, 'false_contradiction'),
        _case('units', 'units', {'quantities': [{'value': 1, 'unit': 'hour'}, {'value': 60, 'unit': 'minute'}]}, {'equivalent_minutes': 60}, 'units'),
        _case('unsupported-data', 'unsupported_data', {'supported': [], 'claim': 'historic cost'}, {'historic_cost_known': False}, 'grounded'),
        _case('criticality', 'criticality', {'register': 'unavailable'}, {'criticality_known': False}, 'criticality'),
        _case('repair-episodes', 'repair_episodes', {'event_type': 'repair_episode', 'periods': ['A', 'B']}, {'measure': 'repeat_interval'}, 'interval_target', target=True),
        _case('scope', 'scope', {'requested': 'pump'}, {'widening_allowed': False}, 'scope'),
        _case('missing-join', 'missing_join', {'equipment_to_event_link': None}, {'join_known': False}, 'missing_join'),
        _case('outage', 'outage', {'coverage_status': 'unavailable'}, {'cached_fallback_allowed': False}, 'outage'),
        _case('periods', 'periods', {'periods_comparable': False}, {'comparison_requires_period': True}, 'periods'),
        _case('attribution', 'attribution', {'direct_request_distinct_from': 'third_party'}, {'attribution_preserved': True}, 'attribution'),
        _case('source', 'source', {'evidence': 'verified_route_only'}, {'source_required': True}, 'source'),
    ]


def _digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _script_fingerprint():
    return {Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (pilot, hypotheses_interests, hypotheses_topic_projection, models,
                           __import__(__name__, fromlist=['']))}


def _callable_fingerprint(value):
    code = getattr(value, '__code__', None) or getattr(getattr(value, '__call__', None), '__code__', None)
    if code is None:
        raise ValueError('inspectable comparison component required')
    def describe(item):
        if hasattr(item, 'co_code'):
            return {'bytecode': item.co_code.hex(), 'names': item.co_names,
                    'constants': [describe(constant) for constant in item.co_consts]}
        if isinstance(item, (str, int, float, bool, type(None))):
            return item
        if isinstance(item, tuple):
            return [describe(element) for element in item]
        return repr(item)
    return _digest(describe(code))


def _fixed_request(request):
    return {k: v for k, v in request.items() if k != 'dimensions'}


def _target_binding_changed(old, new, old_cards, new_cards):
    """The target must use a newly exposed, source-bound interest on its own dimension."""
    dimension = 'repair_repeat_interval'
    before = next((d for d in old['dimensions'] if d['dimension_id'] == dimension), None)
    after = next((d for d in new['dimensions'] if d['dimension_id'] == dimension), None)
    if not before or not after or before == after:
        return False
    added = set(after['interest_ids']) - set(before['interest_ids'])
    return bool(added and any(c.get('meeting_interest', {}).get('dimension_id') == dimension
                              and c['meeting_interest'].get('interest_id') in added for c in new_cards)
                and old_cards != new_cards)


def _outside_git(path):
    path = Path(path).resolve(strict=True)
    if not path.is_dir():
        raise ValueError('audit directory required')
    result = subprocess.run(['git', '-C', str(path), 'rev-parse', '--show-toplevel'],
                            capture_output=True, text=True, check=False)
    if result.returncode == 0:
        raise ValueError('audit directory must be outside Git worktrees')
    return path


def _privacy_check(value, contexts):
    """Reject direct echoes of private requirement/fragment and marker tokens.

    This cannot prove absence of paraphrased personal data; callers must use a
    synthetic or otherwise approved isolation boundary for the injected model.
    """
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).casefold()
    for context in contexts:
        for interest in context.get('interests', []):
            private = [interest.get('requirement', '')]
            private += [e.get('fragment', '') for e in interest.get('evidence', [])]
            for text in private:
                if not isinstance(text, str):
                    continue
                if len(text.strip()) >= 25 and text.strip().casefold() in encoded:
                    raise ValueError('private source material echoed')
                for marker in re.findall(r'\b[A-ZА-ЯЁ][A-ZА-ЯЁ0-9_]{7,}\b', text):
                    if marker.casefold() in encoded:
                        raise ValueError('private source marker echoed')


def _observation(kind, cards, scope):
    text = ' '.join(str(c.get('analyst_brief', '')) for c in cards).casefold()
    def has(*parts):
        return all(part in text for part in parts)
    if kind == 'true_contradiction':
        return has('противореч', 'подтвержден', 'одинаков') and 'не подтвержден' not in text
    if kind == 'false_contradiction':
        return has('противореч', 'не подтвержден', 'период')
    if kind == 'units':
        return has('сопостав', 'единиц', 'измерен')
    if kind == 'criticality':
        return has('критичност', 'источник')
    if kind == 'missing_join':
        return has('отсутств', 'связ', 'оборудован')
    if kind == 'outage':
        return has('недоступ', 'не делать вывод')
    if kind == 'interval_target':
        methods = [c.get('metrics_method', '').casefold() for c in cards
                   if c.get('meeting_interest', {}).get('dimension_id') == 'repair_repeat_interval']
        return len(methods) == 1 and all(word in methods[0] for word in ('сопостав', 'интервал', 'повтор', 'эпизод', 'период'))
    if kind == 'scope':
        return all(scope.casefold() in c.get('scope', '').casefold() for c in cards)
    if kind == 'grounded':
        return all(not re.search(r'\d', ' '.join(str(c.get(f, '')) for f in pilot.TEXT_FIELDS)) for c in cards)
    if kind == 'periods':
        return any('период' in c.get('metrics_method', '').casefold() for c in cards)
    if kind == 'attribution':
        return all(c.get('meeting_interest', {}).get('attribution') in pilot.ATTRIBUTIONS for c in cards)
    if kind == 'source':
        return all(c.get('source_references') and c.get('meeting_interest', {}).get('sha256') for c in cards)
    raise ValueError('unknown scenario check')


def _write_audit(audit_dir, audit):
    path = audit_dir / ('scenario-comparison-' + uuid4().hex + '.json')
    data = json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as output:
        output.write(data)
    return str(path)


def evaluate_scenarios(*, baseline, candidate, scope, coverage_provider, generator,
                       private_verifier, scenarios, audit_dir, authority=None):
    """Run the comparison, retaining a safe partial audit if execution fails."""
    audit_dir = _outside_git(audit_dir)
    # Keep the inputs stable across the entire run; callbacks are untrusted and
    # may otherwise mutate caller-owned mappings between individual scenarios.
    baseline, candidate = copy.deepcopy(baseline), copy.deepcopy(candidate)
    attempts, records = [], []
    try:
        return _evaluate_scenarios(
            baseline=baseline, candidate=candidate, scope=scope,
            coverage_provider=coverage_provider, generator=generator,
            private_verifier=private_verifier, scenarios=scenarios,
            audit_dir=audit_dir, attempts=attempts, records=records, authority=authority)
    except Exception as exc:
        # Exception text and unvalidated model output may contain private data.
        _write_audit(audit_dir, {'activation_allowed': False, 'failure': type(exc).__name__,
                                 'attempts': attempts, 'scenarios': records})
        raise


def _evaluate_scenarios(*, baseline, candidate, scope, coverage_provider, generator,
                        private_verifier, scenarios, audit_dir, attempts, records, authority):
    """Evaluate fixed cases against actual pilot generation and private review.

    Each call to the injected generator receives the usual pilot request plus a
    synthetic ``scenario``. Audit is written exclusively outside Git. Raises on
    unavailable, altered or unsafe inputs; such errors must reject activation.
    """
    if scenarios != fixed_scenarios():
        raise ValueError('fixed scenario suite changed or incomplete')
    if generator is private_verifier or not callable(generator) or not callable(private_verifier):
        raise ValueError('separate generator and verifier required')
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError('scope required')
    for version in (baseline, candidate):
        if not isinstance(version, dict) or not version.get('knowledge_version'):
            raise ValueError('knowledge pin required')
    if baseline['knowledge_version'] == candidate['knowledge_version']:
        raise ValueError('distinct knowledge pins required')
    contexts = [baseline['interest_context'], candidate['interest_context']]
    components = {'generator': _callable_fingerprint(generator),
                  'verifier': _callable_fingerprint(private_verifier)}
    scripts = _script_fingerprint()
    try:
        frozen = pilot._coverage(coverage_provider())
    except Exception as exc:
        raise RuntimeError('coverage unavailable') from exc
    _privacy_check(frozen, contexts)

    def checked_coverage():
        try:
            current = pilot._coverage(coverage_provider())
        except Exception as exc:
            raise RuntimeError('coverage unavailable') from exc
        if current != frozen:
            raise ValueError('coverage or generation drift')
        return copy.deepcopy(frozen)

    for scenario in scenarios:
        sides = {}
        for name, version in (('baseline', baseline), ('candidate', candidate)):
            captured = {}
            def run(request):
                # Release pins are bookkeeping, never experimental treatment.
                request.pop('knowledge_version', None)
                request['scenario'] = {'id': scenario['id'], 'kind': scenario['kind'],
                                       'input': copy.deepcopy(scenario['input'])}
                captured['input'] = copy.deepcopy(request)
                _privacy_check(request, contexts)
                if name == 'candidate':
                    if _fixed_request(sides['baseline']['input']) != _fixed_request(request):
                        raise ValueError('baseline and candidate inputs differ')
                attempt = {'side': name, 'input': copy.deepcopy(request), 'output': None}
                attempts.append(attempt)
                output = generator(copy.deepcopy(request))
                _privacy_check(output, contexts)
                captured['output'] = copy.deepcopy(output)
                attempt['output'] = copy.deepcopy(output)
                return output
            reviews = []
            def verify(request):
                result = private_verifier(copy.deepcopy(request))
                _privacy_check(result, contexts)
                reviews.append({'input_sha256': _digest(request), 'output': copy.deepcopy(result)})
                return result
            result = pilot.generate_dimension_pilot(
                scope=scope, knowledge_version=version['knowledge_version'],
                interest_context=copy.deepcopy(version['interest_context']),
                dimension_projection=copy.deepcopy(version['dimension_projection']),
                coverage_provider=checked_coverage, generator=run,
                private_verifier=verify)
            if result.get('rejected_cards'):
                raise ValueError('pilot rejected generated cards')
            _privacy_check(result['cards'], contexts)
            observations = {check['kind']: _observation(check['kind'], result['cards'], scope)
                            for check in scenario['checks']}
            sides[name] = {'input': captured['input'], 'output': captured['output'],
                           'verified_cards': result['cards'], 'observations': observations,
                           'reviews': reviews}
            for field in ('input', 'output', 'verified_cards'):
                sides[name][field + '_sha256'] = _digest(sides[name][field])
        checks = scenario['checks']
        old, new = sides['baseline']['observations'], sides['candidate']['observations']
        mandatory_pass = all(new[c['kind']] for c in checks if c['mandatory'])
        degraded = any(old[c['kind']] and not new[c['kind']] for c in checks)
        critical = any(c['critical'] and old[c['kind']] and not new[c['kind']] for c in checks)
        improved = (any(c['target'] and not old[c['kind']] and new[c['kind']] for c in checks)
                    and _target_binding_changed(sides['baseline']['input'], sides['candidate']['input'],
                                                sides['baseline']['verified_cards'], sides['candidate']['verified_cards']))
        records.append({'id': scenario['id'], 'kind': scenario['kind'], 'input': copy.deepcopy(scenario['input']),
                        'expected': copy.deepcopy(scenario['expected']), 'checks': copy.deepcopy(checks),
                        **sides, 'comparison': {'mandatory_pass': mandatory_pass, 'degraded': degraded,
                                                'new_critical_error': critical, 'improved': improved}})
    checked_coverage()
    if components != {'generator': _callable_fingerprint(generator), 'verifier': _callable_fingerprint(private_verifier)} or scripts != _script_fingerprint():
        raise ValueError('comparison components changed')
    allowed = (all(r['comparison']['mandatory_pass'] and not r['comparison']['degraded']
                   and not r['comparison']['new_critical_error'] for r in records)
               and any(r['comparison']['improved'] for r in records))
    audit = {'activation_allowed': allowed, 'generation_id': frozen['generation']['run_id'],
             'coverage_digest': _digest(frozen), 'coverage': frozen, 'scenarios': records,
             'components_sha256': components, 'scripts_sha256': scripts,
             'knowledge_payload_sha256': {'baseline': _digest(baseline),
                                          'candidate': _digest(candidate)}}
    if authority is not None:
        if not isinstance(authority, AssessmentAuthority):
            raise ValueError('configured assessment authority required')
        audit['authority_hmac_sha256'] = authority._mac(audit)
    path = _write_audit(audit_dir, audit)
    return {**audit, 'audit_path': path}
