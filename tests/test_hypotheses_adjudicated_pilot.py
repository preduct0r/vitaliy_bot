"""Offline contracts for the audited projection; all sources are synthetic.

The original projection, candidates, and verdicts remain the audit record.
The pilot must replay adjudication before accepting its revised projection.
"""
import copy
import json
from pathlib import Path

import pytest

from test_hypotheses_conflict_adjudication import case as source_case
from test_hypotheses_dimensions_pilot import dimension_case, topic_case
from tof_reliability.vitaly_bot import hypotheses_pilot as pilot
from tof_reliability.vitaly_bot import hypotheses_topic_projection as topics


@pytest.fixture
def adjudication_case(tmp_path):
    # The other imported fixture writes source-0.txt in tmp_path too.
    audit_dir = tmp_path / 'adjudication'
    audit_dir.mkdir()
    return source_case.__wrapped__(audit_dir)


def revised(data):
    return topics.adjudicate_conflicts(**data)


def bind(data, projection):
    # Passing the complete private audit is essential: a self-consistent hash
    # alone cannot establish that the revised projection was adjudicated.
    return pilot.bind_expanded_projection(
        profile=data['profile'], source_index=data['source_index'], projection=projection,
        original_projection=data['original_projection'],
        original_verdicts=data['original_verdicts'],
        original_candidates=data['original_candidates'],
        adjudications=data['adjudications'])


def test_verified_revision_is_bound_without_mutating_history(adjudication_case):
    data = adjudication_case
    before = copy.deepcopy(data)
    projection = revised(data)
    assert projection['version'] == topics.ADJUDICATED_DIMENSION_VERSION
    assert not data['original_projection']['dimensions']
    result = bind(data, projection)
    row = result['dimensions'][0]
    assert row['independent_meeting_count'] == 2
    assert row['attribution_meeting_counts'] == {'third_party': 1, 'interpretation': 1}
    assert {e['source_id'] for e in row['evidence']} == {'source-0', 'source-1'}
    assert {e['source_id'] for r in result['unconfirmed'] for e in r['evidence']} >= {'source-2', 'source-3', 'source-4'}
    assert data == before


@pytest.mark.parametrize('damage', [
    'profile', 'index', 'original_projection', 'verdicts', 'candidates',
    'adjudication', 'prompt', 'trace', 'review_decision', 'fragment',
    'source_hash', 'source_bytes', 'projection_hash', 'projection_evidence',
    'projection_count', 'projection_version', 'conflict_unreviewed',
])
def test_revised_projection_fails_closed_on_forgery_or_unresolved_conflict(adjudication_case, damage):
    data = copy.deepcopy(adjudication_case)
    projection = revised(data)
    if damage == 'profile': data['profile']['interests'][0]['requirement'] += ' changed'
    elif damage == 'index': data['source_index']['version'] += '-changed'
    elif damage == 'original_projection': data['original_projection']['unconfirmed'].clear()
    elif damage == 'verdicts': data['original_verdicts'][0]['verdict']['confidence'] = 'low'
    elif damage == 'candidates': data['original_candidates'].pop()
    elif damage == 'adjudication': data['adjudications']['projection_sha256'] = '0' * 64
    elif damage == 'prompt': data['adjudications']['decisions'][0]['reviews'][0]['prompt'] += ' forged'
    elif damage == 'trace': data['adjudications']['decisions'][0]['reviews'][0]['trace'] += ' forged'
    elif damage == 'review_decision': data['adjudications']['decisions'][0]['reviews'][0]['decision'] = 'yes'
    elif damage == 'fragment': data['profile']['interests'][0]['evidence'][0]['fragment'] += ' forged'
    elif damage == 'source_hash': data['source_index']['sources'][0]['sha256'] = '0' * 64
    elif damage == 'source_bytes': Path(data['source_index']['sources'][0]['path']).write_text('forged', encoding='utf-8')
    elif damage == 'projection_hash': projection['adjudication_sha256'] = '0' * 64
    elif damage == 'projection_evidence': projection['dimensions'][0]['evidence'][0]['fragment'] += ' forged'
    elif damage == 'projection_count': projection['dimensions'][0]['independent_meeting_count'] += 1
    elif damage == 'projection_version': projection['version'] = 'unrecognized'
    elif damage == 'conflict_unreviewed': data['adjudications']['decisions'].pop()
    with pytest.raises(ValueError):
        bind(data, projection)


def test_recovered_evidence_flows_through_empty_scope_pilot(adjudication_case, dimension_case, topic_case, tmp_path):
    data = adjudication_case
    bound = bind(data, revised(data))
    context, coverage, base, cards = copy.deepcopy(dimension_case)
    # Preserve the pilot's synthetic corpus metadata while supplying a real
    # recovered, source-verified row from the independently audited revision.
    item = copy.deepcopy(data['profile']['interests'][0])
    sources = {s['id']: s for s in data['source_index']['sources']}
    for evidence in item['evidence']:
        source = sources[evidence['source_id']]
        evidence.update(date=source['date'], sha256=source['sha256'],
                        reference=f"{source['id']}:{evidence['locator']}")
    item['attributions'] = sorted({e['attribution'] for e in item['evidence']})
    context['interests'].append(item)
    base['dimensions'].append(copy.deepcopy(bound['dimensions'][0]))
    card = copy.deepcopy(topic_case[4][6])
    card['meeting_interest'] = {'interest_id': item['id'],
        'topic_id': 'techcard_coverage_verification',
        'dimension_id': 'repair_techcard_equipment_operations_coverage'}
    cards.append(card)
    for card in cards:
        card['scope'] = ''
    store = pilot.TrustedRequestStore(tmp_path / 'trusted.sqlite')
    trusted = store.create({'run_id': 'run', 'case_id': 'case', 'user_id': 'user', 'scope': ''})
    seen, checks = [], []
    def generator(request):
        seen.append(copy.deepcopy(request))
        return copy.deepcopy(cards)
    def verifier(request):
        checks.append(copy.deepcopy(request))
        return {'supported': True, 'confidence': 'high'}
    run = pilot.generate_dimension_pilot(
        scope='', knowledge_version='synthetic-k', interest_context=context,
        dimension_projection=base, coverage_provider=lambda: copy.deepcopy(coverage),
        generator=generator, private_verifier=verifier,
        trusted_context=trusted, trusted_store=store)
    assert run['scope'] == seen[0]['scope'] == ''
    assert 5 <= len(run['cards']) <= 7
    assert len(checks) == len(cards)
    assert all(c['scope'] and 'насос' not in c['scope'].casefold() for c in run['cards'])
    recovered = next(r for r in seen[0]['dimensions'] if r['dimension_id'] == card['meeting_interest']['dimension_id'])
    assert recovered['independent_meeting_count'] == 2
    assert recovered['attribution_meeting_counts'] == {'third_party': 1, 'interpretation': 1}
    assert all(c['missing_evidence'] for c in run['cards'])
    actual_routes = {r['capability_id']: r for r in seen[0]['coverage']['routes']}
    for route in coverage['routes']:
        assert actual_routes[route['capability_id']]['availability'] == route['availability']
        assert actual_routes[route['capability_id']]['missing_evidence'] == route['missing_evidence']
    assert actual_routes[pilot.NO_VERIFIED_ROUTE]['availability'] == 'unavailable'
    assert actual_routes[pilot.NO_VERIFIED_ROUTE]['missing_evidence']
    payload = json.dumps(seen, ensure_ascii=False)
    assert all(e['fragment'] not in payload for e in item['evidence'])
    assert 'sha256' not in payload and 'chars:' not in payload
    pilot.validate_pilot(run, coverage=coverage, interest_context=context,
                         trusted_context=trusted, trusted_store=store)
