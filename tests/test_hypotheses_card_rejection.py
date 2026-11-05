"""Seven synthetic source-backed directions with per-card private verification."""

import copy
import json

import pytest

from test_hypotheses_seven_compatible import DIRECTIONS, case
from tof_reliability.vitaly_bot import hypotheses_pilot as pilot


REJECTED = 'equipment_criticality'
REASON = 'Предположение о критичности оборудования не подтверждено источниками.'


def generate(case, verifier, *, cards=None, tmp_path=None):
    context, labels, audit, coverage, original = case
    calls = []

    def generator(model_context):
        calls.append(copy.deepcopy(model_context))
        return copy.deepcopy(original if cards is None else cards)

    options = {}
    if tmp_path is not None:
        store = pilot.TrustedRequestStore(tmp_path / 'trusted.sqlite')
        trusted = store.create({'run_id': 'run-synthetic', 'case_id': 'case-synthetic',
                                'user_id': 'user-synthetic', 'scope': 'насосы'})
        options = {'trusted_context': trusted, 'trusted_store': store}
    result = pilot.generate_pilot(
        scope='насосы', knowledge_version='k1', interest_context=context,
        verified_labels=labels, verifier_audit=audit,
        coverage_provider=lambda: copy.deepcopy(coverage), generator=generator,
        private_verifier=verifier, **options)
    return result, calls, options


def verifier_rejecting(topics):
    calls = []

    def verify(request):
        calls.append(copy.deepcopy(request))
        if request['topic_id'] in topics:
            return {'supported': False, 'confidence': 'high', 'reason': REASON}
        return {'supported': True, 'confidence': 'high'}

    return verify, calls


def test_one_semantic_rejection_keeps_six_unchanged_cards_and_private_diagnostic(case, tmp_path):
    approve, _ = verifier_rejecting(set())
    baseline, _, _ = generate(case, approve)
    verifier, checks = verifier_rejecting({REJECTED})
    run, model_calls, trusted = generate(case, verifier, tmp_path=tmp_path)

    assert len(model_calls) == 1  # no retry to replace an inconvenient verdict
    assert len(checks) == 7
    assert len({call['topic_id'] for call in checks}) == 7
    assert len(run['cards']) == 6
    assert [card['number'] for card in run['cards']] == list(range(1, 7))
    expected = {c['meeting_interest']['topic_id']: c for c in baseline['cards']
                if c['meeting_interest']['topic_id'] != REJECTED}
    for card in run['cards']:
        original = copy.deepcopy(expected[card['meeting_interest']['topic_id']])
        original['number'] = card['number']
        assert card == original  # includes all scores, claims, evidence, and bindings
        assert card['meeting_interest']['sha256']
        assert card['meeting_interest']['reference']
    assert len(expected) == len({c['meeting_interest']['topic_id'] for c in run['cards']})
    assert REJECTED not in {c['meeting_interest']['topic_id'] for c in run['cards']}
    justification = run['breadth_justification']
    assert set(justification['eligible_topic_ids']) == {direction[0] for direction in DIRECTIONS}
    assert justification['reason'].strip()
    # Private diagnostics retain the exact rejection reason and the rejected card,
    # while the breadth notice explicitly explains that an idea was omitted.
    diagnostic = json.dumps(run, ensure_ascii=False)
    assert REASON in diagnostic
    assert case[4][5]['management_question'] in diagnostic
    assert any(word in justification['reason'].casefold()
               for word in ('отклон', 'исключ', 'пропущ', 'опущ', 'rejected', 'omitted'))
    pilot.validate_pilot(run, coverage=case[3], interest_context=case[0], **trusted)


def test_three_semantic_rejections_fail_below_five_verified_cards(case):
    verifier, checks = verifier_rejecting({direction[0] for direction in DIRECTIONS[:3]})
    with pytest.raises(ValueError, match='(?i)fewer|insufficient|five|пяти|недостат'):
        generate(case, verifier)
    assert len({call['topic_id'] for call in checks}) >= 3


@pytest.mark.parametrize('bad', [RuntimeError('unavailable'), TimeoutError('timeout'),
                                 None, {}, {'supported': 'false', 'confidence': 'high'},
                                 {'supported': True, 'confidence': 'low'}])
def test_verifier_failure_or_uncertainty_never_counts_as_semantic_rejection(case, bad):
    def verify(request):
        if request['topic_id'] == REJECTED:
            if isinstance(bad, Exception):
                raise bad
            return bad
        return {'supported': True, 'confidence': 'high'}

    with pytest.raises(ValueError):
        generate(case, verify)


def test_all_seven_verified_are_kept(case):
    verifier, checks = verifier_rejecting(set())
    run, calls, _ = generate(case, verifier)
    assert len(calls) == 1
    assert len(checks) == len(run['cards']) == 7
    assert len({card['meeting_interest']['topic_id'] for card in run['cards']}) == 7
    pilot.validate_pilot(run, coverage=case[3], interest_context=case[0])


def test_duplicate_topics_cannot_replace_rejected_direction(case):
    cards = copy.deepcopy(case[4])
    cards[-1] = copy.deepcopy(cards[0])
    cards[-1]['number'] = 7
    verifier, _ = verifier_rejecting({REJECTED})
    with pytest.raises(ValueError):
        generate(case, verifier, cards=cards)


def test_rejection_does_not_bypass_fact_guard_on_another_card(case):
    cards = copy.deepcopy(case[4])
    cards[-1]['decision'] = 'Простой составил 987654 часов.'
    verifier, _ = verifier_rejecting({REJECTED})
    with pytest.raises(ValueError):
        generate(case, verifier, cards=cards)
