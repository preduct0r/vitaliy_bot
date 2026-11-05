"""Synthetic contracts for the independent hypothesis verifier; no live data."""

import copy
import json

import pytest

from test_hypotheses_seven_compatible import case
from tof_reliability.vitaly_bot import hypotheses_pilot as pilot


PRIVATE_SENTINEL = 'PRIVATE_MEETING_SECRET_SENTINEL'


def run(case, cards, verifier, *, on_generate=None):
    context, labels, audit, coverage, _ = copy.deepcopy(case)
    context['private_meeting_detail'] = PRIVATE_SENTINEL
    context['interests'][0]['private_meeting_detail'] = PRIVATE_SENTINEL
    calls = []

    def generator(request):
        calls.append(copy.deepcopy(request))
        if on_generate:
            on_generate(request)
        return copy.deepcopy(cards)

    result = pilot.generate_pilot(
        scope='насосы', knowledge_version='synthetic-k', interest_context=context,
        verified_labels=labels, verifier_audit=audit,
        coverage_provider=lambda: copy.deepcopy(coverage), generator=generator,
        private_verifier=verifier)
    return result, calls


def test_verifier_receives_exact_scores_and_bounded_scale_separate_from_evidence(case):
    cards = copy.deepcopy(case[4])
    cards[0]['ranking'] = {'utility': 0, 'feasibility': 1.5, 'interest': 5, 'novelty': 0}
    cards[1]['ranking'] = {'utility': 4, 'feasibility': 0, 'interest': 2, 'novelty': 0}
    requests = []

    def verifier(request):
        requests.append(copy.deepcopy(request))
        return {'supported': True, 'confidence': 'high'}

    result, calls = run(case, cards, verifier)
    assert len(calls) == 1 and len(requests) == len(result['cards']) == 7
    expected = {c['meeting_interest']['topic_id']: c['ranking'] for c in cards}
    for request in requests:
        # The ranking is model-supplied internal context, not source evidence.
        internal = request['ranking_context']
        assert internal['scores'] == expected[request['topic_id']]
        assert internal['scale'] == {'min': 0, 'max': 5}
        assert internal['source_fact'] is False
        assert 'ranking' not in request['card']
        assert PRIVATE_SENTINEL not in json.dumps(request, ensure_ascii=False)
        assert set(request['card']) <= set(pilot.TEXT_FIELDS + ('hard_questions',))


def test_generator_is_told_to_keep_ranking_out_of_user_facing_claims(case):
    seen = []
    result, _ = run(case, case[4], lambda _: {'supported': True, 'confidence': 'high'},
                    on_generate=lambda request: seen.append(request))
    assert len(result['cards']) == 7
    instructions = seen[0]['instructions'].casefold()
    assert ('ranking' in instructions or 'рейтинг' in instructions or 'ранжирован' in instructions)
    assert any(term in instructions for term in ('внутренн', 'служебн', 'internal'))
    assert any(term in instructions for term in ('процент', '%'))
    assert any(term in instructions for term in ('не включ', 'не указыва', 'не помещ', 'не вывод', 'не пиш'))


def test_ungrounded_ranking_claim_is_rejected_and_identifies_field(case):
    cards = copy.deepcopy(case[4])
    cards[0]['ranking'] = {'utility': 5, 'feasibility': 3, 'interest': 2, 'novelty': 0}
    cards[0]['decision'] = 'Все оценки оставлены нулевыми; выбрать проверку ремонта.'
    topic = cards[0]['meeting_interest']['topic_id']
    seen = []

    def verifier(request):
        seen.append(copy.deepcopy(request))
        if request['topic_id'] == topic:
            context = request.get('ranking_context', {})
            # Absence of the actual scores is not permission to invent them.
            scores = context.get('scores')
            assert scores == cards[0]['ranking']
            assert context['scale'] == {'min': 0, 'max': 5}
            return {'supported': False, 'confidence': 'high',
                    'reason': 'decision: заявление о нулевых оценках не подтверждено'}
        return {'supported': True, 'confidence': 'high'}

    result, calls = run(case, cards, verifier)
    assert len(calls) == 1 and len(seen) == 7
    assert len(result['cards']) == 6
    assert topic not in {c['meeting_interest']['topic_id'] for c in result['cards']}
    rejected = result['rejected_cards']
    assert len(rejected) == 1
    assert 'decision' in rejected[0]['verdict']['reason']
    assert PRIVATE_SENTINEL not in json.dumps(rejected, ensure_ascii=False)


@pytest.mark.parametrize('score', [-0.01, 5.01])
def test_out_of_bounds_ranking_never_reaches_verifier(case, score):
    cards = copy.deepcopy(case[4])
    cards[0]['ranking']['interest'] = score
    seen = []
    with pytest.raises(ValueError):
        run(case, cards, lambda request: seen.append(request) or
            {'supported': True, 'confidence': 'high'})
    assert not seen
