"""Synthetic source-bound labels used by legacy pilot regression tests."""
from tof_reliability.vitaly_bot.hypotheses_topic_projection import _TOPICS

TOPIC, LABEL, _ = _TOPICS[0]

def projection(context):
    labels, audit = [], []
    for item in context['interests']:
        evidence = (item['evidence'] or [{'source_id': 'missing', 'locator': 'chars:0-1'}])[0]
        labels.append({'interest_id': item['id'], 'topic_id': TOPIC, 'description': LABEL})
        audit.append({'interest_id': item['id'], 'topic_id': TOPIC,
                      'source_id': evidence['source_id'], 'locator': evidence['locator'],
                      'verdict': {'supported': True, 'confidence': 'high'}})
    return {'verified_labels': labels, 'verifier_audit': audit,
            'private_verifier': lambda _: {'supported': True, 'confidence': 'high'}}

def project_cards(cards):
    for card in cards:
        meeting = card.get('meeting_interest')
        if isinstance(meeting, dict):
            meeting.setdefault('topic_id', TOPIC)
    return cards
