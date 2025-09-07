# tests/integration/test_topic_gate_language_aware.py
import os
import re
import unicodedata

import pytest

pytestmark = pytest.mark.integration


def _norm_upper(s: str) -> str:
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace('’', "'").replace('“', '"').replace('”', '"')
    return re.sub(r'\s+', ' ', s).strip().upper()


def _last_bot_msg(resp_json):
    return resp_json['message'][-1]['message']


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
def test_turn0_invalid_topic_en_starts_with_INVALID_and_localized_prompt(client):
    """
    Invalid English-ish topic should yield a reply that:
      - starts with 'INVALID:',
      - includes the EN one-liner quoting the topic,
      - has exactly one question,
      - ≤80 words,
      - does NOT start debating (no stance lines).
    """
    topic = 'asdf???'
    start = f'topic: {topic}, side: con'
    r = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r.status_code == 201, r.text
    bot = _last_bot_msg(r.json())
    assert isinstance(bot, str) and bot.strip()

    up = _norm_upper(bot)

    # Must start with INVALID:
    assert up.startswith('INVALID:'), (
        f"Expected reply to start with 'INVALID:'. Got: {bot!r}"
    )

    # Must quote the original topic and use the EN template opening
    expect_en = (
        f'INVALID: "{_norm_upper(topic)}" ISN\'T DEBATE-READY. '
        'PLEASE PROVIDE A VALID, DEBATE-READY TOPIC.'
    )
    assert expect_en in up, (
        f'Missing EN gate one-liner.\nWanted contains: {expect_en}\nGot: {bot!r}'
    )

    # Must NOT open with stance lines (no PRO/CON opener on gate)
    assert 'I WILL GLADLY TAKE THE' not in up
    assert not re.search(r'\bPRO\b', up) and not re.search(r'\bCON\b', up)

    # Must include exactly one question and be concise
    assert bot.count('?') == 1, f'Expected exactly one question. Got: {bot!r}'
    assert len(bot.split()) <= 80, f'Reply too long: {len(bot.split())} words'


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
def test_turn0_invalid_topic_es_starts_with_INVALID_and_localized_prompt(client):
    """
    Invalid Spanish topic should yield a reply that:
      - starts with 'INVALID:',
      - includes the ES one-liner quoting the topic,
      - has exactly one question,
      - ≤80 words,
      - does NOT start debating (no stance lines).
    """
    topic = 'hola'
    start = f'topic: {topic}, side: con'
    r = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r.status_code == 201, r.text
    bot = _last_bot_msg(r.json())
    assert isinstance(bot, str) and bot.strip()

    up = _norm_upper(bot)

    # Must start with INVALID:
    assert up.startswith('INVALID:'), (
        f"Expected reply to start with 'INVALID:'. Got: {bot!r}"
    )

    # Must quote the original topic and use the ES template opening
    expect_es = (
        f'INVALID: "{_norm_upper(topic)}" NO ES UN TEMA LISTO PARA DEBATE. '
        'POR FAVOR, PROPORCIONA UN TEMA VALIDO Y LISTO PARA DEBATE.'
    )
    assert expect_es in up, (
        f'Missing ES gate one-liner.\nWanted contains: {expect_es}\nGot: {bot!r}'
    )

    # Must NOT open with stance lines (no PRO/CON opener on gate)
    assert 'CON GUSTO TOMARE EL LADO' not in up
    assert not re.search(r'\bPRO\b', up) and not re.search(r'\bCON\b', up)

    # Must include exactly one question and be concise
    assert bot.count('?') == 1, f'Expected exactly one question. Got: {bot!r}'
    assert len(bot.split()) <= 80, f'Reply too long: {len(bot.split())} words'


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
def test_turn0_valid_topic_skips_gate_and_starts_debate(client):
    """
    A valid debate-ready topic should NOT trigger the gate.
    We assert that the 'INVALID:' prefix is absent and the reply resembles a normal opener.
    """
    start = 'topic: God exists, side: con'
    r = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r.status_code == 201, r.text
    bot = _last_bot_msg(r.json())
    up = _norm_upper(bot)

    # Gate prefix must NOT appear
    assert not up.startswith('INVALID:'), f'Did not expect INVALID gate. Got: {bot!r}'

    # Gate lines must NOT appear
    assert "ISN'T DEBATE-READY" not in up
    assert 'NO ES UN TEMA LISTO PARA DEBATE' not in up

    # Reasonable sign of a normal opening (don’t hardcode phrasing; keep it loose)
    assert any(
        needle in up
        for needle in [
            'I WILL GLADLY TAKE THE',  # EN opener pattern
            'CON GUSTO TOMARE EL LADO',  # ES opener pattern
            'ASSUME THE CON STANCE',  # fallback phrasing
        ]
    ), f'Expected a normal opening, got: {bot!r}'
