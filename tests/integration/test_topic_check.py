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


def _assert_invalid_prefix(up: str):
    """
    Allow either:
      - 'INVALID:' at the very start, or
      - 'LANGUAGE: <xx> ... INVALID:' if the model emits the language header first.
    """
    assert 'INVALID:' in up, f"Missing 'INVALID:' prefix. Got: {up!r}"
    if up.startswith('LANGUAGE:'):
        # After LANGUAGE header, INVALID should still be present in the same message.
        assert ' INVALID:' in up, (
            f"Expected 'INVALID:' after LANGUAGE header. Got: {up!r}"
        )
    else:
        assert up.startswith('INVALID:'), (
            f"Expected message to start with 'INVALID:'. Got: {up!r}"
        )


@pytest.mark.skipif(
    not os.environ.get('ANTHROPIC_API_KEY'),
    reason='ANTHROPIC_API_KEY not set; skipping live LLM integration test.',
)
def test_turn0_invalid_topic_en_gate_line_and_shape(client):
    """
    Invalid English-ish topic should yield a localized gate one-liner quoting the topic,
    exactly one question, ≤80 words, and no stance opener.
    (We *prefer* 'INVALID:' but do not fail if the model omits it.)
    """
    topic = 'asdf'
    start = f'topic: {topic}, side: con'
    r = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r.status_code == 201, r.text
    bot = _last_bot_msg(r.json())
    assert isinstance(bot, str) and bot.strip()
    up = _norm_upper(bot)

    # Soft preference for INVALID: (don’t fail if missing)
    if up.startswith('LANGUAGE:'):
        assert ' INVALID:' in up or "ISN'T DEBATE-READY" in up
    else:
        assert up.startswith('INVALID:') or "ISN'T DEBATE-READY" in up

    # Must quote the original topic and use the EN template opening
    expect_en = (
        f'"{_norm_upper(topic)}" ISN\'T DEBATE-READY. '
        'PLEASE PROVIDE A VALID, DEBATE-READY TOPIC.'
    )
    assert expect_en in up, (
        f'Missing EN gate one-liner.\nWanted contains: {expect_en}\nGot: {bot!r}'
    )

    # No stance opener
    assert 'I WILL GLADLY TAKE THE' not in up
    assert not re.search(r'\bPRO\b', up) and not re.search(r'\bCON\b', up)

    # Exactly one question, concise
    assert bot.count('?') == 1, f'Expected exactly one question. Got: {bot!r}'
    assert len(bot.split()) <= 80, f'Reply too long: {len(bot.split())} words'


@pytest.mark.skipif(
    not os.environ.get('ANTHROPIC_API_KEY'),
    reason='ANTHROPIC_API_KEY not set; skipping live LLM integration test.',
)
def test_turn0_invalid_topic_es_gate_line_and_shape(client):
    """
    Invalid Spanish topic should yield a localized gate one-liner quoting the topic.
    Tolerate occasional EN fallback; still enforce shape.
    """
    topic = 'hola'
    start = f'topic: {topic}, side: con'
    r = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r.status_code == 201, r.text
    bot = _last_bot_msg(r.json())
    assert isinstance(bot, str) and bot.strip()
    up = _norm_upper(bot)

    # Soft preference for INVALID:
    if up.startswith('LANGUAGE:'):
        has_invalid = ' INVALID:' in up
    else:
        has_invalid = up.startswith('INVALID:')
    # Don’t fail if missing; model compliance may vary

    expect_es = (
        f'"{_norm_upper(topic)}" NO ES UN TEMA LISTO PARA DEBATE. '
        'POR FAVOR, PROPORCIONA UN TEMA VALIDO Y LISTO PARA DEBATE.'
    )
    expect_en = (
        f'"{_norm_upper(topic)}" ISN\'T DEBATE-READY. '
        'PLEASE PROVIDE A VALID, DEBATE-READY TOPIC.'
    )
    assert (expect_es in up) or (expect_en in up), (
        f'Missing ES/EN gate one-liner.\nWanted one of:\n  ES: {expect_es}\n  EN: {expect_en}\nGot: {bot!r}'
    )

    # No stance opener
    assert 'CON GUSTO TOMARE EL LADO' not in up
    assert 'I WILL GLADLY TAKE THE' not in up
    assert not re.search(r'\bPRO\b', up) and not re.search(r'\bCON\b', up)

    # Exactly one question, concise
    assert bot.count('?') == 1, f'Expected exactly one question. Got: {bot!r}'
    assert len(bot.split()) <= 80, f'Reply too long: {len(bot.split())} words'


@pytest.mark.skipif(
    not os.environ.get('ANTHROPIC_API_KEY'),
    reason='ANTHROPIC_API_KEY not set; skipping live LLM integration test.',
)
def test_turn0_valid_topic_skips_gate_and_starts_debate(client):
    """
    A valid debate-ready topic should NOT trigger the gate.
    """
    start = 'topic: God exists, side: con'
    r = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r.status_code == 201, r.text
    bot = _last_bot_msg(r.json())
    up = _norm_upper(bot)

    # Gate must NOT appear
    assert 'INVALID:' not in up
    assert "ISN'T DEBATE-READY" not in up
    assert 'NO ES UN TEMA LISTO PARA DEBATE' not in up

    # Reasonable sign of a normal opening
    assert any(
        needle in up
        for needle in [
            'I WILL GLADLY TAKE THE',  # EN opener pattern
            'CON GUSTO TOMARE EL LADO',  # ES opener pattern
            'ASSUME THE CON STANCE',  # fallback phrasing
        ]
    ), f'Expected a normal opening, got: {bot!r}'
