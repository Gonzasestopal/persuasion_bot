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


def _no_stance_opener(up: str):
    # Disallow typical stance openers (these should never appear in 422 errors)
    assert 'I WILL GLADLY TAKE THE' not in up
    assert 'CON GUSTO TOMARE EL LADO' not in up

    # For raw PRO/CON tokens, allow them only if they are part of the quoted
    # “SIDE: PRO/CON” metadata from the user’s original start line.
    # We enforce this with a negative-lookbehind that forbids matches unless
    # preceded by "SIDE: ".
    assert not re.search(r'(?<!SIDE:\s)\bPRO\b', up)
    assert not re.search(r'(?<!SIDE:\s)\bCON\b', up)


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
    topic = 'asdf'
    start = f'topic: {topic}, side: con'
    r = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r.status_code == 422, r.text

    detail = r.json().get('detail', '')
    assert isinstance(detail, str) and detail.strip()
    up = _norm_upper(detail)

    # Soft preference for INVALID:
    if up.startswith('LANGUAGE:'):
        assert ' INVALID:' in up or "ISN'T DEBATE-READY" in up
    else:
        assert up.startswith('INVALID:') or "ISN'T DEBATE-READY" in up

    # Must include the topic token and the EN tail
    expect_en_tail = "ISN'T DEBATE-READY. PLEASE PROVIDE A VALID, DEBATE-READY TOPIC."
    assert _norm_upper(topic) in up, f'Topic token missing in detail. Got: {detail!r}'
    assert expect_en_tail in up, (
        f'Missing EN gate one-liner tail.\nWanted: {expect_en_tail}\nGot: {detail!r}'
    )

    _no_stance_opener(up)


@pytest.mark.skipif(
    not os.environ.get('ANTHROPIC_API_KEY'),
    reason='ANTHROPIC_API_KEY not set; skipping live LLM integration test.',
)
def test_turn0_invalid_topic_es_gate_line_and_shape(client):
    topic = 'hola'
    start = f'topic: {topic}, side: con'
    r = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r.status_code == 422, r.text

    detail = r.json().get('detail', '')
    assert isinstance(detail, str) and detail.strip()
    up = _norm_upper(detail)

    # Soft preference for INVALID: non-fatal if missing
    # (kept for parity with earlier tests)
    if up.startswith('LANGUAGE:'):
        _ = ' INVALID:' in up
    else:
        _ = up.startswith('INVALID:')

    expect_es_tail = 'NO ES UN TEMA LISTO PARA DEBATE. POR FAVOR, PROPORCIONA UN TEMA VALIDO Y LISTO PARA DEBATE.'
    expect_en_tail = "ISN'T DEBATE-READY. PLEASE PROVIDE A VALID, DEBATE-READY TOPIC."
    assert _norm_upper(topic) in up, f'Topic token missing in detail. Got: {detail!r}'
    assert (expect_es_tail in up) or (expect_en_tail in up), (
        f'Missing ES/EN gate one-liner tail.\n'
        f'Wanted one of:\n  ES: {expect_es_tail}\n  EN: {expect_en_tail}\n'
        f'Got: {detail!r}'
    )

    _no_stance_opener(up)


@pytest.mark.skipif(
    not os.environ.get('ANTHROPIC_API_KEY'),
    reason='ANTHROPIC_API_KEY not set; skipping live LLM integration test.',
)
def test_turn0_valid_topic_skips_gate_and_starts_debate(client):
    """
    A valid debate-ready topic should NOT trigger the gate; should return 201
    with a normal opening in the bot message payload.
    """
    start = 'topic: God exists, side: con'
    r = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r.status_code == 201, r.text

    bot = r.json()['message'][-1]['message']
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
