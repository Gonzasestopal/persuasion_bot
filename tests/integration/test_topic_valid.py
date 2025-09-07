# tests/integration/test_topic_quality_gate_prompt.py
import os
import re
import unicodedata

import pytest

from app.infra.llm import reset_llm_singleton_cache

pytestmark = pytest.mark.integration


# ----------------------------
# Helpers
# ----------------------------


def _norm_upper(s: str) -> str:
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace('’', "'")
    s = re.sub(r'\s+', ' ', s)
    return s.strip().upper()


def _last_bot_msg(resp_json):
    return resp_json['message'][-1]['message']


def looks_like_english(text: str) -> bool:
    t = _norm_upper(text)
    lex = [
        ' PLEASE ',
        ' TOPIC ',
        ' ARGUMENT ',
        ' EVIDENCE ',
        "LET'S KEEP ON TOPIC",
        "I CAN'T CHANGE",
        'PROVIDE A VALID, DEBATE-READY TOPIC',
    ]
    return any(w in t for w in lex)


def looks_like_spanish(text: str) -> bool:
    if any(ch in text for ch in ('¿', '¡')):
        return True
    if re.search(r'[áéíóúñÁÉÍÓÚÑ]', text or ''):
        return True
    t = _norm_upper(text)
    lex = [
        ' TEMA ',
        ' ARGUMENTO ',
        ' EVIDENCIA ',
        'MANTENGAMONOS EN EL TEMA',
        'PROPORCIONA UN TEMA VALIDO Y LISTO PARA DEBATE',
    ]
    return any(w in t for w in lex)


def assert_language(text: str, lang: str):
    if lang == 'en':
        assert looks_like_english(text), f'Expected English; got: {text!r}'
    elif lang == 'es':
        assert looks_like_spanish(text), f'Expected Spanish; got: {text!r}'
    else:
        raise AssertionError(f'Unsupported lang {lang!r}')


# ----------------------------
# Tests
# ----------------------------


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
def test_turn0_trivial_topic_en_shows_valid_topic_prompt(client):
    """
    Trivial topic ('hello') at TURN 0 should trigger the gate:
      - Contains exact EN one-liner: 'Please provide a valid, debate-ready topic.'
      - No stance opening phrases/tokens,
      - Mentions the topic string,
      - Exactly one question,
      - ≤80 words,
      - English-ish.
    """
    reset_llm_singleton_cache()

    topic = 'hello'
    start = f'Topic: {topic}. Side: PRO.'
    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    bot = _last_bot_msg(r1.json())
    assert isinstance(bot, str) and bot.strip()
    assert_language(bot, 'en')

    up = _norm_upper(bot)

    # Must include the exact EN gate one-liner
    expect_line = 'PLEASE PROVIDE A VALID, DEBATE-READY TOPIC.'
    assert expect_line in up, (
        f'Missing gate one-liner.\nWanted: {expect_line}\nGot: {bot!r}'
    )

    # Must NOT open with stance
    assert ' I WILL GLADLY TAKE ' not in up
    assert not re.search(r'\bPRO\b', up)
    assert not re.search(r'\bCON\b', up)

    # Should mention the given trivial topic at least once
    assert topic.upper() in up, f"Should reference topic '{topic}'. Got: {bot!r}"

    # Exactly one question, concise
    assert bot.count('?') == 1, f'Expected exactly one question. Got: {bot!r}'
    assert len(bot.split()) <= 80, f'Reply too long: {len(bot.split())} words'

    # No end markers
    assert 'MATCH CONCLUDED' not in up
    assert 'THE DEBATE HAS ALREADY ENDED' not in up
    assert '<DEBATE_ENDED>' not in up


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
def test_turn0_gibberish_topic_es_shows_valid_topic_prompt(client):
    """
    Gibberish topic in Spanish start should trigger the ES gate one-liner and stay Spanish.
    """
    reset_llm_singleton_cache()

    topic = 'asdf??? !!!'
    start = f'topic: {topic}. side: CON.'
    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    bot = _last_bot_msg(r1.json())
    assert isinstance(bot, str) and bot.strip()
    assert_language(bot, 'es')

    up = _norm_upper(bot)

    # Exact ES one-liner (diacritics normalized): "Por favor, proporciona un tema válido y listo para debate."
    expect_es = 'POR FAVOR, PROPORCIONA UN TEMA VALIDO Y LISTO PARA DEBATE.'
    assert expect_es in up, (
        f'Missing ES gate one-liner.\nWanted: {expect_es}\nGot: {bot!r}'
    )

    # Must NOT open with stance
    assert ' CON GUSTO TOMARE EL LADO ' not in up
    assert not re.search(r'\bPRO\b', up)
    assert not re.search(r'\bCON\b', up)

    # Mentions the original gibberish topic (normalized)
    assert _norm_upper(topic) in up, f"Should mention topic '{topic}'. Got: {bot!r}"

    # Exactly one question, concise
    assert bot.count('?') == 1, f'Expected exactly one question. Got: {bot!r}'
    assert len(bot.split()) <= 80, f'Reply too long: {len(bot.split())} words'


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
def test_turn0_good_topic_en_skips_gate_and_opens_normally(client):
    """
    Debate-ready topic should skip the gate:
      - Should NOT include the 'provide a valid topic' line,
      - Should include stance acknowledgement (opening phrase or 'CON' token),
      - Exactly one question,
      - ≤80 words,
      - English-ish.
    """
    reset_llm_singleton_cache()

    start = 'Topic: God exists. Side: CON.'
    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    bot = _last_bot_msg(r1.json())
    assert isinstance(bot, str) and bot.strip()
    assert_language(bot, 'en')

    up = _norm_upper(bot)

    # Should NOT contain the gate one-liner
    assert 'PROVIDE A VALID, DEBATE-READY TOPIC' not in up

    # Should look like a standard opening: either explicit phrase or stance token
    has_opening_phrase = ' I WILL GLADLY TAKE ' in up
    has_stance_token = bool(re.search(r'\bCON\b', up))
    assert has_opening_phrase or has_stance_token, (
        f'Expected stance/opening. Got: {bot!r}'
    )

    assert bot.count('?') == 1, f'Expected exactly one question. Got: {bot!r}'
    assert len(bot.split()) <= 80, f'Reply too long: {len(bot.split())} words'


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
def test_turn0_good_topic_es_skips_gate_and_opens_normally(client):
    """
    Spanish debate-ready topic should skip the gate and open normally.
    """
    reset_llm_singleton_cache()

    start = 'topic: Dios existe. side: PRO.'
    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    bot = _last_bot_msg(r1.json())
    assert isinstance(bot, str) and bot.strip()
    assert_language(bot, 'es')

    up = _norm_upper(bot)

    # No ES gate one-liner
    assert 'PROPORCIONA UN TEMA VALIDO Y LISTO PARA DEBATE' not in up

    has_opening_phrase = ' CON GUSTO TOMARE EL LADO ' in up
    has_stance_token = bool(re.search(r'\bPRO\b', up))
    assert has_opening_phrase or has_stance_token, (
        f'Expected stance/opening. Got: {bot!r}'
    )

    assert bot.count('?') == 1, f'Expected exactly one question. Got: {bot!r}'
    assert len(bot.split()) <= 80, f'Reply too long: {len(bot.split())} words'
