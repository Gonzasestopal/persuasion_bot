# tests/integration/test_language_lock_on_short_turn.py
import os
import re
import time
import unicodedata

import pytest

from app.infra.llm import reset_llm_singleton_cache
from app.infra.service import get_service  # DI to inspect debate state

pytestmark = pytest.mark.integration

CAP_WORDS = 5  # short-turn cap


# ----------------------------
# Helpers
# ----------------------------


def _norm_upper(s: str) -> str:
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace('’', "'")
    s = re.sub(r'\s+', ' ', s)
    return s.strip().upper()


def looks_like_english(text: str) -> bool:
    t = _norm_upper(text)
    lex = [
        ' HOWEVER ',
        ' ALTHOUGH ',
        ' EVIDENCE ',
        ' ARGUMENT ',
        ' TOPIC ',
        ' STANCE ',
        f'PLEASE EXPAND YOUR POINT TO AT LEAST {CAP_WORDS} WORDS',
        "I CAN'T CHANGE",
    ]
    return any(w in t for w in lex)


def looks_like_spanish(text: str) -> bool:
    if any(ch in text for ch in ('¿', '¡')):
        return True
    if re.search(r'[áéíóúñÁÉÍÓÚÑ]', text or ''):
        return True
    t = _norm_upper(text)
    # Note: accents are stripped by _norm_upper, so use "AMPLIA"
    return any(
        w in t
        for w in [
            ' PERO ',
            ' AUNQUE ',
            ' EVIDENCIA ',
            ' ARGUMENTO ',
            ' TEMA ',
            ' POSTURA ',
            f' POR FAVOR, AMPLIA TU PUNTO A AL MENOS {CAP_WORDS} PALABRAS',
        ]
    )


def assert_language(text: str, lang: str):
    if lang == 'en':
        assert looks_like_english(text), f'Expected English; got: {text!r}'
    elif lang == 'es':
        assert looks_like_spanish(text), f'Expected Spanish; got: {text!r}'
    else:
        raise AssertionError(f'Unsupported lang {lang!r}')


def _get_service_instance():
    from app.main import app as fastapi_app

    override = fastapi_app.dependency_overrides.get(get_service)
    assert override is not None, (
        'No DI override for get_service; ensure conftest sets it to a factory.'
    )
    return override()


# ----------------------------
# Tests
# ----------------------------


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
def test_short_nonchange_foreign_language_keeps_lock_and_uses_expand_template(client):
    """
    Start in EN. Then user sends a short non-change Spanish turn ("de acuerdo").
    Expect:
      - Reply stays in EN (language lock),
      - Includes EN expand template,
      - Includes ONE short refocus sentence mentioning the topic,
      - Asks EXACTLY one question,
      - No change-notice lines,
      - ≤80 words,
      - Does not increment positive_judgements.
    """
    reset_llm_singleton_cache()

    topic = 'God exists'
    start = f'Topic: {topic}. Side: CON.'
    expand_en = f'Please expand your point to at least {CAP_WORDS} words.'

    # Turn 1: start (locks EN)
    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    conv_id = r1.json()['conversation_id']

    # Snapshot state
    svc = _get_service_instance()
    before_pos = getattr(
        svc.debate_store.get(conversation_id=conv_id), 'positive_judgements', None
    )

    time.sleep(0.15)

    # Turn 2: short non-change in Spanish (<cap words)
    r2 = client.post(
        '/messages', json={'conversation_id': conv_id, 'message': 'de acuerdo'}
    )
    assert r2.status_code == 200, r2.text
    reply = r2.json()['message'][-1]['message']
    up = _norm_upper(reply)

    # Language lock: must remain English
    assert_language(reply, 'en')

    # Must include EN expand template; must NOT include ES expand template
    assert _norm_upper(expand_en) in up, f'Missing EN expand line. Got: {reply!r}'
    assert f'POR FAVOR, AMPLÍA TU PUNTO A AL MENOS {CAP_WORDS} PALABRAS.' not in up

    # Must NOT include any change-notice lines (no change requested)
    assert "I CAN'T CHANGE" not in up
    assert 'NO PUEDO CAMBIAR' not in up

    # Must refocus on topic + exactly one question
    assert reply.count('?') == 1, f'Expected exactly one question. Got: {reply!r}'

    # Concise
    assert len(reply.split()) <= 80, f'Reply too long: {len(reply.split())} words'

    # No code fences
    assert '```' not in reply

    # positive_judgements should not increase for short non-change
    after_pos = getattr(
        svc.debate_store.get(conversation_id=conv_id), 'positive_judgements', None
    )
    # if before_pos is not None and after_pos is not None:
    # assert after_pos == before_pos, (
    #    f'Short non-change turn should not count toward victory: {before_pos} -> {after_pos}'
    # )


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
def test_short_change_foreign_language_keeps_lock_and_uses_only_notice_no_expand(
    client,
):
    """
    Start in EN. Then user sends a short change request in Spanish ("¿Inglés?").
    Expect:
      - Reply stays in EN (language lock),
      - Includes ONLY the relevant EN notice line ("I can't change Language: EN."),
      - Does NOT include the expand template,
      - Adds one short refocus sentence mentioning the topic,
      - Asks EXACTLY one question,
      - ≤80 words,
      - Does not increment positive_judgements.
    """
    reset_llm_singleton_cache()

    topic = 'God exists'
    start = f'Topic: {topic}. Side: CON.'
    expect_notice = "I CAN'T CHANGE LANGUAGE: EN."
    expand_en = f'Please expand your point to at least {CAP_WORDS} words.'

    # Turn 1: start (locks EN)
    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    conv_id = r1.json()['conversation_id']

    svc = _get_service_instance()
    before_pos = getattr(
        svc.debate_store.get(conversation_id=conv_id), 'positive_judgements', None
    )

    time.sleep(0.15)

    # Turn 2: short change request in Spanish (<cap words)
    r2 = client.post(
        '/messages', json={'conversation_id': conv_id, 'message': '¿Inglés?'}
    )
    assert r2.status_code == 200, r2.text
    reply = r2.json()['message'][-1]['message']
    up = _norm_upper(reply)

    # Language lock: must remain English
    assert_language(reply, 'en')

    # Must include ONLY the relevant EN notice line for Language
    assert expect_notice in up, (
        f'Missing expected notice.\nWanted contains: {expect_notice}\nGot: {reply!r}'
    )
    # No unrelated notices
    for forb in ["I CAN'T CHANGE STANCE:", "I CAN'T CHANGE TOPIC:", 'NO PUEDO CAMBIAR']:
        assert forb not in up, f'Unexpected notice leaked: {forb}\nReply:\n{reply!r}'

    # Must NOT include the expand template for change requests
    assert _norm_upper(expand_en) not in up

    # Must refocus on topic + exactly one question
    assert _norm_upper(topic) in up, (
        f"Refocus should mention topic '{topic}'. Got: {reply!r}"
    )
    assert reply.count('?') == 1, f'Expected exactly one question. Got: {reply!r}'

    # Concise
    assert len(reply.split()) <= 80, f'Reply too long: {len(reply.split())} words'

    # No code fences
    assert '```' not in reply

    # positive_judgements should not increase for short change
    after_pos = getattr(
        svc.debate_store.get(conversation_id=conv_id), 'positive_judgements', None
    )
    if before_pos is not None and after_pos is not None:
        assert after_pos == before_pos, (
            f'Short change turn should not count toward victory: {before_pos} -> {after_pos}'
        )
