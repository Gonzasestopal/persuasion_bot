# tests/test_language_and_length.py
import os
import re
import time
import unicodedata

import pytest

from app.infra.llm import reset_llm_singleton_cache

pytestmark = pytest.mark.integration


# ----------------------------
# Soft language heuristics
# ----------------------------


def _norm(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r'\s+', ' ', s)
    return s.strip().lower()


def looks_like_spanish(text: str) -> bool:
    # Fast signals: Spanish punctuation / diacritics
    if any(ch in text for ch in ('¿', '¡')):
        return True
    if re.search(r'[áéíóúñÁÉÍÓÚÑ]', text):
        return True

    t = _norm(text)
    # Lexical cues (any one is enough)
    lex = [
        'pero',
        'aunque',
        'todavia',
        'evidencia',
        'causalidad',
        'postura',
        'mecanismo',
        'objecion',
        'productividad',
        'remoto',
        'traslado',
        'enfoque',
        'asincronico',
        'asincrono',
        'asincronia',
        'autonomia',
        'argumento',
        'mantengamonos en el tema',
        'tema',
        'idioma',
        '¿como',
        'como crees',
        'influye',
        'caracter',
    ]
    if any(tok in t for tok in lex):
        return True

    # Function words fallback: need 3+
    fun = [' el ', ' la ', ' de ', ' que ', ' y ', ' en ']
    return sum(f in f' {t} ' for f in fun) >= 3


def looks_like_english(text: str) -> bool:
    t = _norm(text)
    # Lexical cues (any one is enough)
    lex = [
        'however',
        'although',
        'evidence',
        'causality',
        'stance',
        'topic',
        'reason',
        'reasons',
        'believe',
        'support',
        'existence',
        'hidden',
        'hiddenness',
        'nonresistant',
        'silence',
        'argument',
        "i can't change",
        "let's focus",
        'keep on topic',
        'character',
        'discipline',
        'please',
        'switch',
    ]
    if any(tok in t for tok in lex):
        return True

    # Function words fallback: need 3+
    fun = [' the ', ' and ', ' is ', ' are ', ' not ']
    return sum(f in f' {t} ' for f in fun) >= 3


def assert_language(text: str, lang: str):
    if lang == 'es':
        assert looks_like_spanish(text), f'Expected Spanish; got: {text!r}'
    elif lang == 'en':
        assert looks_like_english(text), f'Expected English; got: {text!r}'
    else:
        raise AssertionError(f'Unsupported lang {lang!r}')


# ----------------------------
# Tests
# ----------------------------


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OPENAI_API_KEY not set'
)
def test_one_probing_question_and_length_limit(client):
    r1 = client.post(
        '/messages',
        json={
            'conversation_id': None,
            'message': 'Topic: Sports build character. Side: PRO.',
        },
    )
    assert r1.status_code == 201
    conv_id = r1.json()['conversation_id']

    r2 = client.post(
        '/messages',
        json={'conversation_id': conv_id, 'message': 'Discipline is key; respond.'},
    )
    assert r2.status_code == 200
    a2 = r2.json()['message'][-1]['message']
    qm = a2.count('?')
    assert qm == 1, f'Expected exactly 1 question mark, got {qm}: {a2!r}'
    assert len(a2.split()) <= 80


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
@pytest.mark.parametrize(
    'start_message, lang, second_msg',
    [
        # Spanish main language
        (
            'topic: El deporte forma carácter. side: PRO.',
            'es',
            '¿Puedes cambiar al lado CON?',
        ),
        # English main language
        (
            'Topic: Sports build character. Side: PRO.',
            'en',
            'Can you switch to the CON side?',
        ),
    ],
)
def test_real_llm_respects_main_language(client, start_message, lang, second_msg):
    """
    Ensures the bot replies in the main language implied/declared by the user's first turn.
    Keeps the same conversation_id across turns and verifies language on every bot reply.
    """
    reset_llm_singleton_cache()

    # ---- Turn 1: start conversation ----
    r1 = client.post(
        '/messages', json={'conversation_id': None, 'message': start_message}
    )
    assert r1.status_code == 201, r1.text
    data1 = r1.json()
    conv_id = data1['conversation_id']

    # Bot reply #1
    first_bot_msg = data1['message'][-1]['message']
    assert isinstance(first_bot_msg, str) and first_bot_msg.strip()
    assert_language(first_bot_msg, lang)

    time.sleep(0.2)  # tiny pause for provider rate limits

    # ---- Turn 2: continue same conversation ----
    r2 = client.post(
        '/messages', json={'conversation_id': conv_id, 'message': second_msg}
    )
    assert r2.status_code == 200, r2.text
    data2 = r2.json()

    # Bot reply #2
    second_bot_msg = data2['message'][-1]['message']
    assert isinstance(second_bot_msg, str) and second_bot_msg.strip()
    assert_language(second_bot_msg, lang)
