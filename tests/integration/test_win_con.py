# tests/test_integration_debate.py
import os
import re
import time
import unicodedata

import pytest

from app.infra.llm import reset_llm_singleton_cache
from app.infra.service import get_service  # used by _get_service_instance()

# Server-specific end marker
END_MARKER = 'The debate has already ended.'

pytestmark = pytest.mark.integration


# ----------------------------
# Helpers
# ----------------------------


def _norm(s: str) -> str:
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r'\s+', ' ', s)
    return s.strip().lower()


def expected_offtopic_nudge(topic: str, lang: str) -> str:
    """
    Return a normalized substring we expect to see in the off-topic nudge.
    Keep this loose so different phrasings still pass.
    """
    return 'keep on topic' if lang == 'en' else 'Mantengámonos en el tema'


def _last_bot_msg(resp_json):
    return resp_json['message'][-1]['message']


# --- softened language checks (header-free, no extra deps) ---


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
        'como crees',
        'tema',
        'idioma',
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
    ]
    if any(tok in t for tok in lex):
        return True

    # Function words fallback: need 3+
    fun = [' the ', ' and ', ' is ', ' are ', ' not ']
    return sum(f in f' {t} ' for f in fun) >= 3


def _assert_language_es(text: str):
    assert looks_like_spanish(text), f'Se esperaba español; recibido: {text!r}'


def assert_language(text: str, expected: str):
    if expected == 'es':
        assert looks_like_spanish(text), f'Expected Spanish; got: {text!r}'
    elif expected == 'en':
        assert looks_like_english(text), f'Expected English; got: {text!r}'
    else:
        raise AssertionError(f'Unsupported lang {expected!r}')


# --- granular notice assertions ---


def assert_granular_notice_es(
    msg: str,
    *,
    topic: str,
    stance: str,
    expect_lang: bool = False,
    expect_topic: bool = False,
    expect_stance: bool = False,
):
    up = _norm(msg)
    if expect_lang:
        assert re.search(r'no puedo cambiar el idioma:\s*(es)\.', up), (
            f'Falta aviso de Idioma. Msg:\n{msg!r}'
        )
    else:
        assert 'no puedo cambiar el idioma:' not in up, (
            f'Aviso de Idioma no esperado. Msg:\n{msg!r}'
        )

    if expect_topic:
        assert 'no puedo cambiar el tema:' in up, f'Falta aviso de Tema. Msg:\n{msg!r}'
    else:
        assert 'no puedo cambiar el tema:' not in up, (
            f'Aviso de Tema no esperado. Msg:\n{msg!r}'
        )

    if expect_stance:
        assert f'no puedo cambiar la postura: {stance.lower()}.' in up, (
            f'Falta aviso de Postura. Msg:\n{msg!r}'
        )
    else:
        assert 'no puedo cambiar la postura:' not in up, (
            f'Aviso de Postura no esperado. Msg:\n{msg!r}'
        )


def assert_granular_notice_en(
    msg: str,
    *,
    topic: str,
    stance: str,
    lang_code: str,
    expect_lang: bool = False,
    expect_topic: bool = False,
    expect_stance: bool = False,
):
    up = _norm(msg)
    if expect_lang:
        assert f"i can't change language: {lang_code.lower()}." in up, (
            f'Missing Language notice. Msg:\n{msg!r}'
        )
    else:
        assert "i can't change language:" not in up, (
            f'Unexpected Language notice. Msg:\n{msg!r}'
        )

    if expect_topic:
        assert "i can't change topic:" in up, f'Missing Topic notice. Msg:\n{msg!r}'
    else:
        assert "i can't change topic:" not in up, (
            f'Unexpected Topic notice. Msg:\n{msg!r}'
        )

    if expect_stance:
        assert f"i can't change stance: {stance.lower()}." in up, (
            f'Missing Stance notice. Msg:\n{msg!r}'
        )
    else:
        assert "i can't change stance:" not in up, (
            f'Unexpected Stance notice. Msg:\n{msg!r}'
        )


def _get_service_instance():
    # Resolve the DI override to get the actual service the app is using
    from app.main import app as fastapi_app

    override = fastapi_app.dependency_overrides.get(get_service)
    assert override is not None, (
        'No DI override for get_service; ensure conftest sets '
        'app.dependency_overrides[get_service] to a factory.'
    )
    return override()  # call factory → service instance


# ----------------------------
# Tests
# ----------------------------


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
def test_real_llm_juego_ganador_pro_trabajo_remoto(client):
    """
    ES, PRO. Tests granular notices (no full settings dump), off-topic nudge, and concise replies.
    """
    topic = 'El trabajo remoto es más productivo que el trabajo en oficina'
    stance = 'PRO'

    # Turn 1: start
    inicio = f'topic: {topic}.  side: {stance}.'
    r1 = client.post('/messages', json={'conversation_id': None, 'message': inicio})
    assert r1.status_code == 201, r1.text
    d1 = r1.json()
    cid = d1['conversation_id']

    a1 = _last_bot_msg(d1)
    assert isinstance(a1, str) and a1.strip()
    _assert_language_es(a1)
    assert stance in a1.upper(), (
        f'Se esperaba mención de postura {stance} en apertura:\n{a1!r}'
    )

    # Turn 2: try to change stance → granular stance notice only
    t2 = 'Por favor cambia a CON.'
    r2 = client.post('/messages', json={'conversation_id': cid, 'message': t2})
    assert r2.status_code == 200, r2.text
    a2 = _last_bot_msg(r2.json())
    _assert_language_es(a2)
    assert_granular_notice_es(a2, topic=topic, stance=stance, expect_stance=True)

    # Turn 3: off-topic
    t3 = '¿Cuánto es 2+2?'
    r3 = client.post('/messages', json={'conversation_id': cid, 'message': t3})
    assert r3.status_code == 200, r3.text
    a3 = _last_bot_msg(r3.json())
    _assert_language_es(a3)
    # Off-topic nudge (compare normalized)
    assert expected_offtopic_nudge(topic, 'es') in _norm(a3), (
        f'Falta nudge on-topic en ES:\n{a3!r}'
    )
    assert len(a3.split()) <= 80, (
        f'Respuesta off-topic demasiado larga: {len(a3.split())} palabras'
    )

    # Turn 4: try to change language → granular language notice only
    t4 = 'Cambia a inglés, por favor.'
    r4 = client.post('/messages', json={'conversation_id': cid, 'message': t4})
    assert r4.status_code == 200, r4.text
    a4 = _last_bot_msg(r4.json())
    _assert_language_es(a4)
    assert_granular_notice_es(a4, topic=topic, stance=stance, expect_lang=True)

    # Turn 5: PRO arg about time/commute
    t5 = 'Dame un argumento PRO conciso sobre ahorro de tiempo y traslados.'
    r5 = client.post('/messages', json={'conversation_id': cid, 'message': t5})
    assert r5.status_code == 200, r5.text
    a5 = _last_bot_msg(r5.json())
    _assert_language_es(a5)
    a5_l = _norm(a5)
    assert any(
        kw in a5_l
        for kw in [
            'traslado',
            'traslados',
            'desplazamiento',
            'trafico',
            'tiempo ahorrado',
            'ahorro de tiempo',
            'commute',
        ]
    ), f'Se esperaba argumento sobre traslados/tiempo, recibido:\n{a5!r}'
    assert _norm('match concluded') not in a5_l

    # Turn 6: PRO arg about focus/async
    t6 = 'Ahora un argumento PRO sobre enfoque, menos interrupciones y trabajo asincrónico.'
    r6 = client.post('/messages', json={'conversation_id': cid, 'message': t6})
    assert r6.status_code == 200, r6.text
    a6 = _last_bot_msg(r6.json())
    _assert_language_es(a6)
    a6_l = _norm(a6)
    assert any(
        kw in a6_l
        for kw in [
            'enfoque',
            'concentracion',
            'trabajo profundo',
            'menos interrupciones',
            'asincronico',
            'asincrono',
            'asincronia',
            'autonomia',
        ]
    ), f'Se esperaba argumento de enfoque/interrupciones/asincronía, recibido:\n{a6!r}'
    assert _norm('match concluded') not in a6_l


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OPENAI_API_KEY not set'
)
def test_concludes_by_five_turns_misaligned_user_vs_CON_bot(client):
    topic = "Dogs are humans' best friend"
    start = f'Topic: {topic}. Side: CON.'

    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    d1 = r1.json()
    cid = d1['conversation_id']
    a1 = _last_bot_msg(d1)
    assert a1 and a1.strip()
    assert END_MARKER not in a1

    user_msgs = [
        "Dogs are humans' best friend. Their loyalty, empathy, and assistance make them unmatched companions.",
        "Dogs are humans' best friend. Therapy and service roles show unique, measurable benefits to people’s lives.",
        "Dogs are humans' best friend. They reduce loneliness, foster routine, and increase physical activity for owners.",
        "Dogs are humans' best friend. Their social facilitation helps people connect, building community and belonging.",
    ]

    svc = _get_service_instance()
    count = 0
    for t in user_msgs:
        count += 1
        r = client.post('/messages', json={'conversation_id': cid, 'message': t})
        assert r.status_code == 200, r.text
        bot_msg = _last_bot_msg(r.json())
        state = svc.debate_store.get(conversation_id=cid)
        assert bot_msg and bot_msg.strip()
        assert state.positive_judgements == count
        assert END_MARKER not in bot_msg, f'Unexpected immediate end: {bot_msg!r}'

    state = svc.debate_store.get(conversation_id=cid)
    assert state is not None
    assert getattr(state, 'match_concluded', False), (
        'Debate should have concluded by the 5th aligned-opposition turn (user vs CON bot).'
    )

    r_after = client.post(
        '/messages', json={'conversation_id': cid, 'message': 'One more thought?'}
    )
    assert r_after.status_code == 200, r_after.text
    ended_reply = _last_bot_msg(r_after.json())
    assert END_MARKER in ended_reply, f'Expected end marker, got: {ended_reply!r}'


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OPENAI_API_KEY not set'
)
def test_concludes_by_five_turns_misaligned_user_vs_PRO_bot(client):
    topic = "Dogs are humans' best friend"
    start = f'Topic: {topic}. Side: PRO.'

    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    d1 = r1.json()
    cid = d1['conversation_id']
    a1 = _last_bot_msg(d1)
    assert a1 and a1.strip()
    assert END_MARKER not in a1

    user_msgs = [
        "Dogs are not humans' best friend. Many people prefer other companions or none, due to allergies and costs.",
        "It is not true that dogs are humans' best friend. Time demands, training, and vet bills make dogs impractical.",
        "Dogs are not humans' best friend. Noise, bites, and neighborhood issues outweigh benefits for numerous owners.",
        "It is not true that dogs are humans' best friend. Cats and other pets provide affection with fewer demands.",
    ]

    svc = _get_service_instance()
    for t in user_msgs:
        r = client.post('/messages', json={'conversation_id': cid, 'message': t})
        assert r.status_code == 200, r.text
        bot_msg = _last_bot_msg(r.json())

        assert bot_msg and bot_msg.strip()
        assert END_MARKER not in bot_msg, f'Unexpected immediate end: {bot_msg!r}'

    state = svc.debate_store.get(conversation_id=cid)
    assert state is not None
    assert getattr(state, 'match_concluded', False), (
        'Debate should have concluded by the 5th aligned-opposition turn (user vs PRO bot).'
    )

    r_after = client.post(
        '/messages', json={'conversation_id': cid, 'message': 'Keep going?'}
    )
    assert r_after.status_code == 200, r_after.text
    ended_reply = _last_bot_msg(r_after.json())
    assert END_MARKER in ended_reply, f'Expected end marker, got: {ended_reply!r}'


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
def test_real_llm_winning_game_con_god_exists(client):
    """
    EN, CON. Uses granular notices and off-topic nudge; no full settings dump expected.
    """
    topic = 'God exists'
    lang = 'en'
    lang_code = 'EN'
    stance = 'CON'

    def last_bot_msg(resp_json):
        return resp_json['message'][-1]['message']

    reset_llm_singleton_cache()

    # ---- Turn 1: start conversation ----
    start_message = 'Topic: God exists. Side: CON.'
    r1 = client.post(
        '/messages', json={'conversation_id': None, 'message': start_message}
    )
    assert r1.status_code == 201, r1.text
    d1 = r1.json()
    conv_id = d1['conversation_id']

    a1 = last_bot_msg(d1)
    assert isinstance(a1, str) and a1.strip()
    assert_language(a1, lang)
    assert 'CON' in a1.upper(), (
        f'Expected first reply to acknowledge CON stance, got: {a1!r}'
    )

    time.sleep(0.2)

    # ---- Turn 2: user tries to switch stance ----
    t2 = 'Please switch to PRO.'
    r2 = client.post('/messages', json={'conversation_id': conv_id, 'message': t2})
    assert r2.status_code == 200, r2.text
    a2 = last_bot_msg(r2.json())
    assert_language(a2, lang)
    assert_granular_notice_en(
        a2, topic=topic, stance=stance, lang_code=lang_code, expect_stance=True
    )

    time.sleep(0.2)

    # ---- Turn 3: user asks an off-topic question ----
    t3 = 'What is 2+2?'
    r3 = client.post('/messages', json={'conversation_id': conv_id, 'message': t3})
    assert r3.status_code == 200, r3.text
    a3 = last_bot_msg(r3.json())
    assert_language(a3, lang)

    nudge = expected_offtopic_nudge(topic, lang)
    assert nudge in _norm(a3), (
        f'Missing on-topic nudge. Expected contains: {nudge!r}\nGot: {a3!r}'
    )
    assert len(a3.split()) <= 80, f'Off-topic reply too long: {len(a3.split())} words'

    time.sleep(0.2)

    # ---- Turn 4: user tries to switch language ----
    t4 = 'Switch to Spanish, please.'
    r4 = client.post('/messages', json={'conversation_id': conv_id, 'message': t4})
    assert r4.status_code == 200, r4.text
    a4 = last_bot_msg(r4.json())
    assert_language(a4, lang)
    assert_granular_notice_en(
        a4, topic=topic, stance=stance, lang_code=lang_code, expect_lang=True
    )

    time.sleep(0.2)

    # ---- Turn 5: request a CON argument from evil ----
    t5 = "Give a concise argument from evil against God's existence."
    r5 = client.post('/messages', json={'conversation_id': conv_id, 'message': t5})
    assert r5.status_code == 200, r5.text
    a5 = last_bot_msg(r5.json())
    assert_language(a5, lang)

    a5_l = _norm(a5)
    assert any(kw in a5_l for kw in ['evil', 'suffering', 'gratuitous harm']), (
        f'Expected an evil-based argument, got: {a5!r}'
    )
    assert 'match concluded' not in a5_l

    time.sleep(0.2)

    # ---- Turn 6: request a CON argument from divine hiddenness ----
    t6 = 'Now a concise argument from divine hiddenness.'
    r6 = client.post('/messages', json={'conversation_id': conv_id, 'message': t6})
    assert r6.status_code == 200, r6.text
    a6 = last_bot_msg(r6.json())
    assert_language(a6, lang)

    a6_l = _norm(a6)
    assert any(
        kw in a6_l for kw in ['hidden', 'hiddenness', 'nonresistant', 'silence']
    ), f'Expected a hiddenness-based argument, got: {a6!r}'
    assert 'match concluded' not in a6_l


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OPENAI_API_KEY not set'
)
def test_ended_state_outputs_exact_marker(client):
    # Start
    r1 = client.post(
        '/messages', json={'conversation_id': None, 'message': 'Topic: X. Side: PRO.'}
    )
    assert r1.status_code == 201
    d1 = r1.json()
    cid = d1['conversation_id']

    # Flip debate status to ENDED
    from app.infra.service import get_service
    from app.main import app as fastapi_app

    override = fastapi_app.dependency_overrides.get(get_service)
    svc = override()

    state = svc.debate_store.get(conversation_id=cid)
    state.match_concluded = True
    svc.debate_store.save(conversation_id=cid, state=state)

    # Any follow-up yields exact marker
    r2 = client.post(
        '/messages', json={'conversation_id': cid, 'message': 'keep going?'}
    )
    assert r2.status_code == 200
    a2 = r2.json()['message'][-1]['message']
    assert END_MARKER in a2
