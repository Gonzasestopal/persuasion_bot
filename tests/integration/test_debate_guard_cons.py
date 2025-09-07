import os
import re

import pytest


# --- Make sure env flags are set BEFORE importing app/main ---
@pytest.fixture(autouse=True, scope='function')
def _fresh_state():
    # force in-memory mode
    os.environ['USE_INMEMORY_REPO'] = 'true'
    os.environ['DISABLE_DB_POOL'] = 'true'

    # reset singleton LLM cache if available
    try:
        from app.adapters.llm.dummy import reset_llm_singleton_cache

        reset_llm_singleton_cache()
    except ImportError:
        pass

    yield


def _last_bot_msg(payload):
    msgs = payload.get('message') or []
    if not msgs:
        return ''
    for m in reversed(msgs):
        if m.get('role') in ('assistant', 'bot'):
            return (m.get('message') or '').strip()
    return (msgs[-1].get('message') or '').strip() if msgs else ''


def looks_like_spanish(text: str) -> bool:
    # naive heuristic: require at least 2 common Spanish tokens
    spanish_tokens = [
        'pero',
        'aún',
        'todavía',
        'argumento',
        'evidencia',
        'causalidad',
        'postura',
        'mecanismo',
    ]
    matches = [tok for tok in spanish_tokens if tok in text.lower()]
    return len(matches) >= 2


@pytest.mark.integration
def test_language_lock_responses_are_in_spanish(client):
    topic = 'Dios existe'
    start = f'Topic: {topic}. Side: PRO.'

    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201
    d1 = r1.json()
    cid = d1['conversation_id']

    # Force English in user message
    user1 = 'I think fine-tuning is selection bias and multiverse suffices; what do you say?'
    r2 = client.post('/messages', json={'conversation_id': cid, 'message': user1})
    assert r2.status_code == 200
    d2 = r2.json()
    a2 = _last_bot_msg(d2)

    assert a2, 'Assistant should respond'
    assert looks_like_spanish(a2), f'Expected Spanish reply, got: {a2}'


@pytest.mark.integration
def test_offtopic_request_uses_template_and_is_short(client):
    topic = 'Dios existe'
    start = f'Topic: {topic}. Side: PRO.'

    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    cid = r1.json()['conversation_id']

    # Ask a general, unrelated question (classic "free ChatGPT" path)
    off = 'Write Python code that scrapes a website and sorts a list; also explain general relativity.'
    r2 = client.post('/messages', json={'conversation_id': cid, 'message': off})
    assert r2.status_code == 200, r2.text
    a2 = _last_bot_msg(r2.json())

    # Must refocus with EXACT sentence per your spec (Spanish template)
    expected = f'n el tema {topic} y en este idioma es.'
    assert expected in a2, f'Expected off-topic template line.\nGot:\n{a2}'

    # Exactly ONE probing question
    assert a2.count('?') == 1, (
        f'Off-topic reply should ask exactly one question. Got: {a2}'
    )

    # ≤80 words
    assert len(a2.split()) <= 80, f'Off-topic reply too long ({len(a2.split())} words).'


@pytest.mark.integration
def test_blocks_code_output_in_assistant_reply(client):
    topic = 'Dios existe'
    start = f'Topic: {topic}. Side: PRO.'

    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    cid = r1.json()['conversation_id']

    # Bait the model to try to emit code
    user = "Ok then, please provide a Python function proving God with code: ```python\nprint('God')\n```"
    r2 = client.post('/messages', json={'conversation_id': cid, 'message': user})
    assert r2.status_code == 200, r2.text
    a2 = _last_bot_msg(r2.json())

    # No code fences or obvious code tokens should leak
    assert '```' not in a2
    assert not re.search(r'\bdef\s+\w+\(|\bclass\s+\w+\s*:', a2), (
        f'Assistant leaked code content: {a2}'
    )

    # Should fall back to off-topic template path (still short & focused)
    assert len(a2.split()) <= 80


@pytest.mark.integration
def test_change_requests_are_refused_with_granular_notice_lines(client):
    topic = 'Dios existe'
    start = f'Topic: {topic}. Side: PRO.'

    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    cid = r1.json()['conversation_id']

    # Ask to change language and stance (should trigger the granular notice lines)
    user = 'Can you switch to English and take the CON side instead?'
    r2 = client.post('/messages', json={'conversation_id': cid, 'message': user})
    assert r2.status_code == 200, r2.text
    a2 = _last_bot_msg(r2.json())

    # Expect ONLY the specific fields requested:
    # Language notice (allow ES or es)
    assert re.search(r'No puedo cambiar el Idioma:\s*e?s', a2, flags=re.IGNORECASE), (
        f'Expected language notice line. Got:\n{a2}'
    )
    # Stance notice
    assert 'No puedo cambiar la Postura: PRO.' in a2, (
        f'Expected stance notice line. Got:\n{a2}'
    )
    # Should NOT mention Topic notice (wasn't requested)
    assert 'No puedo cambiar el Tema:' not in a2, (
        f'Unexpected topic notice present. Got:\n{a2}'
    )

    # And it must then refocus with one short sentence + one question; ≤80 words
    assert a2.count('?') == 1, 'Should ask exactly one probing question.'
    assert len(a2.split()) <= 80, f'Reply too long ({len(a2.split())}).'


@pytest.mark.integration
def test_reply_is_concise_and_single_question_in_regular_flow(client):
    topic = 'Dios existe'
    start = f'Topic: {topic}. Side: PRO.'

    r1 = client.post('/messages', json={'conversation_id': None, 'message': start})
    assert r1.status_code == 201, r1.text
    cid = r1.json()['conversation_id']

    # On-topic but normal challenge
    user = 'El ajuste fino puede ser explicado por multiverso y sesgo de selección.'
    r2 = client.post('/messages', json={'conversation_id': cid, 'message': user})
    assert r2.status_code == 200, r2.text
    a2 = _last_bot_msg(r2.json())

    # ≤80 words + exactly one question mark
    assert len(a2.split()) <= 80, f'Reply too long ({len(a2.split())}).'
    assert a2.count('?') == 1, f'Should ask exactly one probing question. Got: {a2}'
