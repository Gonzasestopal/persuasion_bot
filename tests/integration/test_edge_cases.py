import os
import re
import time
import unicodedata

import pytest

from app.infra.llm import reset_llm_singleton_cache

pytestmark = pytest.mark.integration


# ----------------------------
# Normalization helpers
# ----------------------------


def _norm_upper(s: str) -> str:
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace('’', "'")  # smart → straight
    s = re.sub(r'\s+', ' ', s)
    return s.strip().upper()


# ----------------------------
# Tests
# ----------------------------


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'), reason='OPENAI_API_KEY not set'
)
def test_change_topic_triggers_exact_notice(client):
    topic, lang_code, stance = 'God exists', 'EN', 'PRO'
    r1 = client.post(
        '/messages',
        json={
            'conversation_id': None,
            'message': f'Topic: {topic}. Side: {stance}.',
        },
    )
    assert r1.status_code == 201
    conv_id = r1.json()['conversation_id']

    r2 = client.post(
        '/messages',
        json={
            'conversation_id': conv_id,
            'message': 'Let’s debate climate change instead.',
        },
    )
    assert r2.status_code == 200
    a2 = r2.json()['message'][-1]['message']
    exact_notice(a2, topic=topic, stance=stance)


def exact_notice(msg: str, topic: str, stance: str = 'PRO'):
    """
    Softer check:
    - Accept granular notice: "I can't change Topic: {topic}."
    - OR legacy notice: "I can't change these settings." + Language/Topic/Stance fields.
    Case/spacing/smart-quote insensitive.
    """
    up = _norm_upper(msg)
    topic_up = topic.upper()
    stance_up = stance.upper()

    # Granular variants (allow optional "THE")
    granular_ok = (
        "I CAN'T CHANGE TOPIC:" in up
        or 'I CANNOT CHANGE TOPIC:' in up
        or "I CAN'T CHANGE THE TOPIC:" in up
        or 'I CANNOT CHANGE THE TOPIC:' in up
    ) and f'TOPIC: {topic_up}' in up

    # Legacy full settings notice
    legacy_ok = (
        "I CAN'T CHANGE THESE SETTINGS." in up
        and 'LANGUAGE: EN' in up
        and f'TOPIC: {topic_up}' in up
        and f'STANCE: {stance_up}' in up
    )

    assert granular_ok or legacy_ok, (
        'Expected a granular topic-change notice like '
        f'"I can\'t change Topic: {topic}." or the legacy full notice '
        '"I can\'t change these settings." with Language/Topic/Stance fields.\n'
        f'Got:\n{msg!r}'
    )


def expected_offtopic_nudge(topic: str, lang: str) -> str:
    if lang == 'en':
        return 'keep on topic'
    if lang == 'es':
        return 'Mantengámonos en el tema'
    raise ValueError(f'Unsupported lang {lang!r}')


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
@pytest.mark.parametrize(
    'start_message, lang, lang_code, topic, off_topic_msg',
    [
        (
            'Topic: God exists. Side: PRO.',
            'en',
            'EN',
            'God exists',
            'What is 2+2?',
        ),
        (
            'topic: Dios existe.  side: PRO.',
            'es',
            'ES',
            'Dios existe',
            '¿Cuánto es 2+2?',
        ),
    ],
)
def test_real_llm_refocuses_on_topic_when_offtopic(
    client, start_message, lang, lang_code, topic, off_topic_msg
):
    """
    Ensures that when the user goes off-topic, the bot:
      - Replies in the declared language,
      - Includes the on-topic nudge phrase for that language,
      - Keeps reply short (<= 80 words per your prompt).
    """

    def assert_language(text: str, expected: str):
        up = _norm_upper(text)
        if expected == 'es':
            # soft: look for obvious Spanish cues or ES marker
            ok = (
                '¿' in text
                or '¡' in text
                or ' ES' in up
                or any(
                    w in up
                    for w in [' PERO ', ' AUNQUE ', ' CAUSALIDAD ', ' ARGUMENTO ']
                )
            )
            assert ok, f'Expected Spanish; got: {text!r}'
        elif expected == 'en':
            ok = ' EN' in up or any(
                w in up
                for w in [
                    ' HOWEVER ',
                    ' ALTHOUGH ',
                    ' EVIDENCE ',
                    ' STANCE ',
                    ' TOPIC ',
                ]
            )
            assert ok, f'Expected English; got: {text!r}'
        else:
            raise AssertionError(f'Unsupported lang {expected!r}')

    # ---- Turn 1: start conversation ----
    r1 = client.post(
        '/messages', json={'conversation_id': None, 'message': start_message}
    )
    assert r1.status_code == 201, r1.text
    data1 = r1.json()
    conv_id = data1['conversation_id']

    first_bot_msg = data1['message'][-1]['message']
    assert isinstance(first_bot_msg, str) and first_bot_msg.strip()

    time.sleep(0.2)  # tiny pause for provider rate limits

    # ---- Turn 2: send OFF-TOPIC message ----
    r2 = client.post(
        '/messages', json={'conversation_id': conv_id, 'message': off_topic_msg}
    )
    assert r2.status_code == 200, r2.text
    data2 = r2.json()

    second_bot_msg = data2['message'][-1]['message']
    assert isinstance(second_bot_msg, str) and second_bot_msg.strip()
    assert_language(second_bot_msg, lang)

    nudge = expected_offtopic_nudge(topic, lang).upper()
    assert nudge in _norm_upper(second_bot_msg), (
        f'Expected on-topic nudge missing.\nExpected contains: {nudge!r}\nGot: {second_bot_msg!r}'
    )

    word_count = len(second_bot_msg.split())
    assert word_count <= 80, f'Off-topic reply too long: {word_count} words'


def expected_immutable_notice(topic: str, lang_code: str, stance: str) -> str:
    # English immutable notice, per legacy rules
    return "I can't change these settings."


@pytest.mark.skipif(
    not os.environ.get('OPENAI_API_KEY'),
    reason='OPENAI_API_KEY not set; skipping live LLM integration test.',
)
@pytest.mark.parametrize(
    'start_message, expected_stance, second_msg, second_expected_stance',
    [
        (
            'Topic: Sports build character. Side: PRO.',
            'PRO',
            'Can I make you take the CON stance?',
            'PRO',
        )
    ],
)
def test_real_llm_never_changes_stance(
    client,
    start_message,
    expected_stance,
    second_msg,
    second_expected_stance,
):
    reset_llm_singleton_cache()

    # ---- Turn 1: start conversation ----
    r1 = client.post(
        '/messages', json={'conversation_id': None, 'message': start_message}
    )
    assert r1.status_code == 201, r1.text
    data1 = r1.json()
    conv_id = data1['conversation_id']

    first_bot_msg = data1['message'][-1]['message']
    assert expected_stance in _norm_upper(first_bot_msg)

    time.sleep(0.2)

    # ---- Turn 2: continue same conversation ----
    r2 = client.post(
        '/messages', json={'conversation_id': conv_id, 'message': second_msg}
    )
    assert r2.status_code == 200, r2.text
    data2 = r2.json()

    second_bot_msg = data2['message'][-1]['message']
    assert second_expected_stance in _norm_upper(second_bot_msg)
