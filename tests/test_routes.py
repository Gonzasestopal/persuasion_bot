import os
import time

import pytest

from app.factories import get_service
from app.main import app
from app.settings import settings


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
            'Can I make you take the CON side?',
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
    # ---- Turn 1: start conversation ----
    r1 = client.post(
        '/messages', json={'conversation_id': None, 'message': start_message}
    )
    assert r1.status_code == 201, r1.text
    data1 = r1.json()

    # Keep the returned conversation_id to continue the same debate thread
    conv_id = data1['conversation_id']

    # The bot's message should reflect the initial side (per your prompt rules)
    first_bot_msg = data1['message'][-1]['message']
    assert expected_stance in first_bot_msg.upper()

    # Tiny pause to avoid rate limits with some providers
    time.sleep(0.2)

    # ---- Turn 2: continue same conversation ----
    r2 = client.post(
        '/messages', json={'conversation_id': conv_id, 'message': second_msg}
    )
    assert r2.status_code == 200, r2.text
    data2 = r2.json()

    second_bot_msg = data2['message'][-1]['message']
    assert second_expected_stance in second_bot_msg


def test_returns_422_on_invalid_start(client):
    """
    Starting a conversation without 'Topic: ...' and 'Side: PRO|CON' should
    trigger your parser to raise ValueError -> route returns 422.
    """
    r = client.post(
        '/messages', json={'conversation_id': None, 'message': 'hello there'}
    )
    assert r.status_code == 422, r.text

    # Optional: detail text check (keep loose to avoid brittle tests)
    detail = r.json().get('detail', '')
    assert 'topic' in detail.lower() or 'side' in detail.lower()


def test_returns_422_on_exceeding_topic_length(client):
    """
    Starting a conversation with a 'Topic' longer than allowed (e.g. >50 chars)
    should raise a ValueError / validation error -> route returns 422.
    """
    too_long_topic = 'A' * 101  # 101 chars, exceeds limit

    r = client.post(
        '/messages',
        json={
            'conversation_id': None,
            'message': f'Topic: {too_long_topic}\nSide: PRO',
        },
    )
    assert r.status_code == 422, r.text

    # Optional: check detail mentions "topic" or "length"
    detail = r.json().get('detail', '')
    assert 'topic' in detail.lower() or 'length' in detail.lower()


def test_returns_404_on_unknown_conversation_id(client):
    """
    Continuing a conversation with a non-existent conversation_id should raise
    KeyError -> route returns 404.
    """
    r = client.post(
        '/messages', json={'conversation_id': 999_999_999, 'message': 'continue please'}
    )
    assert r.status_code == 404, r.text

    # Optional: exact message match based on your route
    assert r.json().get('detail') == 'conversation_id not found or expired'


def test_returns_503_on_timeout(client, monkeypatch):
    """
    Force an asyncio.wait_for timeout by setting the request timeout to 0 seconds.
    This guarantees a TimeoutError -> route returns 503, regardless of LLM speed.
    """
    # Set timeout to zero only for this test
    monkeypatch.setattr(settings, 'REQUEST_TIMEOUT_S', 0)

    # Start conversation (this will hit the timeout in wait_for)
    r = client.post(
        '/messages', json={'conversation_id': None, 'message': 'Topic: X. Side: PRO.'}
    )
    assert r.status_code == 503, r.text
    assert r.json().get('detail') == 'response generation timed out'

    # (Restore is automatic after test because monkeypatch patched the object attribute for test scope)


def _temporarily_remove_di_override():
    """Temporarily remove any DI override on get_service for this test."""
    had_override = get_service in app.dependency_overrides
    saved = app.dependency_overrides.get(get_service)
    if had_override:
        del app.dependency_overrides[get_service]
    return had_override, saved


def _restore_di_override(had_override, saved):
    if had_override:
        app.dependency_overrides[get_service] = saved


def test_returns_500_on_missing_api_key(client, monkeypatch):
    """
    When provider API keys are missing/misconfigured, the app should surface a 500 ConfigError.
    This test clears provider keys and ensures we use the *real factory* (no DI override).
    """
    # Ensure we are not using a pre-built service from conftest
    had_override, saved = _temporarily_remove_di_override()
    try:
        # Clear env + settings for both providers so fallback can't succeed
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        monkeypatch.setattr(settings, 'OPENAI_API_KEY', None, raising=False)
        monkeypatch.setattr(settings, 'ANTHROPIC_API_KEY', None, raising=False)

        # If you use a fallback adapter, ensure it tries OpenAI first (or either),
        # but both should be missing anyway:
        monkeypatch.setattr(settings, 'PRIMARY_LLM', 'openai', raising=False)
        monkeypatch.setattr(settings, 'SECONDARY_LLM', 'claude', raising=False)

        r = client.post(
            '/messages',
            json={'conversation_id': None, 'message': 'Topic: X. Side: PRO.'},
        )
        assert r.status_code == 500, r.text

        # Be flexible on the message, but it should mention config / api key.
        detail = r.json().get('detail', '').lower()
        allowed = ('config' in detail, 'api_key' in detail, 'misconfigured' in detail)
        assert any(allowed), detail
    finally:
        _restore_di_override(had_override, saved)
