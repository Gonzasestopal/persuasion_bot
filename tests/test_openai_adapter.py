# tests/test_openai_score_judge.py
import json
import re

import pytest

from app.adapters.scoring.openai import _JSON_SCHEMA, OpenAIScoreJudge
from app.domain.ports.scoring import ScoreFeatures

# ---------- Fakes ----------


class _FakeOpenAIResponse:
    def __init__(self, output_text: str):
        self._output_text = output_text

    @property
    def output_text(self) -> str:
        # The adapter reads .output_text (simple path)
        return self._output_text


class _FakeResponses:
    def __init__(self):
        self.last_kwargs = None
        self._to_return = _FakeOpenAIResponse(
            '{"alignment":"UNKNOWN","concession":false,"reason":"underdetermined","confidence":0.5}'
        )

    def set_return_text(self, text: str):
        self._to_return = _FakeOpenAIResponse(text)

    def create(self, **kwargs):
        # record the call for assertions
        self.last_kwargs = kwargs
        return self._to_return


class _FakeOpenAIClient:
    def __init__(self, *args, **kwargs):
        self.responses = _FakeResponses()


# ---------- Helpers ----------


def _get_messages(sent_kwargs):
    """
    Return the list of messages sent to the API, regardless of whether the
    adapter used the Responses API ("input") or Chat Completions-style ("messages").
    """
    if "input" in sent_kwargs:
        msgs = sent_kwargs["input"]
    elif "messages" in sent_kwargs:
        msgs = sent_kwargs["messages"]
    else:
        raise AssertionError(f"Expected 'input' or 'messages' in kwargs. Got keys: {list(sent_kwargs.keys())}")

    assert isinstance(msgs, list) and len(msgs) >= 2, "Expected at least system + user messages"
    return msgs


def _find_system_and_user(msgs):
    """
    Find first system and last user message (typical pattern: system first, user second).
    Be tolerant if order changes slightly.
    """
    sys_msg = next((m for m in msgs if m.get("role") == "system"), None)
    usr_msg = None
    for m in reversed(msgs):
        if m.get("role") == "user":
            usr_msg = m
            break
    assert sys_msg is not None, "No system message found"
    assert usr_msg is not None, "No user message found"
    return sys_msg, usr_msg


def _content_str(msg):
    """
    Some SDKs allow content to be string or list of parts.
    We coerce to a flat string for assertions.
    """
    c = msg.get("content", "")
    if isinstance(c, list):
        # concatenate plain text parts
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def _json_equal_or_wrapped(user_content_str, expected_payload: dict):
    """
    Accept either:
      - exactly the dict (serialized compactly)
      - or wrapped like {"features": {...}} or {"data": {...}}
    """
    try:
        parsed = json.loads(user_content_str)
    except Exception as e:
        raise AssertionError(f"User content is not valid JSON: {user_content_str}") from e

    if parsed == expected_payload:
        return True

    # common wrappers
    for k in ("features", "data", "payload"):
        inner = parsed.get(k) if isinstance(parsed, dict) else None
        if inner == expected_payload:
            return True

    # Otherwise, allow superset with the same keys/values for the expected payload
    if isinstance(parsed, dict) and all(k in parsed and parsed[k] == v for k, v in expected_payload.items()):
        return True

    raise AssertionError(f"JSON payload mismatch.\nExpected (or wrapped): {expected_payload}\nGot: {parsed}")


# ---------- Fixtures ----------


@pytest.fixture
def fake_openai(monkeypatch):
    """
    Monkeypatch the OpenAI class inside the target module so the judge uses our fake client.
    """
    import app.adapters.scoring.openai as target_mod

    def _fake_ctor(*args, **kwargs):
        return _FakeOpenAIClient()

    monkeypatch.setattr(target_mod, 'OpenAI', _fake_ctor)
    return target_mod  # return module in case caller wants access


@pytest.fixture
def judge(fake_openai):
    # api_key isn't used by the fake
    return OpenAIScoreJudge(api_key='sk-test', model='gpt-4o-mini', temperature=0.0)


@pytest.fixture
def sample_features() -> ScoreFeatures:
    return {
        'entailment_threshold': 0.65,
        'contradiction_threshold': 0.70,
        'pmin': 0.75,
        'margin': 0.15,
        'min_user_len': 30,
        'thesis_entailment': 0.12,
        'thesis_contradiction': 0.81,
        'pair_entailment': 0.20,
        'pair_contradiction': 0.78,
        'pair_confident': True,
        'thesis_confident': True,
        'stance': 'PRO',
        'user_len': 120,
    }


# ---------- Tests ----------


@pytest.mark.asyncio
async def test_score_success_parses_json_and_builds_request(
    judge, fake_openai, sample_features
):
    # Arrange: make fake return a valid JSON verdict
    expected_verdict = {
        'alignment': 'OPPOSITE',
        'concession': True,
        'reason': 'thesis_opposition',
        'confidence': 0.87,
    }
    judge.client.responses.set_return_text(json.dumps(expected_verdict))

    # Act
    result = await judge.score(features=sample_features)

    # Assert: parsed result
    assert result == expected_verdict

    # Assert: request payload sent to OpenAI
    sent = judge.client.responses.last_kwargs
    assert sent is not None
    # model/temperature passthrough
    assert sent.get('model') == 'gpt-4o-mini'
    assert sent.get('temperature') == 0.0

    # Messages/system prompt & features serialization
    msgs = _get_messages(sent)
    sys_msg, usr_msg = _find_system_and_user(msgs)

    sys_content = _content_str(sys_msg).lower()
    # only check that it's clearly a meta-judge prompt
    assert any(tok in sys_content for tok in ('meta-judge', 'score judge', 'scoring')), sys_content

    user_content = _content_str(usr_msg)
    # user message should serialize the features as compact JSON or a wrapped object containing them
    expected_user_compact = json.dumps(sample_features, ensure_ascii=False, separators=(',', ':'))
    # accept exact match OR JSON-equal/wrapped
    if user_content != expected_user_compact:
        _json_equal_or_wrapped(user_content, sample_features)

    # Response format should enforce json_schema and match our schema (allowing suffix variations)
    rf = sent.get('response_format') or {}
    assert rf.get('type') == 'json_schema'
    js = rf.get('json_schema') or {}
    name = js.get('name') or ''
    assert name == _JSON_SCHEMA['name'] or name.startswith(_JSON_SCHEMA['name'])
    schema = js.get('schema') or {}
    props = (schema.get('properties') or {})
    for k in ('alignment', 'concession', 'confidence'):
        assert k in props, f"Missing '{k}' in response schema properties"


@pytest.mark.asyncio
async def test_score_bad_json_returns_none(judge, fake_openai, sample_features):
    # Arrange: corrupt the return to non-JSON text
    judge.client.responses.set_return_text('not-json, sorry')

    # Act
    result = await judge.score(features=sample_features)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_score_minimal_features_still_serializes(judge, fake_openai):
    # Even with a tiny feature set, we should send valid JSON and not crash
    minimal = {'thesis_entailment': 0.5, 'thesis_contradiction': 0.6}  # type: ignore
    await judge.score(features=minimal)  # should not raise

    sent = judge.client.responses.last_kwargs
    assert sent is not None

    # Ensure the minimal dict was serialized as valid JSON (either raw or wrapped)
    msgs = _get_messages(sent)
    _, usr_msg = _find_system_and_user(msgs)
    user_content = _content_str(usr_msg)

    # If it's exactly the compact JSON, fine; otherwise ensure it's valid JSON and contains our keys/values.
    expected_user_compact = json.dumps(minimal, ensure_ascii=False, separators=(',', ':'))
    if user_content != expected_user_compact:
        _json_equal_or_wrapped(user_content, minimal)
