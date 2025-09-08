# tests/test_openai_score_judge.py
import json

import pytest

from app.adapters.scoring.openai import _JSON_SCHEMA, OpenAIScoreJudge
from app.domain.ports.score_judge import ScoreFeatures

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


# ---------- Fixtures ----------


@pytest.fixture
def fake_openai(monkeypatch):
    """
    Monkeypatch the OpenAI class inside the target module so the judge uses our fake client.
    """
    import app.adapters.judge.openai_score_judge as target_mod

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
    assert sent['model'] == 'gpt-4o-mini'
    assert sent['temperature'] == 0.0

    # Messages: system + user with serialized features
    input_msgs = sent['input']
    assert isinstance(input_msgs, list) and len(input_msgs) == 2
    assert input_msgs[0]['role'] == 'system'
    assert (
        'meta-judge' in input_msgs[0]['content']
    )  # rough check system prompt got through

    # user message should be compact JSON (no spaces), matching our serialization
    expected_user = json.dumps(
        sample_features, ensure_ascii=False, separators=(',', ':')
    )
    assert input_msgs[1]['role'] == 'user'
    assert input_msgs[1]['content'] == expected_user

    # Response format should enforce json_schema and match our schema name
    rf = sent['response_format']
    assert rf['type'] == 'json_schema'
    assert rf['json_schema']['name'] == _JSON_SCHEMA['name']
    # a couple of key checks on the schema structure
    props = rf['json_schema']['schema']['properties']
    assert 'alignment' in props and 'concession' in props and 'confidence' in props


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
    # Ensure the minimal dict was serialized as compact JSON
    expected_user = json.dumps(minimal, ensure_ascii=False, separators=(',', ':'))
    assert sent['input'][1]['content'] == expected_user
