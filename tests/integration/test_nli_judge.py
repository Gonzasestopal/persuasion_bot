# tests/integration/test_nli_judge.py
import json
from typing import Any, Dict, List

import pytest

from app.adapters.llm.anthropic import AnthropicAdapter  # adjust if needed
from app.adapters.llm.constants import (
    JUDGE_SCORE_SYSTEM_PROMPT,
    AnthropicModels,
    Difficulty,
)
from app.adapters.llm.types import JudgeResult
from app.domain.nli.judge_payload import NLIJudgePayload

pytestmark = pytest.mark.integration


# -------- Minimal fake Anthropic client --------
class _FakeBlock:
    def __init__(self, text: str):
        self.type = 'text'
        self.text = text


class _FakeResp:
    def __init__(self, content: List[_FakeBlock]):
        self.content = content


class FakeAsyncAnthropic:
    """
    Captures calls; returns predetermined single-line JSON text.
    Set self.return_text in each test.
    """

    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        # Align with adapter's schema: accept/confidence/reason
        self.return_text: str = (
            '{"accept":true,"confidence":0.84,"reason":"strong_contradiction"}'
        )

        class _Messages:
            def __init__(self, outer):
                self._outer = outer

            async def create(self, **kwargs):
                self._outer.calls.append(kwargs)
                return _FakeResp([_FakeBlock(self._outer.return_text)])

        self.messages = _Messages(self)


# -------- Helpers --------
def _payload_dict() -> Dict[str, Any]:
    """Build a payload matching current NLIJudgePayload schema."""
    return NLIJudgePayload(
        topic='Free speech should be restricted',
        stance='con',
        user_text='Restrictions are dangerous.',
        bot_text='Harm mitigation can justify limits.',
        thesis_scores={'entailment': 0.20, 'contradiction': 0.60, 'neutral': 0.20},
        pair_best={'entailment': 0.15, 'contradiction': 0.55, 'neutral': 0.30},
        max_sent_contra=0.60,
        on_topic=True,
        user_wc=18,
        language='en',
        turn_index=0,
        # NEW: policy/progress must be dicts (not str/float)
        policy={'required_positive_judgements': 2, 'max_assistant_turns': 8},
        progress={'positive_judgements': 0, 'assistant_turns': 0},
    ).to_dict()


# -------- Fixtures --------
@pytest.fixture
def fake_anthropic():
    return FakeAsyncAnthropic()


@pytest.fixture
def adapter(fake_anthropic):
    return AnthropicAdapter(
        api_key='sk-test',
        client=fake_anthropic,
        model=AnthropicModels.CLAUDE_35,
        difficulty=Difficulty.EASY,
        temperature=0.0,
        max_output_tokens=160,
    )


# -------- Tests --------
@pytest.mark.asyncio
async def test_nli_judge_parses_valid_response(
    adapter: AnthropicAdapter, fake_anthropic: FakeAsyncAnthropic
):
    decision = await adapter.nli_judge(payload=_payload_dict())

    # Parsed decision
    assert isinstance(decision, JudgeResult)
    assert decision.accept is True
    assert 0.8 <= decision.confidence <= 0.9

    # reason may be an Enum or a string; normalize to string for assertion
    reason_str = getattr(decision.reason, 'value', str(decision.reason))
    assert reason_str == 'ambiguous_evidence'

    # Exactly one Anthropic call with expected params
    assert len(fake_anthropic.calls) == 1
    sent = fake_anthropic.calls[0]
    assert sent['system'] == JUDGE_SCORE_SYSTEM_PROMPT
    assert sent['model'] == adapter.model
    assert sent['temperature'] == 0.0
    assert sent['max_tokens'] >= 120

    # Ensure we sent the JSON payload as the single user message
    msgs = sent['messages']
    assert isinstance(msgs, list) and len(msgs) == 1
    assert msgs[0]['role'] == 'user'
    content = msgs[0]['content']
    assert isinstance(content, list) and content and content[0]['type'] == 'text'
    sent_text = content[0]['text']
    parsed = json.loads(sent_text)

    # Keys required by the current schema
    for k in ('topic', 'stance', 'user_text', 'bot_text', 'nli', 'policy', 'progress'):
        assert k in parsed, f'missing key: {k}'
    assert parsed['stance'] in ('pro', 'con')

    # Nested NLI keys
    nli = parsed['nli']
    for k in ('thesis_scores', 'pair_best', 'max_sent_contra', 'on_topic', 'user_wc'):
        assert k in nli, f'missing nli key: {k}'


@pytest.mark.asyncio
async def test_nli_judge_low_confidence_still_parses(
    adapter: AnthropicAdapter, fake_anthropic: FakeAsyncAnthropic
):
    fake_anthropic.return_text = (
        '{"accept":false,"confidence":0.25,"reason":"weak_support"}'
    )

    decision = await adapter.nli_judge(payload=_payload_dict())

    assert decision.accept is False
    assert 0.0 <= decision.confidence <= 0.3
    reason_str = getattr(decision.reason, 'value', str(decision.reason))
    assert reason_str == 'ambiguous_evidence'


@pytest.mark.asyncio
async def test_nli_judge_invalid_json_raises(
    adapter: AnthropicAdapter, fake_anthropic: FakeAsyncAnthropic
):
    fake_anthropic.return_text = 'INVALID: not-json'
    with pytest.raises(ValueError):
        await adapter.nli_judge(payload=_payload_dict())


@pytest.mark.asyncio
async def test_nli_judge_invalid_reason_falls_back_to_ambiguous(
    adapter: AnthropicAdapter, fake_anthropic: FakeAsyncAnthropic
):
    # Adapter maps unknown reasons via ALIASES/ALLOWED_REASONS to 'ambiguous_evidence'
    fake_anthropic.return_text = (
        '{"accept":false,"confidence":0.9,"reason":"bad_label"}'
    )
    decision = await adapter.nli_judge(payload=_payload_dict())
    reason_str = getattr(decision.reason, 'value', str(decision.reason))
    assert reason_str == 'ambiguous_evidence'


@pytest.mark.asyncio
async def test_nli_judge_confidence_out_of_range_is_clamped(
    adapter: AnthropicAdapter, fake_anthropic: FakeAsyncAnthropic
):
    # Adapter clamps confidence into [0,1] instead of raising
    fake_anthropic.return_text = '{"accept":false,"confidence":1.7,"reason":"oops"}'
    decision = await adapter.nli_judge(payload=_payload_dict())
    # Expect clamped to 1.0
    assert decision.confidence == 1.0
