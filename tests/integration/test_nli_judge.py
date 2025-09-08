# tests/integration/test_nli_judge.py
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import pytest

from app.adapters.llm.anthropic import (  # adjust if needed
    AnthropicAdapter,
)
from app.adapters.llm.constants import JUDGE_SYSTEM_PROMPT, AnthropicModels, Difficulty
from app.adapters.llm.types import JudgeResult

pytestmark = pytest.mark.integration
from app.domain.nli.judge_payload import NLIJudgePayload


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
        self.return_text: str = '{"verdict":"OPPOSITE","concession":true,"confidence":0.84,"reason":"strong_contradiction"}'

        class _Messages:
            def __init__(self, outer):
                self._outer = outer

            async def create(self, **kwargs):
                self._outer.calls.append(kwargs)
                return _FakeResp([_FakeBlock(self._outer.return_text)])

        self.messages = _Messages(self)


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
    payload = NLIJudgePayload(
        topic='The death penalty should be abolished',
        stance='pro',
        user_text='Abolition risks justice for the worst crimes.',
        bot_text='Irreversibility and wrongful convictions undermine legitimacy.',
        thesis_scores={'entailment': 0.22, 'contradiction': 0.71, 'neutral': 0.07},
        pair_best={'entailment': 0.10, 'contradiction': 0.68, 'neutral': 0.22},
        max_sent_contra=0.71,
        on_topic=True,
        user_wc=37,
    ).to_dict()

    decision = await adapter.nli_judge(system=JUDGE_SYSTEM_PROMPT, payload=payload)

    # Parsed decision
    assert isinstance(decision, JudgeResult)
    assert decision.verdict == 'OPPOSITE'
    assert decision.concession is True
    assert 0.8 <= decision.confidence <= 0.9
    assert decision.reason == 'strong_contradiction'

    # Exactly one Anthropic call with expected params
    assert len(fake_anthropic.calls) == 1
    sent = fake_anthropic.calls[0]
    assert sent['system'] == JUDGE_SYSTEM_PROMPT
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

    # Keys required by the JUDGE_SYSTEM_PROMPT
    for k in (
        'topic',
        'stance',
        'user_text',
        'bot_text',
        'thesis_scores',
        'pair_best',
        'max_sent_contra',
        'on_topic',
        'user_wc',
    ):
        assert k in parsed, f'missing key: {k}'
    assert parsed['stance'] in ('pro', 'con')


@pytest.mark.asyncio
async def test_nli_judge_low_confidence_still_parses(
    adapter: AnthropicAdapter, fake_anthropic: FakeAsyncAnthropic
):
    fake_anthropic.return_text = '{"verdict":"SAME","concession":false,"confidence":0.25,"reason":"weak_support"}'

    decision = await adapter.nli_judge(
        system=JUDGE_SYSTEM_PROMPT,
        payload=NLIJudgePayload(
            topic='God exists',
            stance='pro',
            user_text='It might be true for some people.',
            bot_text='Fine-tuning suggests intentional design.',
            thesis_scores={'entailment': 0.40, 'contradiction': 0.20, 'neutral': 0.40},
            pair_best={'entailment': 0.35, 'contradiction': 0.25, 'neutral': 0.40},
            max_sent_contra=0.30,
            on_topic=True,
            user_wc=12,
        ).to_dict(),
    )

    assert decision.verdict == 'SAME'
    assert decision.concession is False
    assert 0.0 <= decision.confidence <= 0.3
    assert decision.reason == 'weak_support'


@pytest.mark.asyncio
async def test_nli_judge_invalid_json_raises(
    adapter: AnthropicAdapter, fake_anthropic: FakeAsyncAnthropic
):
    fake_anthropic.return_text = 'INVALID: not-json'
    with pytest.raises(ValueError):
        await adapter.nli_judge(
            system=JUDGE_SYSTEM_PROMPT,
            payload=NLIJudgePayload(
                topic='Free speech should be restricted',
                stance='con',
                user_text='Restrictions are dangerous.',
                bot_text='Harm mitigation can justify limits.',
                thesis_scores={
                    'entailment': 0.20,
                    'contradiction': 0.60,
                    'neutral': 0.20,
                },
                pair_best={'entailment': 0.15, 'contradiction': 0.55, 'neutral': 0.30},
                max_sent_contra=0.60,
                on_topic=True,
                user_wc=18,
            ).to_dict(),
        )


@pytest.mark.asyncio
async def test_nli_judge_invalid_verdict_raises(
    adapter: AnthropicAdapter, fake_anthropic: FakeAsyncAnthropic
):
    fake_anthropic.return_text = (
        '{"verdict":"MAYBE","concession":false,"confidence":0.9,"reason":"bad_label"}'
    )
    with pytest.raises(ValueError):
        await adapter.nli_judge(
            system=JUDGE_SYSTEM_PROMPT,
            payload=NLIJudgePayload(
                topic='Economic growth should be prioritized over environment',
                stance='pro',
                user_text='We need balance, not extremes.',
                bot_text='Growth lifts people from poverty.',
                thesis_scores={
                    'entailment': 0.33,
                    'contradiction': 0.44,
                    'neutral': 0.23,
                },
                pair_best={'entailment': 0.28, 'contradiction': 0.45, 'neutral': 0.27},
                max_sent_contra=0.45,
                on_topic=True,
                user_wc=22,
            ).to_dict(),
        )


@pytest.mark.asyncio
async def test_nli_judge_confidence_out_of_range_raises(
    adapter: AnthropicAdapter, fake_anthropic: FakeAsyncAnthropic
):
    fake_anthropic.return_text = (
        '{"verdict":"UNKNOWN","concession":false,"confidence":1.7,"reason":"oops"}'
    )
    with pytest.raises(ValueError):
        await adapter.nli_judge(
            system=JUDGE_SYSTEM_PROMPT,
            payload=NLIJudgePayload(
                topic='AI should be regulated',
                stance='pro',
                user_text='Regulations lag behind innovation.',
                bot_text='Guardrails prevent harm.',
                thesis_scores={
                    'entailment': 0.30,
                    'contradiction': 0.30,
                    'neutral': 0.40,
                },
                pair_best={'entailment': 0.25, 'contradiction': 0.30, 'neutral': 0.45},
                max_sent_contra=0.30,
                on_topic=True,
                user_wc=15,
            ).to_dict(),
        )
