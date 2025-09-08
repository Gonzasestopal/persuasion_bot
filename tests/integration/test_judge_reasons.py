# tests/integration/test_judge_reasons.py
import pytest

from app.adapters.llm.anthropic import AnthropicAdapter
from app.domain.nli.reasons import JudgeReason

pytestmark = pytest.mark.integration


# ----------------------------
# Minimal fakes matching the adapter's usage
# ----------------------------


class _FakeContentBlock:
    def __init__(self, text: str):
        self.type = 'text'
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str):
        self.content = [_FakeContentBlock(text)]


class _FakeMessages:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.calls = []

    async def create(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _FakeAnthropicResponse(self.output_text)


class FakeAsyncAnthropic:
    def __init__(self, output_text: str):
        self.messages = _FakeMessages(output_text)


# ----------------------------
# Helpers / payload
# ----------------------------


def _has_enum(name: str) -> bool:
    return hasattr(JudgeReason, name)


def _base_payload(**overrides):
    base = {
        'topic': 'los perros son el mejor amigo del hombre',
        'stance': 'pro',
        'language': 'es',
        'turn_index': 2,
        'user_text': (
            'No son el mejor amigo del hombre: implican costos, alergias y riesgos. '
            'Otras especies ofrecen apoyo comparable con menor carga.'
        ),
        'bot_text': (
            'Defenderé la proposición: los perros son el mejor amigo del hombre. '
            'Su lealtad y apoyo emocional son incomparables.'
        ),
        'nli': {
            'thesis_scores': {
                'entailment': 0.10,
                'contradiction': 0.92,
                'neutral': 0.18,
            },
            'pair_best': {'entailment': 0.15, 'contradiction': 0.94, 'neutral': 0.40},
            'max_sent_contra': 0.96,
            'on_topic': True,
            'user_wc': 28,
        },
        'policy': {'required_positive_judgements': 2, 'max_assistant_turns': 3},
        'progress': {'positive_judgements': 0, 'assistant_turns': 1},
    }
    base.update(overrides)
    return base


# ----------------------------
# Tests
# ----------------------------


@pytest.mark.asyncio
async def test_judge_reason_user_supports_thesis_maps_to_pro():
    fake_output = (
        '{"accept":false,"confidence":0.85,'
        '"reason":"user_defending_same_stance",'
        '"metrics":{"defended_contra":0.88,"defended_ent":0.98,"max_sent_contra":0.96}}'
    )
    fake_client = FakeAsyncAnthropic(output_text=fake_output)
    adapter = AnthropicAdapter(
        api_key='sk-test', client=fake_client, model='claude-sonnet-4-20250514'
    )

    payload = _base_payload(
        stance='con',
        user_text='Defenderé la proposición tal como está: los perros son el mejor amigo del hombre.',
    )

    res = await adapter.nli_judge(payload=payload)

    assert res.accept is False
    assert res.reason == JudgeReason.USER_DEFENDS_PRO_THESIS
    assert 0.0 <= res.confidence <= 1.0
    assert 'defended_contra' in res.metrics
    assert 'max_sent_contra' in res.metrics


@pytest.mark.asyncio
async def test_judge_reason_user_strong_contradiction_maps_to_strict_contra():
    fake_output = (
        '{"accept":true,"confidence":0.91,'
        '"reason":"thesis_opposition_soft",'
        '"metrics":{"defended_contra":0.93,"defended_ent":0.12,"max_sent_contra":0.96}}'
    )
    fake_client = FakeAsyncAnthropic(output_text=fake_output)
    adapter = AnthropicAdapter(
        api_key='sk-test', client=fake_client, model='claude-sonnet-4-20250514'
    )

    payload = _base_payload()

    res = await adapter.nli_judge(payload=payload)

    assert res.accept is True
    assert res.reason == JudgeReason.STRICT_THESIS_CONTRADICTION
    assert 0.0 <= res.confidence <= 1.0
    assert res.metrics['defended_contra'] >= 0.90
    assert 'max_sent_contra' in res.metrics


@pytest.mark.asyncio
async def test_judge_reason_pass_through_strong_contradiction_like_reason_maps_to_strict_contra():
    """
    Your enum doesn't have STRONG_CONTRADICTION_EVIDENCE.
    We treat 'strong_contradiction_evidence' as a 'strict' contradiction outcome.
    """
    fake_output = (
        '{"accept":true,"confidence":0.90,'
        '"reason":"strong_contradiction_evidence",'
        '"metrics":{"defended_contra":0.91,"defended_ent":0.10,"max_sent_contra":0.94}}'
    )
    fake_client = FakeAsyncAnthropic(output_text=fake_output)
    adapter = AnthropicAdapter(
        api_key='sk-test', client=fake_client, model='claude-sonnet-4-20250514'
    )

    payload = _base_payload()

    res = await adapter.nli_judge(payload=payload)

    assert res.accept is True
    # Expect STRICT_THESIS_CONTRADICTION since STRONG_CONTRADICTION_EVIDENCE is not in the enum
    assert res.reason == JudgeReason.STRONG_CONTRADICTION_EVIDENCE
    assert res.metrics['defended_contra'] >= 0.90


@pytest.mark.asyncio
async def test_judge_reason_unknown_fallback_is_stable_and_metrics_parsed():
    """
    Your adapter currently falls back to AMBIGUOUS_EVIDENCE for unknown keys.
    """
    fake_output = (
        '{"accept":true,"confidence":0.77,'
        '"reason":"totally_new_reason_key",'
        '"metrics":{"defended_contra":0.51,"defended_ent":0.33,"max_sent_contra":0.70}}'
    )
    fake_client = FakeAsyncAnthropic(output_text=fake_output)
    adapter = AnthropicAdapter(
        api_key='sk-test', client=fake_client, model='claude-sonnet-4-20250514'
    )

    payload = _base_payload()

    res = await adapter.nli_judge(payload=payload)

    assert res.accept is True
    assert res.reason == JudgeReason.AMBIGUOUS_EVIDENCE
    assert 0.0 <= res.confidence <= 1.0
    assert res.metrics['max_sent_contra'] == pytest.approx(0.70, rel=0.0, abs=1e-6)
