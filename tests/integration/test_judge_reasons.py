# tests/integration/test_judge_reasons.py
import pytest

from app.adapters.llm.anthropic import AnthropicAdapter
from app.domain.nli.reasons import JudgeReason

# ----------------------------
# Fakes that match adapter usage
# ----------------------------

pytestmark = pytest.mark.integration


class _FakeMessages:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.calls = []

    async def create(self, *args, **kwargs):
        # record the call for optional inspection
        self.calls.append((args, kwargs))
        # Shape expected by AnthropicAdapter._parse_single_text:
        # resp["content"][0]["text"] (with a type)
        return {'content': [{'type': 'text', 'text': self.output_text}]}


class FakeAsyncAnthropic:
    """
    Minimal fake matching Anthropic client surface that the adapter uses:
      self.client.messages.create(...)
    """

    def __init__(self, output_text: str):
        self.messages = _FakeMessages(output_text)


# ----------------------------
# Tests
# ----------------------------


@pytest.mark.asyncio
async def test_judge_reason_user_supports_thesis_maps_to_pro():
    """
    Simula LLM con reason legacy 'user_defending_same_stance' y verifica
    que el adapter lo normaliza a USER_DEFENDS_PRO_THESIS.
    """
    fake_output = (
        '{"accept":false,"confidence":0.85,'
        '"reason":"user_defending_same_stance",'
        '"metrics":{"defended_contra":0.88,"defended_ent":0.98,"max_sent_contra":0.96}}'
    )
    fake_client = FakeAsyncAnthropic(output_text=fake_output)

    adapter = AnthropicAdapter(
        api_key='sk-test',
        client=fake_client,
        model='claude-sonnet-4-20250514',
    )

    payload = {
        'topic': 'los perros son el mejor amigo del hombre',
        'stance': 'con',
        'language': 'es',
        'turn_index': 1,
        'user_text': (
            'Defenderé la proposición tal como está: los perros son el mejor amigo del hombre.'
        ),
        'bot_text': 'Los perros han sido compañeros leales de los humanos durante miles de años.',
        'nli': {
            'thesis_scores': {
                'entailment': 0.97,
                'contradiction': 0.87,
                'neutral': 0.11,
            },
            'pair_best': {'entailment': 0.22, 'contradiction': 0.97, 'neutral': 0.57},
            'max_sent_contra': 0.95,
            'on_topic': True,
            'user_wc': 32,
        },
        'policy': {'required_positive_judgements': 2, 'max_assistant_turns': 3},
        'progress': {'positive_judgements': 0, 'assistant_turns': 1},
    }

    res = await adapter.nli_judge(payload=payload)

    assert res.accept is False
    assert res.reason == JudgeReason.USER_DEFENDS_PRO_THESIS
    assert 0.0 <= res.confidence <= 1.0
    assert isinstance(res.metrics, dict)
    assert 'defended_contra' in res.metrics
    assert 'max_sent_contra' in res.metrics


@pytest.mark.asyncio
async def test_judge_reason_user_strong_contradiction_maps_to_strict_contra():
    """
    Simula contradicción fuerte y reason legacy 'thesis_opposition_soft';
    el adapter debe normalizar a STRICT_THESIS_CONTRADICTION y ACCEPT=true.
    """
    fake_output = (
        '{"accept":true,"confidence":0.91,'
        '"reason":"thesis_opposition_soft",'
        '"metrics":{"defended_contra":0.93,"defended_ent":0.12,"max_sent_contra":0.96}}'
    )
    fake_client = FakeAsyncAnthropic(output_text=fake_output)

    adapter = AnthropicAdapter(
        api_key='sk-test',
        client=fake_client,
        model='claude-sonnet-4-20250514',
    )

    payload = {
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

    res = await adapter.nli_judge(payload=payload)

    assert res.accept is True
    assert res.reason == JudgeReason.STRICT_THESIS_CONTRADICTION
    assert 0.0 <= res.confidence <= 1.0
    assert isinstance(res.metrics, dict)
    assert res.metrics['defended_contra'] >= 0.90
    assert 'max_sent_contra' in res.metrics
