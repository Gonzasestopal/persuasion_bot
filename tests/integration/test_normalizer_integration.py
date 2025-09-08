import os
import re
import unicodedata

import pytest

from app.adapters.llm.anthropic import AnthropicAdapter

pytestmark = pytest.mark.integration


def _norm_upper(s: str) -> str:
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace('’', "'").replace('“', '"').replace('”', '"')
    return re.sub(r'\s+', ' ', s).strip().upper()


requires_key = pytest.mark.skipif(
    not os.environ.get('ANTHROPIC_API_KEY'),
    reason='ANTHROPIC_API_KEY not set; skipping live Anthropic integration test.',
)


@requires_key
@pytest.mark.asyncio
async def test_topic_normalizer_valid_en_double_negation_normalizes_anthropic():
    """
    Integration (Anthropic):
    Input with hedge + double negation should normalize to a clean claim.
    """
    adapter = AnthropicAdapter(
        api_key=os.environ['ANTHROPIC_API_KEY'],
        model='claude-sonnet-4-20250514',
    )

    topic = 'I don’t think God does not exist'
    res = await adapter.check_topic(topic, stance='pro')

    assert res.is_valid == True
    assert isinstance(res.normalized, str) and res.normalized.strip()
    assert _norm_upper(res.normalized) in {'GOD EXISTS'}, (
        f'Unexpected normalization: {res.normalized} (raw={res.raw})'
    )

    assert res.reason == ''
    assert 'VALID' in _norm_upper(res.raw)
    assert '\n' not in res.raw


@requires_key
@pytest.mark.asyncio
async def test_topic_normalizer_valid_minimal_claim_round_trip_anthropic():
    """
    Integration (Anthropic):
    Clean claim should pass through unchanged.
    """
    adapter = AnthropicAdapter(
        api_key=os.environ['ANTHROPIC_API_KEY'],
        model='claude-sonnet-4-20250514',
    )

    topic = 'Climate change is real'
    res = await adapter.check_topic(topic, stance='con')

    assert res.is_valid == True
    assert _norm_upper(res.normalized) == 'CLIMATE CHANGE IS REAL'
    assert 'VALID' in _norm_upper(res.raw)
    assert '\n' not in res.raw
