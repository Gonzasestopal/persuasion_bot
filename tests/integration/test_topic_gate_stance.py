# tests/integration/test_topic_normalizer_topicresult_anthropic.py

import os
import re
import unicodedata

import pytest

from app.adapters.llm.anthropic import AnthropicAdapter
from app.adapters.llm.types import TopicResult  # or from where you keep it

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
async def test_topic_normalizer_valid_en_double_negation_returns_topicresult_anthropic():
    """
    Integration (Anthropic):
    Input with hedge + double negation should normalize to a clean claim and return TopicResult.
    """
    adapter = AnthropicAdapter(
        api_key=os.environ['ANTHROPIC_API_KEY'],
        model='claude-sonnet-4-20250514',
    )

    topic = 'I don’t think God does not exist'
    res = await adapter.check_topic(topic, stance='con')

    assert isinstance(res, TopicResult)
    assert res.is_valid is True
    assert isinstance(res.normalized, str) and res.normalized.strip()
    assert _norm_upper(res.normalized) in {'GOD EXISTS'}, (
        f'Unexpected normalization: {res.normalized!r}'
    )
    # reason should be empty string (or None if you prefer—keep consistent with your implementation)
    assert res.reason in ('', None)
    # normalized_stance is optional until you compute stance in the checker
    assert res.normalized_stance in ('con', None)


@requires_key
@pytest.mark.asyncio
async def test_topic_normalizer_invalid_es_one_liner_topicresult_anthropic():
    """
    Integration (Anthropic):
    'hola' should yield an INVALID one-liner; language is auto-detected.
    Accept ES or EN per the spec.
    """
    adapter = AnthropicAdapter(
        api_key=os.environ['ANTHROPIC_API_KEY'],
        model='claude-sonnet-4-20250514',
    )

    topic = 'hola'
    res = await adapter.check_topic(topic, stance='pro')

    assert isinstance(res, TopicResult)
    assert res.is_valid is False
    assert res.normalized is None
    assert isinstance(res.reason, str) and res.reason.strip()

    raw_up = _norm_upper(res.reason)
    assert raw_up.startswith('INVALID:')

    es_ok = 'NO ES UN TEMA LISTO PARA DEBATE' in raw_up
    en_ok = "ISN'T DEBATE-READY. PLEASE PROVIDE A VALID, DEBATE-READY TOPIC." in raw_up

    assert es_ok or en_ok, f'Unexpected INVALID line: {res.reason!r}'

    # Optional 2nd sentence check — only if Spanish path:
    if es_ok and 'POR FAVOR, PROPORCIONA' in raw_up:
        assert 'POR FAVOR, PROPORCIONA UN TEMA VALIDO Y LISTO PARA DEBATE' in raw_up


@requires_key
@pytest.mark.asyncio
async def test_topic_normalizer_valid_minimal_claim_round_trip_topicresult_anthropic():
    """
    Integration (Anthropic):
    Clean claim passes through unchanged; TopicResult reflects it.
    """
    adapter = AnthropicAdapter(
        api_key=os.environ['ANTHROPIC_API_KEY'],
        model='claude-sonnet-4-20250514',
    )

    topic = 'Climate change is real'
    res = await adapter.check_topic(topic, stance='pro')

    assert isinstance(res, TopicResult)
    assert res.is_valid is True
    assert _norm_upper(res.normalized) == 'CLIMATE CHANGE IS REAL'
    assert res.reason in ('', None)
