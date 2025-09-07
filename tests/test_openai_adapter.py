from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.adapters.llm.openai import OpenAIAdapter
from app.domain.concession_policy import DebateState
from app.domain.models import Conversation, Message

pytestmark = pytest.mark.unit


class FakeResponses:
    def __init__(self, calls, output_text=None):
        self.calls = calls
        self._output_text = output_text

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self._output_text)


class FakeClient:
    def __init__(self, calls, output_text='FAKE-OUTPUT'):
        self.responses = FakeResponses(calls, output_text)


def test_adapter_config():
    client = Mock()
    adapter = OpenAIAdapter(
        api_key='test',
        client=client,
    )

    assert adapter.client == client
    assert adapter.temperature == 0.3


@pytest.mark.asyncio
async def test_adapter_generate_builds_prompt_and_returns_output(monkeypatch):
    calls = []
    client = FakeClient(calls)
    expires_at = datetime.utcnow()
    adapter = OpenAIAdapter(
        api_key='sk-test', client=client, model='gpt-4o', temperature=0.3
    )
    state = DebateState(stance='con', topic='god exists', lang='en')
    conv = Conversation(id=1, topic='X', stance='con', expires_at=expires_at)
    out = await adapter.generate(conversation=conv, state=state)

    assert out == 'FAKE-OUTPUT'
    assert len(calls) == 1
    sent = calls[0]

    assert sent['model'] == 'gpt-4o'
    assert sent['temperature'] == 0.3

    msgs = sent['input']
    assert msgs[0] == {'role': 'system', 'content': adapter.con_system_prompt}
    assert msgs[1]['role'] == 'user'
    assert "You are debating the topic 'X'" in msgs[1]['content']
    assert 'Take the con stance.' in msgs[1]['content']


@pytest.mark.asyncio
async def test_adapter_debate_maps_roles_and_respects_history(monkeypatch):
    calls = []
    client = FakeClient(calls)
    adapter = OpenAIAdapter(
        api_key='sk-test', client=client, model='gpt-4o', temperature=0.2
    )

    msgs = [
        Message(role='user', message='u1'),
        Message(role='bot', message='b1'),
        Message(role='user', message='u2'),
        Message(role='bot', message='b2'),
    ]
    state = DebateState(stance='con', topic='god exists', lang='en')
    out = await adapter.debate(messages=msgs, state=state)
    assert out == 'FAKE-OUTPUT'

    assert len(calls) == 1
    sent = calls[0]
    assert sent['model'] == 'gpt-4o'
    assert sent['temperature'] == 0.2

    input_msgs = sent['input']
    assert input_msgs[0] == {'role': 'system', 'content': adapter.con_system_prompt}

    role_map = {'user': 'user', 'bot': 'assistant'}
    assert len(input_msgs) == len(msgs) + 1

    for i, m in enumerate(msgs, start=1):
        assert set(input_msgs[i].keys()) == {'role', 'content'}
        assert input_msgs[i]['role'] == role_map[m.role]
        assert input_msgs[i]['content'] == m.message


@pytest.mark.asyncio
async def test_check_topic_valid_is_not_implemented_yet():
    calls = []
    client = FakeClient(calls, output_text='VALID')
    adapter = OpenAIAdapter(api_key='sk-test', client=client, model='gpt-4o')

    with pytest.raises(NotImplementedError):
        await adapter.check_topic('God exists', language='en')


@pytest.mark.asyncio
async def test_check_topic_invalid_is_not_implemented_yet():
    calls = []
    client = FakeClient(calls, output_text='INVALID: greeting')
    adapter = OpenAIAdapter(api_key='sk-test', client=client, model='gpt-4o')

    with pytest.raises(NotImplementedError):
        await adapter.check_topic('hello', language='en')
