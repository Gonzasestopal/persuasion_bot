from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from app.domain.concession_policy import DebateState
from app.domain.errors import InvalidTopic
from app.domain.models import Conversation, Message
from app.services.message_service import MessageService

pytestmark = pytest.mark.unit


class CheckerResult:
    def __init__(self, is_valid: bool, reason: str = ''):
        self.is_valid = is_valid
        self.reason = reason

    def __bool__(self):
        return self.is_valid


@pytest.fixture
def repo():
    return SimpleNamespace(
        create_conversation=AsyncMock(),
        get_conversation=AsyncMock(),
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(),
        all_messages=AsyncMock(),
    )


@pytest.fixture
def llm():
    return SimpleNamespace(
        generate=AsyncMock(return_value='bot reply'),
        debate=AsyncMock(return_value='bot msg processing reply'),
    )


@pytest.fixture
def debate_store():
    return SimpleNamespace(
        create=Mock(return_value=DebateState(stance='con', topic='X', lang='en')),
        save=Mock(),
    )


@pytest.mark.asyncio
async def test_start_conversation_invalid_topic_raises_invalidtopic(
    repo, llm, debate_store
):
    parser = Mock(return_value=('X', 'con'))
    topic_checker = SimpleNamespace(
        check_topic=Mock(return_value=CheckerResult(False, reason='gibberish'))
    )

    svc = MessageService(
        parser=parser,
        repo=repo,
        llm=llm,
        debate_store=debate_store,
        topic_checker=topic_checker,
    )

    with pytest.raises(InvalidTopic):
        await svc.handle(message='Topic: X. Side: CON.')

    parser.assert_called_once()
    topic_checker.check_topic.assert_called_once()

    repo.create_conversation.assert_not_called()
    llm.generate.assert_not_awaited()
    repo.add_message.assert_not_awaited()
    repo.last_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_conversation_valid_topic_flows_normally(repo, llm, debate_store):
    parser = Mock(return_value=('X', 'con'))
    topic_checker = SimpleNamespace(check_topic=Mock(return_value=CheckerResult(True)))

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
    conv = Conversation(id=42, topic='X', stance='con', expires_at=expires_at)
    repo.create_conversation.return_value = conv
    repo.last_messages.return_value = [
        Message(role='user', message='Topic: X. Side: CON.'),
        Message(role='bot', message='bot reply'),
    ]

    svc = MessageService(
        parser=parser,
        repo=repo,
        llm=llm,
        debate_store=debate_store,
        topic_checker=topic_checker,
        history_limit=5,
    )

    out = await svc.handle(message='Topic: X. Side: CON.')

    topic_checker.check_topic.assert_called_once()
    repo.create_conversation.assert_awaited_once_with(topic='X', stance='con')
    llm.generate.assert_awaited_once()
    repo.add_message.assert_has_awaits(
        [
            call(conversation_id=42, role='user', text='Topic: X. Side: CON.'),
            call(conversation_id=42, role='bot', text='bot reply'),
        ]
    )
    assert out == {
        'conversation_id': 42,
        'message': [
            Message(role='user', message='Topic: X. Side: CON.'),
            Message(role='bot', message='bot reply'),
        ],
    }
