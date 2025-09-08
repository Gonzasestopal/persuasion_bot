# tests/unit/test_message_service_topic_gate.py
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
    def __init__(
        self,
        is_valid: bool,
        reason: str = '',
        normalized_stance: str = None,
        normalized: str = 'God exists',
    ):
        self.is_valid = is_valid
        self.reason = reason
        self.normalized = normalized
        self.normalized_stance = normalized_stance

    def __bool__(self):
        # Allow truthiness checks if any legacy code does `if result:`
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
    # Minimal store; the state object only needs fields touched by start_conversation
    return SimpleNamespace(
        create=Mock(return_value=DebateState(stance='con', topic='X', lang='en')),
        save=Mock(),
    )


@pytest.mark.asyncio
async def test_start_conversation_invalid_topic_raises_invalidtopic(
    repo, llm, debate_store
):
    """
    When topic_checker reports INVALID, MessageService must:
      - await topic_checker.check_topic(message, stance)
      - raise InvalidTopic
      - NOT create a conversation or call the LLM
    """
    parser = Mock(return_value=('X', 'con'))
    topic_checker = SimpleNamespace(
        check_topic=AsyncMock(
            return_value=CheckerResult(False, reason='gibberish', normalized='X')
        )
    )

    svc = MessageService(
        parser=parser,
        repo=repo,
        llm=llm,
        debate_store=debate_store,
        topic_checker=topic_checker,
    )

    user_msg = 'Topic: X. Side: CON.'
    with pytest.raises(InvalidTopic):
        await svc.handle(message=user_msg)

    # Parser used to extract (topic, stance)
    parser.assert_called_once_with(user_msg)

    # Topic checker consulted with (message, stance)
    topic_checker.check_topic.assert_awaited_once_with(user_msg, 'con')

    # No downstream work
    repo.create_conversation.assert_not_called()
    llm.generate.assert_not_awaited()
    repo.add_message.assert_not_awaited()
    repo.last_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_conversation_valid_topic_flows_normally(repo, llm, debate_store):
    """
    When topic_checker reports VALID, normal start flow proceeds
    with normalized topic (and original stance if no normalized_stance provided).
    """
    parser = Mock(return_value=('God exists', 'con'))
    topic_checker = SimpleNamespace(
        check_topic=AsyncMock(
            return_value=CheckerResult(
                True, reason='', normalized='God exists', normalized_stance=None
            )
        )
    )

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=30)
    conv = Conversation(id=42, topic='God exists', stance='con', expires_at=expires_at)
    repo.create_conversation.return_value = conv
    repo.last_messages.return_value = [
        Message(role='user', message='Topic: God exists. Side: CON.'),
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

    user_msg = 'Topic: God exists. Side: CON.'
    out = await svc.handle(message=user_msg)

    # Topic checker used with (message, stance)
    topic_checker.check_topic.assert_awaited_once_with(user_msg, 'con')

    # Conversation created with normalized topic and original stance
    repo.create_conversation.assert_awaited_once_with(topic='God exists', stance='con')

    # LLM and repo writes happened
    llm.generate.assert_awaited_once()
    repo.add_message.assert_has_awaits(
        [
            call(conversation_id=42, role='user', text=user_msg),
            call(conversation_id=42, role='bot', text='bot reply'),
        ]
    )

    assert out == {
        'conversation_id': 42,
        'message': [
            Message(role='user', message='Topic: God exists. Side: CON.'),
            Message(role='bot', message='bot reply'),
        ],
    }


@pytest.mark.asyncio
async def test_start_conversation_uses_normalized_stance_if_provided(
    repo, llm, debate_store
):
    """
    If the topic checker provides a normalized_stance, the service should use it
    when creating the conversation (e.g., mapping synonyms or canonical forms).
    """
    parser = Mock(return_value=('X', 'con'))  # user asked for CON
    topic_checker = SimpleNamespace(
        check_topic=AsyncMock(
            return_value=CheckerResult(
                True,
                reason='',
                normalized='God exists',
                normalized_stance='pro',  # force a different canonical stance
            )
        )
    )

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)
    conv = Conversation(id=99, topic='God exists', stance='pro', expires_at=expires_at)
    repo.create_conversation.return_value = conv
    repo.last_messages.return_value = [
        Message(role='user', message='Topic: X. Side: CON.'),
        Message(role='bot', message='bot reply'),
    ]

    # Modify MessageService to actually use normalized_stance if present.
    # If you've already updated the service to:
    #   stance_to_use = clean_topic.normalized_stance or stance
    # this test will pass.
    svc = MessageService(
        parser=parser,
        repo=repo,
        llm=llm,
        debate_store=debate_store,
        topic_checker=topic_checker,
        history_limit=5,
    )

    user_msg = 'Topic: X. Side: CON.'
    out = await svc.handle(message=user_msg)

    # Topic checker called with (message, stance)
    topic_checker.check_topic.assert_awaited_once_with(user_msg, 'con')

    # Conversation created using normalized topic AND normalized stance
    repo.create_conversation.assert_awaited_once_with(topic='God exists', stance='pro')

    # LLM invoked and messages written
    llm.generate.assert_awaited_once()
    repo.add_message.assert_has_awaits(
        [
            call(conversation_id=99, role='user', text=user_msg),
            call(conversation_id=99, role='bot', text='bot reply'),
        ]
    )

    assert out == {
        'conversation_id': 99,
        'message': [
            Message(role='user', message='Topic: X. Side: CON.'),
            Message(role='bot', message='bot reply'),
        ],
    }
