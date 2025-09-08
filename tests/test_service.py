from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from app.domain.concession_policy import DebateState
from app.domain.errors import InvalidTopic
from app.domain.models import Conversation, Message
from app.services.concession_service import ConcessionService
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
    # State only needs fields touched by start_conversation
    return SimpleNamespace(
        create=Mock(return_value=DebateState(stance='con', topic='X', lang='en')),
        save=Mock(),
    )


# NUEVO: fixtures para dependencias requeridas por MessageService
@pytest.fixture
def nli():
    # No se usa en estos tests, puede ser un namespace vacío o con métodos ficticios
    return SimpleNamespace(
        analyze=AsyncMock(return_value={}),  # opcional
    )


@pytest.fixture
def judge():
    # start_conversation no lo usa, pero igual proveemos una firma coherente
    fake_judge_result = SimpleNamespace(
        accept=False, ended=False, reason='init', assistant_reply='', confidence=0.0
    )
    return SimpleNamespace(nli_judge=AsyncMock(return_value=fake_judge_result))


@pytest.mark.asyncio
async def test_start_conversation_invalid_topic_raises_invalidtopic(
    repo, llm, debate_store, nli, judge
):
    """
    If topic_checker returns INVALID:
      - MessageService raises InvalidTopic
      - No conversation is created
      - LLM isn't called
    """
    parser = Mock(return_value=('X', 'con'))
    topic_checker = SimpleNamespace(
        check_topic=AsyncMock(
            return_value=CheckerResult(False, reason='gibberish', normalized='X')
        )
    )

    cs = ConcessionService(
        nli=nli,
        judge=judge,
        llm=llm,
        debate_store=debate_store,
    )

    svc = MessageService(
        parser=parser,
        repo=repo,
        llm=llm,
        debate_store=debate_store,
        topic_checker=topic_checker,
        concession_service=cs,
    )

    user_msg = 'Topic: X. Side: CON.'
    with pytest.raises(InvalidTopic):
        await svc.handle(message=user_msg)

    parser.assert_called_once_with(user_msg)
    topic_checker.check_topic.assert_awaited_once_with('X', 'con')
    repo.create_conversation.assert_not_called()
    llm.generate.assert_not_awaited()
    repo.add_message.assert_not_awaited()
    repo.last_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_conversation_valid_topic_flows_normally(
    repo, llm, debate_store, nli, judge
):
    """
    VALID topic → creates conversation with normalized topic and normalized_stance,
    writes user & bot messages, returns last messages.
    """
    parser = Mock(return_value=('God exists', 'con'))
    topic_checker = SimpleNamespace(
        check_topic=AsyncMock(
            return_value=CheckerResult(
                is_valid=True,
                reason='',
                normalized='God exists',
                normalized_stance='con',
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

    cs = ConcessionService(
        nli=nli,
        judge=judge,
        llm=llm,
        debate_store=debate_store,
    )

    svc = MessageService(
        parser=parser,
        repo=repo,
        concession_service=cs,
        llm=llm,
        debate_store=debate_store,
        topic_checker=topic_checker,
        history_limit=5,
    )

    user_msg = 'Topic: God exists. Side: CON.'
    out = await svc.handle(message=user_msg)

    topic_checker.check_topic.assert_awaited_once_with('God exists', 'con')
    repo.create_conversation.assert_awaited_once_with(topic='God exists', stance='con')
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
    repo, llm, debate_store, nli, judge
):
    """
    If checker provides normalized_stance, service uses it verbatim in create_conversation
    and in DebateState initialization.
    """
    parser = Mock(return_value=('X', 'con'))  # user asked for CON
    topic_checker = SimpleNamespace(
        check_topic=AsyncMock(
            return_value=CheckerResult(
                True,
                reason='',
                normalized='God exists',
                normalized_stance='pro',  # canonical stance differs from input
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

    cs = ConcessionService(
        nli=nli,
        judge=judge,
        llm=llm,
        debate_store=debate_store,
    )

    svc = MessageService(
        parser=parser,
        repo=repo,
        llm=llm,
        concession_service=cs,
        debate_store=debate_store,
        topic_checker=topic_checker,
        history_limit=5,
    )

    user_msg = 'Topic: X. Side: CON.'
    out = await svc.handle(message=user_msg)

    topic_checker.check_topic.assert_awaited_once_with('X', 'con')
    repo.create_conversation.assert_awaited_once_with(topic='God exists', stance='pro')
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
