from copy import deepcopy  # <-- add this import
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, create_autospec

import pytest

from app.domain.concession_policy import DebateState
from app.domain.enums import Stance
from app.domain.errors import (
    ConversationExpired,
    ConversationNotFound,
    InvalidContinuationMessage,
    InvalidStartMessage,
)
from app.domain.models import Conversation, Message
from app.services.concession_service import ConcessionService
from app.services.message_service import MessageService

pytestmark = pytest.mark.unit


@pytest.fixture
def debate_store():
    """
    Minimal debate-state store for tests that call start_conversation.
    Exposes `create` and `save` (sync), with a tiny in-memory backing.
    """
    _mem = {}

    def _create(*, conversation_id: int, stance: Stance, lang: str = 'en', **_):
        # Minimal state object with fields your start flow mutates
        s = SimpleNamespace(
            stance=stance,
            lang=lang,
            lang_locked=False,
            assistant_turns=0,
            positive_judgements=0,
            match_concluded=False,
        )
        _mem[conversation_id] = deepcopy(s)
        return deepcopy(s)

    def _save(*, conversation_id: int, state, **_):
        _mem[conversation_id] = deepcopy(state)

    store = SimpleNamespace(
        create=Mock(side_effect=_create),
        save=Mock(side_effect=_save),
    )
    yield store
    _mem.clear()


@pytest.fixture
def repo():
    expired_time = datetime.now(timezone.utc) + timedelta(minutes=60)
    conversation = Conversation(
        id=123, topic='X', stance='con', expires_at=expired_time
    )
    return SimpleNamespace(
        create_conversation=AsyncMock(return_value=42),  # not used here
        get_conversation=AsyncMock(return_value=conversation),
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(
            return_value=[
                Message(role='user', message='I firmly believe...'),
                Message(role='bot', message='OK'),
            ]
        ),
        all_messages=AsyncMock(
            return_value=[
                Message(role='user', message='I firmly believe...'),
                Message(role='bot', message='OK'),
            ]
        ),
    )


@pytest.fixture
def llm():
    return SimpleNamespace(
        generate=AsyncMock(return_value='bot reply'),
        debate=AsyncMock(return_value='bot msg processing reply'),
    )


@pytest.mark.asyncio
async def test_new_conversation(repo, llm):
    parser = Mock(return_value=('X', 'con'))
    svc = MessageService(parser=parser, repo=repo, llm=llm)
    svc.start_conversation = AsyncMock(return_value={'ok': 'start'})
    svc.continue_conversation = AsyncMock()

    txt = 'Topic: X, Side: con'
    out = await svc.handle(message=txt)

    parser.assert_called_once_with(txt)
    # Your method signature is positional: (topic, stance)
    svc.start_conversation.assert_awaited_once_with('X', 'con', txt)
    svc.continue_conversation.assert_not_called()
    assert out == {'ok': 'start'}


@pytest.mark.asyncio
async def test_continue_conversation(repo, llm):
    parser = Mock(side_effect=AssertionError('parser must not be called on continue'))

    svc = MessageService(parser=parser, repo=repo, llm=llm)
    svc.start_conversation = AsyncMock()
    svc.continue_conversation = AsyncMock(return_value={'ok': 'continue'})

    out = await svc.handle(message='I firmly believe...', conversation_id=123)

    parser.assert_not_called()
    svc.continue_conversation.assert_awaited_once_with('I firmly believe...', 123)
    svc.start_conversation.assert_not_called()
    assert out == {'ok': 'continue'}


@pytest.mark.asyncio
async def test_new_conversation_invalid_message(repo, llm):
    parser = Mock()
    parser.side_effect = InvalidStartMessage(
        'message must contain Topic: and Side: fields'
    )
    service = MessageService(parser=parser, repo=repo, llm=llm)
    service.start_conversation = AsyncMock()
    with pytest.raises(
        InvalidStartMessage, match='message must contain Topic: and Side: fields'
    ):
        await service.handle(message='Message missing params')

    service.start_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_continue_conversation_new_topic_or_side(repo, llm):
    parser = Mock()
    service = MessageService(parser=parser, repo=repo, llm=llm)
    service.continue_conversation = AsyncMock()
    with pytest.raises(
        InvalidContinuationMessage,
        match='topic/side must not be provided when continuing a conversation',
    ):
        await service.handle(message='Topic: X, Side: PRO', conversation_id=123)

    service.continue_conversation.assert_not_called()


@pytest.mark.asyncio
async def test_continue_rejects_topic_marker(repo, llm):
    parser = Mock()
    service = MessageService(parser=parser, repo=repo, llm=llm)
    with pytest.raises(InvalidContinuationMessage, match='must not be provided'):
        await service.handle(message='Topic: Cats. anyway...', conversation_id=1)


@pytest.mark.asyncio
async def test_continue_rejects_side_marker(repo, llm):
    parser = Mock()
    service = MessageService(parser=parser, repo=repo, llm=llm)
    with pytest.raises(InvalidContinuationMessage, match='must not be provided'):
        await service.handle(message='Side: PRO. I think...', conversation_id=1)


@pytest.mark.asyncio
async def test_continue_allows_normal_text_and_no_parser(repo, llm):
    parser = Mock(side_effect=AssertionError('parser must not be called'))
    service = MessageService(parser=parser, repo=repo, llm=llm)
    service.continue_conversation = AsyncMock()
    await service.handle(message='We worked alongside: our peers', conversation_id=7)
    service.continue_conversation.assert_called()


@pytest.mark.asyncio
async def test_continue_with_empty_message(repo, llm):
    parser = Mock(side_effect=AssertionError('parser must not be called'))
    service = MessageService(parser=parser, repo=repo, llm=llm)
    with pytest.raises(InvalidContinuationMessage, match='must not be empty'):
        await service.handle(message='', conversation_id=7)


@pytest.mark.asyncio
async def test_start_writes_messages_and_returns_window(llm, debate_store):
    expires_at = datetime.utcnow()
    conv = Conversation(id=42, topic='X', stance='con', expires_at=expires_at)
    user_message = Message(role='user', message='Topic: X, Side: con')
    bot_message = Message(role='bot', message='bot reply')
    repo = SimpleNamespace(
        create_conversation=AsyncMock(return_value=conv),
        get_conversation=AsyncMock(),
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(
            return_value=[
                user_message,
                bot_message,
            ]
        ),
        all_messages=AsyncMock(
            return_value=[
                user_message,
                bot_message,
            ]
        ),
    )

    parser = Mock(return_value=('X', 'con'))
    svc = MessageService(parser=parser, repo=repo, llm=llm, debate_store=debate_store)

    out = await svc.start_conversation(
        topic='X', stance='con', message='Topic: X, Side: con'
    )

    repo.create_conversation.assert_awaited_once_with(topic='X', stance='con')
    repo.add_message.assert_has_awaits(
        [
            call(conversation_id=42, role='user', text='Topic: X, Side: con'),
            call(conversation_id=42, role='bot', text='bot reply'),
        ]
    )
    repo.last_messages.assert_has_awaits(
        [
            call(conversation_id=42, limit=10),  # final return
        ]
    )
    assert out == {
        'conversation_id': 42,
        'message': [
            user_message,
            bot_message,
        ],
    }


@pytest.mark.asyncio
async def test_continue_conversation_writes_and_returns_window(repo, llm, debate_store):
    user_message = Message(role='user', message='I firmly believe...')
    bot_message = Message(role='bot', message='OK')
    parser = Mock(side_effect=AssertionError('parser must not be called on continue'))
    concession_service = Mock()
    concession_service.analyze_conversation = AsyncMock(
        return_value='bot msg processing reply'
    )
    svc = MessageService(
        parser=parser,
        repo=repo,
        llm=llm,
        history_limit=5,
        concession_service=concession_service,
        debate_store=debate_store,
    )

    out = await svc.continue_conversation(
        message='I firmly believe...', conversation_id=123
    )

    repo.get_conversation.assert_awaited_once_with(conversation_id=123)
    repo.touch_conversation.assert_awaited_once_with(conversation_id=123)
    repo.add_message.assert_has_awaits(
        [
            call(conversation_id=123, role='user', text='I firmly believe...'),
            call(conversation_id=123, role='bot', text='bot msg processing reply'),
        ]
    )

    repo.all_messages.assert_has_awaits(
        [
            call(conversation_id=123),
        ]
    )

    repo.last_messages.assert_has_awaits(
        [
            call(conversation_id=123, limit=10),  # final return
        ]
    )
    assert out == {
        'conversation_id': 123,
        'message': [
            user_message,
            bot_message,
        ],
    }


@pytest.mark.asyncio
async def test_continue_conversation_unknown_id_raises_keyerror(llm):
    repo = SimpleNamespace(
        create_conversation=AsyncMock(),
        get_conversation=AsyncMock(return_value=None),  # not found / expired
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(),
        all_messages=AsyncMock(),
    )
    parser = Mock()
    svc = MessageService(parser=parser, repo=repo, llm=llm)

    with pytest.raises(ConversationNotFound, match='not found'):
        await svc.continue_conversation(message='hi', conversation_id=9999)

    repo.get_conversation.assert_awaited_once_with(conversation_id=9999)
    repo.touch_conversation.assert_not_called()
    repo.add_message.assert_not_called()
    repo.last_messages.assert_not_called()


@pytest.mark.asyncio
async def test_continue_conversation_respects_history_limit(llm):
    user_message = Message(role='user', message='hi')
    bot_message = Message(role='bot', message='bot reply')
    expired_time = datetime.now(timezone.utc) + timedelta(minutes=60)
    conversation = Conversation(
        id=123, topic='X', stance='con', expires_at=expired_time
    )
    repo = SimpleNamespace(
        create_conversation=AsyncMock(),
        get_conversation=AsyncMock(return_value=conversation),
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(
            return_value=[
                user_message,
                bot_message,
            ]
        ),
        all_messages=AsyncMock(
            return_value=[
                user_message,
                bot_message,
            ]
        ),
    )

    parser = Mock(side_effect=AssertionError('parser must not be called'))
    concession_service = Mock()
    concession_service.analyze_conversation = AsyncMock(
        return_value='bot msg processing reply'
    )
    svc = MessageService(
        parser=parser,
        repo=repo,
        llm=llm,
        history_limit=2,
        concession_service=concession_service,
    )

    out = await svc.continue_conversation(message='hi', conversation_id=123)

    repo.get_conversation.assert_awaited_once_with(conversation_id=123)
    repo.touch_conversation.assert_awaited_once_with(conversation_id=123)
    repo.add_message.assert_has_awaits(
        [
            call(conversation_id=123, role='user', text='hi'),
            call(conversation_id=123, role='bot', text='bot msg processing reply'),
        ]
    )
    repo.all_messages.assert_has_awaits(
        [
            call(conversation_id=123),  # history for LLM
        ]
    )
    # history_limit=2 → 2 * 2 = 4 messages window
    repo.last_messages.assert_has_awaits(
        [
            call(conversation_id=123, limit=4),  # final return
        ]
    )
    assert out == {
        'conversation_id': 123,
        'message': [
            user_message,
            bot_message,
        ],
    }


@pytest.mark.asyncio
async def test_continue_conversation_expired(repo, llm):
    expired_time = datetime.now(timezone.utc) - timedelta(minutes=1)
    conversation = Conversation(
        id=123, topic='X', stance='con', expires_at=expired_time
    )
    repo.get_conversation.return_value = conversation

    svc = MessageService(parser=Mock(), repo=repo, llm=llm)

    with pytest.raises(ConversationExpired, match='expired'):
        await svc.continue_conversation('hello', 123)

    repo.touch_conversation.assert_not_called()
    repo.add_message.assert_not_called()


@pytest.mark.asyncio
async def test_start_conversation_calls_llm_and_stores_reply():
    expires_at = datetime.utcnow()
    conv = Conversation(id=42, topic='X', stance='con', expires_at=expires_at)
    state = create_autospec(DebateState, instance=True)
    state.stance = 'con'
    state.match_concluded = False
    state.assistant_turns = 0
    state.lang = 'es'
    debate_store = SimpleNamespace(
        create=Mock(return_value=state),
        save=Mock(),
    )
    repo = SimpleNamespace(
        create_conversation=AsyncMock(return_value=conv),
        get_conversation=AsyncMock(),
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(
            side_effect=[
                [
                    Message(role='user', message='Topic: X, Side: con'),
                    Message(role='bot', message='Hello from LLM'),
                ],  # final return
            ]
        ),
        all_messages=AsyncMock(
            return_value=[
                Message(role='user', message='Topic: X, Side: con'),
                Message(role='bot', message='Hello from LLM'),
            ]
        ),
    )

    parser = Mock(return_value=('X', 'con'))
    llm = AsyncMock()
    llm.generate.return_value = 'Hello from LLM'

    svc = MessageService(parser=parser, repo=repo, llm=llm, debate_store=debate_store)

    out = await svc.start_conversation('X', 'con', 'Topic: X, Side: con')

    llm.generate.assert_awaited_once_with(
        conversation=conv,
        state=state,
    )

    repo.add_message.assert_has_awaits(
        [
            call(conversation_id=42, role='user', text='Topic: X, Side: con'),
            call(conversation_id=42, role='bot', text='Hello from LLM'),
        ]
    )

    assert out['message'][-1].message == 'Hello from LLM'


@pytest.mark.asyncio
async def test_continue_conversation_retrieves_all_messages(llm):
    initial_message = Message(
        role='user', message='Topic: Dogs are human best friend, side:pro'
    )
    stance_message = Message(
        role='user',
        message="I will gladly take the PRO stance that dogs are indeed human's best friend. Dogs offer unwavering loyalty and companionship, often providing emotional support and enhancing human well-being.",
    )
    user_message = Message(role='user', message='I firmly believe...')
    bot_message = Message(role='bot', message='OK')
    expired_time = datetime.now(timezone.utc) + timedelta(minutes=60)
    conversation = Conversation(
        id=123, topic='X', stance='con', expires_at=expired_time
    )
    repo = SimpleNamespace(
        create_conversation=AsyncMock(return_value=conversation),
        get_conversation=AsyncMock(return_value=conversation),
        touch_conversation=AsyncMock(),
        add_message=AsyncMock(),
        last_messages=AsyncMock(
            return_value=[
                user_message,
                bot_message,
            ]
        ),
        all_messages=AsyncMock(
            return_value=[
                initial_message,
                stance_message,
                user_message,
                bot_message,
            ]
        ),
    )

    parser = Mock(side_effect=AssertionError('parser must not be called on continue'))
    concession_service = Mock()
    concession_service.analyze_conversation = AsyncMock(
        return_value='bot msg processing reply'
    )
    svc = MessageService(
        parser=parser,
        repo=repo,
        llm=llm,
        history_limit=1,
        concession_service=concession_service,
    )

    out = await svc.continue_conversation(
        message='I firmly believe...', conversation_id=123
    )

    repo.get_conversation.assert_awaited_once_with(conversation_id=123)
    repo.touch_conversation.assert_awaited_once_with(conversation_id=123)
    repo.add_message.assert_has_awaits(
        [
            call(conversation_id=123, role='user', text='I firmly believe...'),
            call(conversation_id=123, role='bot', text='bot msg processing reply'),
        ]
    )
    repo.last_messages.assert_has_awaits(
        [
            call(conversation_id=123, limit=2),
        ]
    )
    repo.all_messages.assert_has_awaits(
        [
            call(conversation_id=123),
        ]
    )
    assert out == {
        'conversation_id': 123,
        'message': [
            user_message,
            bot_message,
        ],
    }


@pytest.mark.asyncio
async def test_continue_conversation_calls_concession_service(repo, llm):
    parser = Mock()
    conversation_id = 123
    concession_service = Mock(spec=ConcessionService)
    messages = await repo.all_messages()
    conversation = await repo.get_conversation(conversation_id=conversation_id)
    svc = MessageService(
        parser=parser, repo=repo, llm=llm, concession_service=concession_service
    )
    await svc.continue_conversation(message='I firmly believe...', conversation_id=123)
    concession_service.analyze_conversation.assert_awaited_once_with(
        messages=messages,
        stance=conversation.stance,
        conversation_id=conversation_id,
        topic=conversation.topic,
    )


# tests/unit/test_message_service_topic_gate.py
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from app.domain.concession_policy import DebateState
from app.domain.errors import TopicInvalid  # <-- your new domain error
from app.domain.models import Conversation, Message
from app.services.message_service import MessageService

pytestmark = pytest.mark.unit


@pytest.fixture
def repo():
    # Minimal repo with async methods
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
    # Minimal debate-store used by start_conversation flow
    return SimpleNamespace(
        create=Mock(return_value=DebateState(stance='con', topic='X', lang='en')),
        save=Mock(),
    )


@pytest.mark.asyncio
async def test_start_conversation_invalid_topic_raises_topicinvalid(
    repo, llm, debate_store
):
    """
    When topic_checker reports INVALID, the service should raise TopicInvalid and
    avoid creating a conversation or calling the LLM.
    """
    parser = Mock(return_value=('X', 'con'))
    topic_checker = SimpleNamespace(
        check_topic=AsyncMock(
            return_value={'is_valid': False, 'reason': 'gibberish', 'raw': 'INVALID'}
        )
    )

    svc = MessageService(
        parser=parser,
        repo=repo,
        llm=llm,
        debate_store=debate_store,
        topic_checker=topic_checker,
    )

    with pytest.raises(TopicInvalid):
        await svc.handle(message='Topic: X. Side: CON.')

    # Parser is used to extract topic/side
    parser.assert_called_once()

    # Topic checker consulted exactly once
    topic_checker.check_topic.assert_awaited()

    # No conversation / LLM work on invalid topic
    repo.create_conversation.assert_not_called()
    llm.generate.assert_not_awaited()
    repo.add_message.assert_not_awaited()
    repo.last_messages.assert_not_awaited()


@pytest.mark.asyncio
async def test_start_conversation_valid_topic_flows_normally(repo, llm, debate_store):
    """
    When topic_checker reports VALID, the normal start flow proceeds:
    conversation is created, user & bot messages are written, and a window is returned.
    """
    parser = Mock(return_value=('X', 'con'))
    topic_checker = SimpleNamespace(
        check_topic=AsyncMock(
            return_value={'is_valid': True, 'reason': '', 'raw': 'VALID'}
        )
    )

    # Prepare repo return values for a successful start
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

    # Topic checker used
    topic_checker.check_topic.assert_awaited()

    # Conversation created and messages written
    repo.create_conversation.assert_awaited_once_with(topic='X', stance='con')
    llm.generate.assert_awaited_once_with(conversation=conv, state=pytest.anything())
    repo.add_message.assert_has_awaits(
        [
            call(conversation_id=42, role='user', text='Topic: X. Side: CON.'),
            call(conversation_id=42, role='bot', text='bot reply'),
        ]
    )

    # Window returned
    assert out == {
        'conversation_id': 42,
        'message': [
            Message(role='user', message='Topic: X. Side: CON.'),
            Message(role='bot', message='bot reply'),
        ],
    }
