from datetime import datetime, timezone
from typing import Optional

from app.domain.enums import Stance
from app.domain.errors import ConversationExpired, ConversationNotFound, InvalidTopic
from app.domain.parser import assert_no_topic_or_side_markers
from app.domain.ports.debate_store import DebateStorePort
from app.domain.ports.llm import LLMPort
from app.domain.ports.message_repo import MessageRepoPort
from app.services.concession_service import ConcessionService
from app.utils.lang import parse_language_line


class MessageService(object):
    def __init__(
        self,
        parser,
        repo: MessageRepoPort,
        concession_service: Optional[ConcessionService] = None,
        llm: Optional[LLMPort] = None,
        topic_checker: Optional[LLMPort] = None,
        debate_store: Optional[DebateStorePort] = None,
        history_limit=5,
    ):
        self.parser = parser
        self.repo = repo
        self.debate_store = debate_store
        self.history_limit = history_limit
        self.llm = llm
        self.topic_checker = topic_checker
        self.concession_service = concession_service or ConcessionService(
            llm=llm,
            debate_store=self.debate_store,
        )

    async def handle(self, message: str, conversation_id: Optional[int] = None):
        if conversation_id is None:
            topic, stance = self.parser(message)
            return await self.start_conversation(topic, stance, message)

        assert_no_topic_or_side_markers(message)
        return await self.continue_conversation(message, conversation_id)

    async def start_conversation(self, topic: str, stance: Stance, message: str = None):
        clean_topic = await self.topic_checker.check_topic(topic, stance)

        if not clean_topic.is_valid:
            raise InvalidTopic(message=clean_topic.reason)

        conversation = await self.repo.create_conversation(
            topic=clean_topic.normalized, stance=clean_topic.normalized_stance
        )

        await self.repo.add_message(
            conversation_id=conversation.id, role='user', text=message
        )

        state = self.debate_store.create(
            conversation_id=conversation.id,
            stance=clean_topic.normalized_stance,
            topic=conversation.topic,
            lang=None,
        )

        raw_reply = await self.llm.generate(conversation=conversation, state=state)

        lang, clean_reply = parse_language_line(raw_reply)
        state.lang = lang or 'en'
        state.lang_locked = True
        state.assistant_turns += 1
        self.debate_store.save(conversation_id=conversation.id, state=state)

        await self.repo.add_message(
            conversation_id=conversation.id, role='bot', text=clean_reply
        )

        return {
            'conversation_id': conversation.id,
            'message': await self.repo.last_messages(
                conversation_id=conversation.id, limit=self.history_limit * 2
            ),
        }

    async def continue_conversation(self, message: str, conversation_id: int):
        conversation = await self.repo.get_conversation(conversation_id=conversation_id)

        if not conversation:
            raise ConversationNotFound('conversation_id not found or expired')

        if conversation.expires_at <= datetime.now(timezone.utc):
            raise ConversationExpired('conversation_id expired')

        cid = conversation.id
        await self.repo.touch_conversation(conversation_id=cid)
        await self.repo.add_message(conversation_id=cid, role='user', text=message)

        full_history = await self.repo.all_messages(conversation_id=cid)

        reply = await self.concession_service.analyze_conversation(
            messages=full_history,
            stance=conversation.stance,
            conversation_id=conversation_id,
            topic=conversation.topic,
        )

        await self.repo.add_message(conversation_id=cid, role='bot', text=reply)

        return {
            'conversation_id': cid,
            'message': await self.repo.last_messages(
                conversation_id=cid, limit=self.history_limit * 2
            ),
        }
