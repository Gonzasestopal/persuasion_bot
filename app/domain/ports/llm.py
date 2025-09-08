import abc
from typing import List

from app.domain.concession_policy import DebateState
from app.domain.models import Conversation, Message


class LLMPort(abc.ABC):
    @abc.abstractmethod
    async def generate(self, conversation: Conversation, state: DebateState) -> str:
        """
        Given a conversation topic and side,
        return the assistant's reply as plain text.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def debate(self, messages: List[Message], state: DebateState) -> str:
        """
        Given a conversation history (list of Messages),
        return the assistant's reply as plain text.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def check_topic(self, topic: str, stance: str) -> str:
        """
        Given a topic return if its coherent or not
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def nli_judge(self, *, payload: dict) -> dict:
        """Return a dict with keys: verdict, concession, confidence, reason."""
        raise NotImplementedError
