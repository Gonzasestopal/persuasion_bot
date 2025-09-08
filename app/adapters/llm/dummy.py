from typing import List

from app.adapters.llm.types import JudgeResult
from app.domain.concession_policy import DebateState
from app.domain.models import Conversation, Message
from app.domain.ports.llm import LLMPort


class DummyLLMAdapter(LLMPort):
    async def generate(self, conversation: Conversation, state: DebateState) -> str:
        return f'I am a bot that defends {conversation.topic} and im stance {conversation.stance}'

    async def debate(self, messages: List[Message], state: DebateState) -> str:
        return f'After considering your last {len(messages)} messages, this is not totally correct.'

    async def check_topic(topic: str) -> bool:
        return True

    async def nli_judge(self, *, payload) -> JudgeResult:
        return JudgeResult(
            verdict='OPPOSITE',
            concession=True,
            confidence=0.87,
            reason='thesis_opposition_strong',
        )
