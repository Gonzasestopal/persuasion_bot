# app/domain/ports/llm.py
from typing import List, Optional

from app.domain.models import Conversation, Message


class LLMPort:
    async def generate(self, conversation: Conversation) -> str:
        raise NotImplementedError

    async def debate(
        self,
        messages: List[Message],
        *,
        scoring_system_msg: Optional[str] = None,  # hidden <SCORING>{...}</SCORING>
    ) -> str:
        raise NotImplementedError
