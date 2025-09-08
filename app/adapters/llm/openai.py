# app/adapters/llm/openai_adapter.py
from typing import Iterable, List, Optional

from openai import OpenAI

from app.adapters.llm.constants import (
    MEDIUM_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    Difficulty,
    OpenAIModels,
)
from app.domain.models import Conversation, Message
from app.domain.ports.llm import LLMPort


class OpenAIAdapter(LLMPort):
    def __init__(
        self,
        api_key: str,
        difficulty: Difficulty = Difficulty.EASY,
        client: Optional[OpenAI] = None,
        model: OpenAIModels = OpenAIModels.GPT_4O,
        temperature: float = 0.3,
        max_output_tokens: int = 80,
    ):
        self.client = client or OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.difficulty = difficulty

    @property
    def system_prompt(self) -> str:
        if self.difficulty == Difficulty.MEDIUM:
            return MEDIUM_SYSTEM_PROMPT
        return SYSTEM_PROMPT

    def _build_user_msg(self, topic: str, side: str) -> str:
        return f"You are debating the topic '{topic}'.\nTake the {side} side.\n\n"

    def _request(self, input_msgs: Iterable[dict]) -> str:
        resp = self.client.responses.create(
            model=self.model,
            input=list(input_msgs),
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return resp.output_text

    async def generate(self, conversation: Conversation) -> str:
        user_message = self._build_user_msg(conversation.topic, conversation.side)
        msgs = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_message},
        ]
        return self._request(msgs)

    @staticmethod
    def _map_history(messages: List[Message]) -> List[dict]:
        return [
            {'role': ('assistant' if m.role == 'bot' else 'user'), 'content': m.message}
            for m in messages
        ]

    async def debate(
        self,
        messages: List[Message],
        *,
        scoring_system_msg: Optional[str] = None,  # hidden system message
    ) -> str:
        mapped = self._map_history(messages)

        input_msgs = [{'role': 'system', 'content': self.system_prompt}]
        if scoring_system_msg:
            # separate system message with hidden telemetry
            input_msgs.append({'role': 'system', 'content': scoring_system_msg})
        input_msgs.extend(mapped)

        reply = self._request(input_msgs)
        return reply
