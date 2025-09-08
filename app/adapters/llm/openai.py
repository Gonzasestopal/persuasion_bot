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
    def system_prompt(self):
        if self.difficulty == Difficulty.MEDIUM:
            return MEDIUM_SYSTEM_PROMPT
        return SYSTEM_PROMPT

    def _build_user_msg(self, topic: str, side: str):
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
        context_footer: Optional[str] = None,  # <-- NEW
    ) -> str:
        mapped = self._map_history(messages)

        # If provided, append footer at the end of the system prompt.
        system = self.system_prompt
        if context_footer:
            # Ensure it’s on its own line and does not interfere with rules.
            system = f'{system.rstrip()}\n\n{context_footer}'

        input_msgs = [{'role': 'system', 'content': system}]
        input_msgs.extend(mapped)
        return self._request(input_msgs)
