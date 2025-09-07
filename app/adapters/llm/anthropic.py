from typing import Iterable, List, Optional

from anthropic import AsyncAnthropic

from app.adapters.llm.constants import (
    MEDIUM_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    AnthropicModels,
    Difficulty,
)
from app.domain.enums import Stance
from app.domain.models import Conversation, Message
from app.domain.ports.llm import LLMPort


class AnthropicAdapter(LLMPort):
    def __init__(
        self,
        api_key: str,
        difficulty: Difficulty = Difficulty.EASY,
        client: Optional[AsyncAnthropic] = None,
        model: AnthropicModels = AnthropicModels.CLAUDE_35,
        temperature: float = 0.3,
        max_output_tokens: int = 120,
    ):
        self.client = client or AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.difficulty = difficulty

    @property
    def system_prompt(self) -> str:
        return (
            MEDIUM_SYSTEM_PROMPT
            if self.difficulty == Difficulty.MEDIUM
            else SYSTEM_PROMPT
        )

    def _build_user_msg(self, topic: str, stance: Stance) -> str:
        return f"You are debating the topic '{topic}'. Take the {stance} stance."

    @staticmethod
    def _map_history(messages: List[Message]) -> List[dict]:
        # Domain -> Anthropic roles
        return [
            {
                'role': ('assistant' if m.role == 'bot' else 'user'),
                'content': [{'type': 'text', 'text': m.message}],
            }
            for m in messages
        ]

    async def _request(self, *, messages: Iterable[dict], system: str) -> str:
        resp = await self.client.messages.create(
            model=self.model,
            system=system,
            messages=list(messages),
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )
        # Join text blocks from the response
        return ''.join(
            block.text
            for block in resp.content
            if getattr(block, 'type', None) == 'text'
        )

    async def generate(self, conversation: Conversation) -> str:
        user = self._build_user_msg(conversation.topic, conversation.stance)
        msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': user}]}]
        return await self._request(messages=msgs, system=self.system_prompt)

    async def debate(self, messages: List[Message]) -> str:
        mapped = self._map_history(messages)
        return await self._request(messages=mapped, system=self.system_prompt)

    def _topic_gate_system_prompt(self) -> str:
        # Keep this tiny and strict; output must be a single line.
        return (
            'You are a strict topic gate for a debate system. '
            'Classify if the proposed topic is debate-ready.\n'
            'Debate-ready: a clear proposition one can argue for/against; not greeting; not gibberish; '
            'not trivial (≤2 content words).\n'
            "Valid examples: 'God exists', 'Sports build character', 'Climate change is real'.\n"
            "Invalid examples: 'hi', 'hello', 'asdf???', '!!!', 'ok'.\n"
            'Output exactly one line:\n'
            "- 'VALID'\n"
            "- or 'INVALID: <short reason>'\n"
        )

    async def check_topic(self, topic: str, language: str = 'en') -> dict:
        """
        Returns:
            {
              "is_valid": "true" | "false",
              "reason": "<short reason or empty>",
              "raw": "<model raw text>"
            }
        """
        sys = self._topic_gate_system_prompt()
        user_text = (
            f'Topic: {topic}\n'
            f'Language: {language}\n'
            "Return exactly 'VALID' or 'INVALID: <reason>'."
        )
        resp = await self.client.messages.create(
            model=self.model,
            system=sys,
            messages=[
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': user_text}],
                }
            ],
            temperature=0.0,  # deterministic
            max_tokens=8,  # tiny budget
        )
        out = ''.join(
            block.text
            for block in resp.content
            if getattr(block, 'type', None) == 'text'
        ).strip()

        up = out.upper()
        if up.startswith('VALID'):
            return {'is_valid': 'true', 'reason': '', 'raw': out}
        if up.startswith('INVALID'):
            reason = out.split(':', 1)[1].strip() if ':' in out else ''
            return {'is_valid': 'false', 'reason': reason, 'raw': out}
        # Fallback if the model misbehaves
        return {'is_valid': 'false', 'reason': 'unrecognized', 'raw': out}
