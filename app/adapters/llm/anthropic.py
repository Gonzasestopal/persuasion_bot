import json
import re
from typing import Iterable, List, Optional

from anthropic import AsyncAnthropic

from app.adapters.llm.constants import (
    MEDIUM_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TOPIC_CHECKER_SYSTEM_PROMPT,
    AnthropicModels,
    Difficulty,
)
from app.adapters.llm.types import TopicResult
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

    def _topic_gate_system_prompt(self, topic, stance) -> str:
        # Keep this tiny and strict; output must be a single line.
        return TOPIC_CHECKER_SYSTEM_PROMPT.replace('{TOPIC}', topic).replace(
            '{STANCE}', stance
        )

    async def check_topic(self, topic: str, stance: str) -> TopicResult:
        """
        Returns TopicResult shaped like:
            {
            "is_valid": True|False,
            "reason": "<one-liner if invalid or ''>",
            "normalized": "<normalized claim or None>",
            "raw": "<model raw text>",
            "stance_normalized": "<pro|con or None>",
            }
        """
        sys = self._topic_gate_system_prompt(topic=topic, stance=stance)

        # IMPORTANT: Send the inputs exactly as the system prompt expects.
        # (Uppercase keys to reduce brittleness.)
        user_prompt = (
            f'TOPIC: {topic}\n'
            f'STANCE: {stance}\n'
            'Return exactly one line as specified in the system instructions.'
        )

        resp = await self.client.messages.create(
            model=self.model,
            system=sys,
            messages=[
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': user_prompt}],
                }
            ],
            temperature=0.0,
            max_tokens=max(
                self.max_output_tokens, 64
            ),  # allow enough room for the JSON
        )

        out = ''.join(
            block.text
            for block in resp.content
            if getattr(block, 'type', None) == 'text'
        ).strip()

        # 1) INVALID (strict localized one-liner). Keep raw line in 'reason'.
        up = out.upper()
        if up.startswith('INVALID:'):
            return TopicResult(
                is_valid=False,
                normalized=None,
                reason=out,
                normalized_stance=None,  # no stance if invalid
                raw=out,
            )
        try:
            obj = json.loads(out)
        except json.JSONDecodeError:
            # Unrecognized format
            return TopicResult(
                is_valid=False,
                normalized=None,
                reason='unrecognized',
                normalized_stance=None,
                raw=out,
            )

        # Validate minimal schema (be strict but helpful)
        required_keys = {
            'status',
            'topic_normalized',
            'stance_final',
        }

        if not isinstance(obj, dict) or required_keys - obj.keys():
            print('rip', obj.keys())
            return TopicResult(
                is_valid=False,
                normalized=None,
                reason='unrecognized',
                normalized_stance=None,
                raw=out,
            )

        if obj.get('status') != 'VALID':
            # If the model returned JSON but not VALID, treat as invalid text.
            return TopicResult(
                is_valid=False,
                normalized=None,
                reason='unrecognized',
                normalized_stance=None,
                raw=out,
            )

        normalized: str = obj.get('topic_normalized') or ''
        stance_final: Optional[str] = obj.get('stance_final')
        if not normalized.strip() or stance_final not in {'pro', 'con'}:
            return TopicResult(
                is_valid=False,
                normalized=None,
                reason='unrecognized',
                normalized_stance=None,
                raw=out,
            )

        return TopicResult(
            is_valid=True,
            normalized=normalized.strip(),
            reason='',
            normalized_stance=stance_final,
            raw=out,
        )
