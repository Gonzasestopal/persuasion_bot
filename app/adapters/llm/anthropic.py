import json
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from anthropic import AsyncAnthropic

from app.adapters.llm.constants import (
    JUDGE_SYSTEM_PROMPT,
    MEDIUM_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    TOPIC_CHECKER_SYSTEM_PROMPT,
    AnthropicModels,
    Difficulty,
)
from app.adapters.llm.types import JudgeResult, TopicResult
from app.domain.enums import Stance
from app.domain.models import Conversation, Message
from app.domain.ports.llm import LLMPort

Jsonable = Union[
    Dict[str, Any], Mapping[str, Any]
]  # or your specific payload dataclass via asdict()


logger = logging.getLogger(__name__)


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
        return self._parse_single_text(resp)

    @staticmethod
    def _parse_single_text(resp) -> str:
        # Join text blocks from the response into a single string
        return ''.join(
            block.text
            for block in getattr(resp, 'content', [])
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
        sys = self._topic_gate_system_prompt(topic=topic, stance=stance)

        logger.info(
            '[topic_gate.req] topic_raw=%r stance_requested=%s model=%s',
            topic,
            stance,
            self.model,
        )

        user_prompt = (
            f'TOPIC: {topic}\n'
            f'STANCE: {stance}\n'
            'Return exactly one line as specified in the system instructions.'
        )

        resp = await self.client.messages.create(
            model=self.model,
            system=sys,
            messages=[
                {'role': 'user', 'content': [{'type': 'text', 'text': user_prompt}]}
            ],
            temperature=0.0,
            max_tokens=max(self.max_output_tokens, 64),
        )
        out = self._parse_single_text(resp).strip()

        # Log the raw one-line model output, truncated (to avoid log spam)
        logger.debug(
            '[topic_gate.raw_out] %s', (out[:300] + '…') if len(out) > 300 else out
        )

        up = out.upper()
        if up.startswith('INVALID:'):
            logger.info('[topic_gate.result] status=INVALID reason=%r', out)
            return TopicResult(
                is_valid=False,
                normalized=None,
                reason=out,
                normalized_stance=None,
                raw=out,
            )

        try:
            obj = json.loads(out)
        except json.JSONDecodeError:
            logger.warning('[topic_gate.parse_error] non_json_out=%r', out[:120])
            return TopicResult(
                is_valid=False,
                normalized=None,
                reason='unrecognized',
                normalized_stance=None,
                raw=out,
            )

        required_keys = {'status', 'topic_normalized', 'stance_final'}
        if (
            not isinstance(obj, dict)
            or required_keys - obj.keys()
            or obj.get('status') != 'VALID'
        ):
            logger.warning(
                '[topic_gate.schema_mismatch] obj_keys=%s',
                sorted(obj.keys()) if isinstance(obj, dict) else type(obj),
            )
            return TopicResult(
                is_valid=False,
                normalized=None,
                reason='unrecognized',
                normalized_stance=None,
                raw=out,
            )

        normalized = (obj.get('topic_normalized') or '').strip()
        stance_final = (obj.get('stance_final') or '').strip().lower()
        pol_raw = (obj.get('polarity_raw') or '').strip().lower()
        pol_norm = (obj.get('polarity_normalized') or '').strip().lower()

        if not normalized or stance_final not in {'pro', 'con'}:
            logger.warning(
                '[topic_gate.bad_values] normalized=%r stance_final=%r',
                normalized,
                stance_final,
            )
            return TopicResult(
                is_valid=False,
                normalized=None,
                reason='unrecognized',
                normalized_stance=None,
                raw=out,
            )

        logger.info(
            '[topic_gate.result] status=VALID lang=%s pol_raw=%s pol_norm=%s stance_req=%s stance_final=%s topic_norm=%r',
            obj.get('lang'),
            pol_raw,
            pol_norm,
            stance,
            stance_final,
            normalized,
        )

        return TopicResult(
            is_valid=True,
            normalized=normalized,
            reason='',
            normalized_stance=stance_final,
            raw=out,
        )

    async def nli_judge(self, *, payload: Jsonable) -> JudgeResult:
        """
        Calls the judge with the NLI payload and returns a structured JudgeResult:
          - accept: bool
          - ended: bool
          - reason: str
          - assistant_reply: str
          - confidence: float (0..1)
        Raises ValueError on invalid responses.
        """
        user_text = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))

        resp = await self.client.messages.create(
            model=self.model,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': user_text}],
                }
            ],
            temperature=0.0,
            max_tokens=220,
        )

        out = self._parse_single_text(resp).strip()
        try:
            obj = json.loads(out)
        except json.JSONDecodeError as e:
            raise ValueError(f'LLM judge returned non-JSON: {out!r}') from e

        if not isinstance(obj, dict):
            raise ValueError('LLM judge: non-object JSON')

        # Required fields
        if 'assistant_reply' not in obj or 'reason' not in obj:
            raise ValueError('LLM judge: missing assistant_reply or reason')

        accept = bool(obj.get('accept', False))
        ended = bool(obj.get('ended', False))
        reason = str(obj.get('reason') or '')
        assistant_reply = str(obj.get('assistant_reply') or '')
        try:
            confidence = float(obj.get('confidence', 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        # Basic validation
        if not assistant_reply or not reason:
            raise ValueError('LLM judge: empty assistant_reply or reason')
        if confidence < 0.0 or confidence > 1.0:
            confidence = max(0.0, min(1.0, confidence))

        return JudgeResult(
            accept=accept,
            ended=ended,
            reason=reason,
            assistant_reply=assistant_reply,
            confidence=confidence,
        )
