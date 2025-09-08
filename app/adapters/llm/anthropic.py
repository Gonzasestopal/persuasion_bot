import json
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from anthropic import AsyncAnthropic

from app.adapters.llm.constants import (
    AWARE_SYSTEM_PROMPT,
    END_SYSTEM_PROMPT,
    JUDGE_SCORE_SYSTEM_PROMPT,
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

Jsonable = Union[Dict[str, Any], Mapping[str, Any]]

logger = logging.getLogger(__name__)


def _stance_str(value: Union[Stance, str]) -> str:
    """
    Normalize stance to lowercase 'pro'|'con' for prompts and to_string contexts.
    """
    if isinstance(value, Stance):
        return value.value.lower()
    return str(value).strip().lower()


def _stance_token_for_system(value: Union[Stance, str]) -> str:
    """
    Some system prompts may expect uppercase tokens: PRO|CON.
    """
    s = _stance_str(value)
    return 'PRO' if s == 'pro' else 'CON'


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
        self.client: AsyncAnthropic = client or AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.difficulty = difficulty

    # -----------------------------
    # Prompt selection
    # -----------------------------
    @property
    def system_prompt(self) -> str:
        return (
            MEDIUM_SYSTEM_PROMPT
            if self.difficulty == Difficulty.MEDIUM
            else SYSTEM_PROMPT
        )

    # -----------------------------
    # Helpers
    # -----------------------------
    @staticmethod
    def _build_user_msg(topic: str, stance: Union[Stance, str]) -> str:
        stance_norm = _stance_str(stance)
        return f"You are debating the topic '{topic}'. Take the {stance_norm} stance."

    @staticmethod
    def _map_history(messages: List[Message]) -> List[dict]:
        # Domain -> Anthropic roles
        mapped: List[dict] = []
        for m in messages:
            role = 'assistant' if m.role == 'bot' else 'user'
            mapped.append(
                {'role': role, 'content': [{'type': 'text', 'text': m.message}]}
            )
        return mapped

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
        """
        Join text blocks from Anthropic response into a single string.
        Handles both object-attributes and dict-like access defensively.
        """
        content: Optional[Sequence[Any]] = getattr(resp, 'content', None)
        if not isinstance(content, (list, tuple)):
            return ''

        parts: List[str] = []
        for block in content:
            # SDK objects: block.type == 'text' and block.text
            btype = getattr(block, 'type', None)
            if btype == 'text':
                text = getattr(block, 'text', '')
                if isinstance(text, str):
                    parts.append(text)
                continue

            # Fallback for dict-like shapes
            if isinstance(block, dict) and block.get('type') == 'text':
                t = block.get('text', '')
                if isinstance(t, str):
                    parts.append(t)

        return ''.join(parts)

    # -----------------------------
    # High-level generation APIs
    # -----------------------------
    async def generate(self, conversation: Conversation) -> str:
        user = self._build_user_msg(conversation.topic, conversation.stance)
        msgs = [{'role': 'user', 'content': [{'type': 'text', 'text': user}]}]
        return await self._request(messages=msgs, system=self.system_prompt)

    async def debate(self, messages: List[Message]) -> str:
        mapped = self._map_history(messages)
        return await self._request(messages=mapped, system=self.system_prompt)

    async def debate_aware(self, messages: List[Message], state) -> str:
        """
        Uses a system prompt that is aware of debate state.
        Expects `state.to_prompt_vars()` -> dict with keys used by AWARE_SYSTEM_PROMPT.
        """
        sys = AWARE_SYSTEM_PROMPT.format(**state.to_prompt_vars())
        mapped = self._map_history(messages)
        return await self._request(messages=mapped, system=sys)

    def _topic_gate_system_prompt(self, topic: str, stance: Union[Stance, str]) -> str:
        """
        Keep this tiny and strict; output must be a single line. We inject normalized tokens.
        """
        return TOPIC_CHECKER_SYSTEM_PROMPT.replace('{TOPIC}', topic).replace(
            '{STANCE}', _stance_token_for_system(stance)
        )

    async def debate_aware_end(
        self, messages: List[Message], prompt_vars: dict, **gen_kwargs
    ) -> str:
        """
        Generate the final/ending line with a minimal convo to avoid drifting back into debate.
        Re-uses the Anthropic client; no dependency on any OpenAI function.
        """
        system_prompt = _fmt(END_SYSTEM_PROMPT, prompt_vars)

        # Optionally include just the last user turn (commented out by default)
        convo: List[dict] = [
            {'role': 'system', 'content': system_prompt},
            # If you want a tiny bit of lexical carryover from the last user turn:
            # {"role": "user", "content": [{"type": "text", "text": messages[-1].message}]} if messages else None,
        ]
        convo = [m for m in convo if m is not None]

        resp = await self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[
                m for m in convo if m.get('role') != 'system'
            ],  # system goes in `system` arg above
            temperature=float(gen_kwargs.get('temperature', 0.0)),
            max_tokens=int(gen_kwargs.get('max_tokens', 120)),
        )
        return self._parse_single_text(resp).strip()

    # -----------------------------
    # Topic checking
    # -----------------------------
    async def check_topic(self, topic: str, stance: Union[Stance, str]) -> TopicResult:
        sys = self._topic_gate_system_prompt(topic=topic, stance=stance)

        stance_req = _stance_str(stance)
        logger.info(
            '[topic_gate.req] topic_raw=%r stance_requested=%s model=%s',
            topic,
            stance_req,
            self.model,
        )

        user_prompt = (
            f'TOPIC: {topic}\n'
            f'STANCE: {stance_req}\n'
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

        # Log raw one-line model output, truncated
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

        # Try to parse JSON
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
            or (required_keys - obj.keys())
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
            stance_req,
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

    # -----------------------------
    # NLI judge (score-only)
    # -----------------------------
    async def nli_judge(self, *, payload: Jsonable) -> JudgeResult:
        """
        Score-only judge. Expects ONE-LINE JSON:
          {"accept":true|false,"confidence":0..1,"reason":"snake_case", "metrics":{...optional...}}
        No assistant_reply. No ended flag (server decides end).
        """
        user_text = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
        logger.debug('[nli_score] sending payload=%s', user_text)

        resp = await self.client.messages.create(
            model=self.model,
            system=JUDGE_SCORE_SYSTEM_PROMPT,  # score-only prompt
            messages=[
                {'role': 'user', 'content': [{'type': 'text', 'text': user_text}]}
            ],
            temperature=0.0,
            max_tokens=160,
        )
        out = self._parse_single_text(resp).strip()
        logger.debug('[nli_score] raw LLM output=%r', out)

        try:
            obj = json.loads(out)
        except json.JSONDecodeError as e:
            logger.error('[nli_score] JSON decode failed: %s | output=%r', e, out)
            raise ValueError(f'LLM judge returned non-JSON: {out!r}') from e

        if not isinstance(obj, dict):
            raise ValueError('LLM score: non-object JSON')

        # Minimal required fields
        accept = bool(obj.get('accept', False))
        reason = str(obj.get('reason') or '')
        try:
            confidence = float(obj.get('confidence', 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        # Optional extras (ignore if missing)
        metrics = obj.get('metrics') if isinstance(obj.get('metrics'), dict) else None

        if not reason:
            raise ValueError('LLM score: missing reason')
        if not (0.0 <= confidence <= 1.0):
            confidence = max(0.0, min(1.0, confidence))

        result = JudgeResult(
            accept=accept,
            confidence=confidence,
            reason=reason,
            metrics=metrics,
        )

        logger.debug(
            '[nli_judge] parsed JudgeResult: accept=%s reason=%s conf=%.2f',
            result.accept,
            result.reason,
            result.confidence,
        )
        return result


def _fmt(template: str, vars: dict) -> str:
    """
    Simple curly-brace formatting compatible with your other prompts.
    """
    return template.format(**vars)
