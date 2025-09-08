from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from openai import OpenAI

from app.adapters.llm.constants import (
    AWARE_SYSTEM_PROMPT,
    END_SYSTEM_PROMPT,
    Difficulty,
    OpenAIModels,
)
from app.adapters.llm.types import JudgeResult, TopicResult
from app.domain.concession_policy import DebateState
from app.domain.enums import Stance
from app.domain.models import Conversation, Message
from app.domain.ports.llm import LLMPort

Jsonable = Union[Dict[str, Any], Mapping[str, Any]]


def _fmt(template: str, vars: dict) -> str:
    # Curly-brace formatting, consistent with other prompts in your codebase
    return template.format(**vars)


class OpenAIAdapter(LLMPort):
    """
    Adapter that renders the system prompt from authoritative DebateState
    and calls the OpenAI Responses API. Supports normal debate and end-only rendering.
    """

    def __init__(
        self,
        api_key: str,
        difficulty: Difficulty = Difficulty.EASY,
        client: Optional[OpenAI] = None,
        model: OpenAIModels = OpenAIModels.GPT_4O,
        temperature: float = 0.3,
        max_output_tokens: int = 160,
    ):
        self.client = client or OpenAI(api_key=api_key)
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.difficulty = difficulty

    # ---------- prompt helpers ----------

    @property
    def pro_system_prompt(self) -> str:
        # Example only; not used in flow. Keep for parity with existing interfaces.
        return self._render_system_prompt(
            state=DebateState(stance='pro', topic='god exists', lang='en')
        )

    @property
    def con_system_prompt(self) -> str:
        return self._render_system_prompt(
            state=DebateState(stance='con', topic='god exists', lang='en')
        )

    def _render_system_prompt(self, state: DebateState) -> str:
        if state is None:
            raise ValueError('DebateState is required, got None')
        return AWARE_SYSTEM_PROMPT.format(**state.to_prompt_vars())

    def _build_user_msg(self, topic: str, stance: Stance) -> str:
        return (
            f"You are debating the proposition: '{topic}'. "
            f'Defend it exactly as written. Do not switch sides or negate it. '
            f'Make a concise, evidence-backed case that it is true. '
            f"Do not mention 'pro' or 'con' unless explicitly asked."
        )

    # ---------- low-level request ----------

    def _request(
        self,
        input_msgs: Iterable[dict],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        model: Optional[str] = None,
    ) -> str:
        """
        Thin wrapper over the Responses API with sensible defaults and overrides.
        """
        resp = self.client.responses.create(
            model=(model or self.model),
            input=list(input_msgs),
            temperature=self._pick_temp(temperature),
            max_output_tokens=self._pick_max_tokens(max_tokens),
        )
        return resp.output_text

    def _pick_temp(self, override: Optional[float]) -> float:
        try:
            return float(self.temperature if override is None else override)
        except Exception:
            return self.temperature

    def _pick_max_tokens(self, override: Optional[int]) -> int:
        try:
            return int(self.max_output_tokens if override is None else override)
        except Exception:
            return self.max_output_tokens

    # ---------- public methods ----------

    async def generate(self, conversation: Conversation, state: DebateState) -> str:
        system_prompt = self._render_system_prompt(state)
        user_message = self._build_user_msg(conversation.topic, conversation.stance)
        msgs = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message},
        ]
        return self._request(msgs)

    @staticmethod
    def _map_history(messages: List[Message]) -> List[dict]:
        return [
            {'role': ('assistant' if m.role == 'bot' else 'user'), 'content': m.message}
            for m in messages
        ]

    async def debate_aware(
        self,
        messages: List[Message],
        state: DebateState,
        **gen_kwargs,
    ) -> str:
        """
        Normal debate rendering with the AWARE_SYSTEM_PROMPT.
        Accepts overrides: temperature, max_tokens, stop, model.
        """
        system_prompt = AWARE_SYSTEM_PROMPT.format(**state.to_prompt_vars())
        mapped = self._map_history(messages)
        input_msgs = [{'role': 'system', 'content': system_prompt}, *mapped]
        return self._request(input_msgs, **self._sanitize_gen_kwargs(gen_kwargs))

    async def debate(
        self,
        messages: List[Message],
        state: DebateState,
        **gen_kwargs,
    ) -> str:
        """
        Back-compat alias that also uses AWARE_SYSTEM_PROMPT.
        """
        system_prompt = self._render_system_prompt(state)
        mapped = self._map_history(messages)
        input_msgs = [{'role': 'system', 'content': system_prompt}, *mapped]
        return self._request(input_msgs, **self._sanitize_gen_kwargs(gen_kwargs))

    async def debate_aware_end(
        self,
        messages: List[Message],
        prompt_vars: dict,
        **gen_kwargs,
    ) -> str:
        """
        End-only renderer. Uses END_SYSTEM_PROMPT and (by default) stricter generation:
          temperature=0.2, max_tokens=80.
        You can override via kwargs: temperature, max_tokens, stop, model.
        """
        system_prompt = _fmt(END_SYSTEM_PROMPT, prompt_vars)

        # Keep messages minimal to avoid drifting back into debate mode.
        # For deterministic output, we only send the system message.
        convo = [{'role': 'system', 'content': system_prompt}]

        # Defaults for end mode; allow explicit overrides from caller.
        defaults = {'temperature': 0.2, 'max_tokens': 80, 'stop': None}
        kwargs = {**defaults, **self._sanitize_gen_kwargs(gen_kwargs)}

        return self._request(convo, **kwargs)

    async def check_topic(self, topic: str, language: str = 'en') -> dict:
        raise NotImplementedError

    async def nli_judge(self, *, payload: Jsonable) -> JudgeResult:
        # Implement if/when you move judge to OpenAI; currently Anthropic is used.
        raise NotImplementedError

    # ---------- util ----------

    @staticmethod
    def _sanitize_gen_kwargs(k: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map service-level kwargs to the Responses API parameters we support.
        Accepts keys: temperature, max_tokens, stop, model.
        Ignores unknown keys to keep callers flexible.
        """
        out: Dict[str, Any] = {}
        if 'temperature' in k:
            out['temperature'] = k['temperature']
        if 'max_tokens' in k:
            # service uses max_tokens; Responses API is max_output_tokens, but we normalize in _request
            out['max_tokens'] = k['max_tokens']
        if 'stop' in k:
            out['stop'] = k['stop']
        if 'model' in k:
            out['model'] = k['model']
        return out
