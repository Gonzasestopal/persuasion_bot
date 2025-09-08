from functools import lru_cache
from typing import Optional

from app.adapters.llm.anthropic import AnthropicAdapter
from app.adapters.llm.constants import (
    AnthropicModels,
    Difficulty,
    OpenAIModels,
    Provider,
)
from app.adapters.llm.dummy import DummyLLMAdapter
from app.adapters.llm.openai import OpenAIAdapter
from app.domain.errors import ConfigError
from app.domain.ports.llm import LLMPort
from app.settings import settings


def make_openai():
    if not settings.OPENAI_API_KEY:
        raise ConfigError('OPENAI_API_KEY is required for provider=openai')

    return OpenAIAdapter(
        api_key=settings.OPENAI_API_KEY,
        model=settings.LLM_MODEL,  # e.g., "gpt-4o"
        max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        difficulty=settings.DIFFICULTY,
    )


def make_claude():
    if not settings.ANTHROPIC_API_KEY:
        raise ConfigError('ANTHROPIC_API_KEY is required for provider=anthropic')

    return AnthropicAdapter(
        api_key=settings.ANTHROPIC_API_KEY,
        model='claude-sonnet-4-20250514',
        max_output_tokens=settings.MAX_OUTPUT_TOKENS,  # your existing budget
        difficulty=settings.DIFFICULTY,
    )


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
    provider = (provider or settings.LLM_PROVIDER.value).lower()
    model = model or settings.LLM_MODEL

    if provider == Provider.ANTHROPIC.value and not settings.ANTHROPIC_API_KEY:
        raise ConfigError('ANTHROPIC_API_KEY is required for provider=anthropic')

    if provider == Provider.OPENAI.value and not settings.OPENAI_API_KEY:
        raise ConfigError('OPENAI_API_KEY is required for provider=openai')

    if not settings.DIFFICULTY:
        raise ConfigError('DIFFICULTY is required')

    try:
        difficulty = Difficulty(settings.DIFFICULTY.strip().lower())
    except ValueError:
        raise ConfigError('ONLY EASY AND MEDIUM DIFFICULTY ARE SUPPORTED')

    if provider == Provider.OPENAI.value:
        wanted_model = (model or OpenAIModels.GPT_4O.value).strip().lower()
        try:
            model_enum = OpenAIModels(wanted_model)
        except ValueError:
            raise ConfigError(f'{wanted_model} is not a valid OpenAI model')
        return OpenAIAdapter(
            api_key=settings.OPENAI_API_KEY,
            model=model_enum,
            difficulty=difficulty,
        )

    elif provider == Provider.ANTHROPIC.value:
        return AnthropicAdapter(
            api_key=settings.ANTHROPIC_API_KEY,
            model=AnthropicModels.CLAUDE_35,
            difficulty=difficulty,
        )

    return DummyLLMAdapter()


@lru_cache(maxsize=1)
def get_llm_singleton() -> LLMPort:
    # Build once per process
    return get_llm()


def reset_llm_singleton_cache():
    get_llm_singleton.cache_clear()


@lru_cache(maxsize=1)
def get_topic_checker_singleton() -> LLMPort:
    return get_topic_checker()


def get_topic_checker() -> LLMPort:
    return make_claude()


@lru_cache(maxsize=1)
def get_judge_singleton() -> LLMPort:
    return get_judge()


def get_judge() -> LLMPort:
    return make_claude()
