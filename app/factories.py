from functools import lru_cache
from typing import Optional

from fastapi import Depends

from app.adapters.llm.anthropic import AnthropicAdapter
from app.adapters.llm.constants import (
    AnthropicModels,
    Difficulty,
    OpenAIModels,
    Provider,
)
from app.adapters.llm.dummy import DummyLLMAdapter
from app.adapters.llm.fallback import FallbackLLM
from app.adapters.llm.openai import OpenAIAdapter
from app.domain.errors import ConfigError
from app.domain.parser import parse_topic_side
from app.repositories.base import get_repo
from app.services.concession_service import ConcessionService
from app.services.message_service import MessageService
from app.settings import settings


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
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
            api_key=settings.OPENAI_API_KEY,
            model=AnthropicModels.CLAUDE_35,
            difficulty=difficulty,
        )

    return DummyLLMAdapter()


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
        model=AnthropicModels.CLAUDE_35,
        max_output_tokens=settings.MAX_OUTPUT_TOKENS,  # your existing budget
        difficulty=settings.DIFFICULTY,
    )


def make_fallback_llm():
    # Choose primary/secondary from settings
    primary_name = (settings.PRIMARY_LLM or 'openai').lower()
    secondary_name = (settings.SECONDARY_LLM or 'claude').lower()

    provider_map = {
        'openai': make_openai,
        'claude': make_claude,
        'anthropic': make_claude,  # alias
    }

    primary = provider_map[primary_name]()
    secondary = provider_map[secondary_name]()

    return FallbackLLM(
        primary=primary,
        secondary=secondary,
        per_provider_timeout_s=settings.LLM_PER_PROVIDER_TIMEOUT_S,  # e.g., 12
        mode='sequential',
        logger=lambda msg: None,  # plug logger if you want
    )


@lru_cache(maxsize=1)
def get_llm_singleton():
    # Build once per process
    return make_openai()


@lru_cache(maxsize=1)
def get_concession_singleton():
    # Share the LLM singleton inside the ConcessionService singleton
    llm = get_llm_singleton()
    return ConcessionService(llm=llm)


def get_service(
    repo=Depends(get_repo),
    llm=Depends(get_llm_singleton),
    concession=Depends(get_concession_singleton),
) -> MessageService:
    return MessageService(
        parser=parse_topic_side, repo=repo, llm=llm, concession_service=concession
    )
