from typing import Optional

from pydantic import AnyUrl, Field, field_validator
from pydantic_settings import BaseSettings

from app.adapters.llm.constants import Difficulty, OpenAIModels, Provider


class Settings(BaseSettings):
    DATABASE_URL: AnyUrl
    HISTORY_LIMIT: int = 5
    EXPIRES_MINUTES: int = 60
    REQUEST_TIMEOUT_S: int = 25
    POOL_MIN: int = 1
    POOL_MAX: int = 10
    OPENAI_API_KEY: Optional[str] = None
    LLM_PROVIDER: Optional[Provider] = Provider.OPENAI
    LLM_MODEL: str = OpenAIModels.GPT_4O
    DIFFICULTY: Optional[Difficulty]
    USE_INMEMORY_REPO: bool = Field(default=False, env='USE_INMEMORY_REPO')
    DISABLE_DB_POOL: bool = Field(default=False, env='DISABLE_DB_POOL')
    ANTHROPIC_API_KEY: Optional[str] = None

    # Behavior
    LLM_TEMPERATURE: float = 0.3
    MAX_OUTPUT_TOKENS: int = 120
    LLM_PER_PROVIDER_TIMEOUT_S: float = 12.0

    MAX_ASSISTANT_TURNS: int = 6
    REQUIRED_POSITIVE_JUDGEMENTS: int = 2

    class Config:
        env_file = '.env'

    @field_validator('LLM_PROVIDER', mode='before')
    def allow_blank(cls, v):
        if v == '' or v is None:
            return None
        return v

    @field_validator('DIFFICULTY', mode='before')
    def allow_blank_difficulty(cls, v):
        if v == '' or v is None:
            return None
        return v

    @field_validator('OPENAI_API_KEY', mode='before')
    def allow_blank_openai(cls, v):
        if v == '' or v is None:
            return None
        return v

    @field_validator('ANTHROPIC_API_KEY', mode='before')
    def allow_blank_anthropic(cls, v):
        if v == '' or v is None:
            return None
        return v


settings = Settings()
