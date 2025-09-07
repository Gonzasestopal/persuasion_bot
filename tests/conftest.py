# conftest.py
import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

load_dotenv()


# Ensure env flags are set BEFORE importing app.main (so lifespan won't open DB)
@pytest.fixture(autouse=True, scope='session')
def _set_global_env():
    os.environ.setdefault('DISABLE_DB_POOL', 'true')
    os.environ.setdefault('USE_INMEMORY_REPO', 'true')
    yield


@pytest.fixture()
def service():
    """
    Build a fresh MessageService and dependencies for EACH TEST.
    This prevents cross-test leakage of debate state and messages.
    """
    # Local imports to avoid importing app.main before env is set
    from app.adapters.llm.dummy import DummyLLMAdapter
    from app.adapters.llm.openai import OpenAIAdapter
    from app.adapters.nli.hf_nli import HFNLIProvider
    from app.adapters.repositories.memory import InMemoryMessageRepo
    from app.adapters.repositories.memory_debate_store import InMemoryDebateStore
    from app.domain.parser import (
        parse_topic_side,
    )  # adjust if your parser lives elsewhere
    from app.services.concession_service import ConcessionService
    from app.services.message_service import MessageService
    from app.settings import settings

    repo = InMemoryMessageRepo()
    debate_store = InMemoryDebateStore()
    nli = HFNLIProvider()

    if OpenAIAdapter and os.environ.get('OPENAI_API_KEY'):
        llm = OpenAIAdapter(
            api_key=settings.OPENAI_API_KEY,
            model=settings.LLM_MODEL,
            temperature=0.3,
        )
    else:
        llm = DummyLLMAdapter()

    concession_service = ConcessionService(
        llm=llm,
        nli=nli,
        debate_store=debate_store,
    )

    return MessageService(
        parser=parse_topic_side,
        repo=repo,
        llm=llm,
        debate_store=debate_store,
        concession_service=concession_service,
    )


@pytest.fixture()
def client(service):
    """
    A TestClient using a per-test service via FastAPI dependency override.
    """
    from app.infra.service import get_service
    from app.main import app

    app.dependency_overrides[get_service] = lambda: service
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()
