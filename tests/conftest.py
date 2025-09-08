# tests/conftest.py
import os

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

from app.domain.parser import parse_topic_side

# Load env first (OPENAI_API_KEY, etc.)
load_dotenv()


@pytest.fixture()
def service():
    """
    Build a fresh MessageService and dependencies for EACH TEST.
    This prevents cross-test leakage of debate state and messages.
    """
    # Local imports to avoid importing app.main before env is set
    # IMPORTANT: set flags BEFORE importing app/factories/settings so nothing tries to init a DB
    os.environ.setdefault('USE_INMEMORY_REPO', '1')
    os.environ.setdefault('DISABLE_DB_POOL', '1')

    from app.adapters.llm.dummy import DummyLLMAdapter
    from app.adapters.llm.openai import OpenAIAdapter  # adjust import if different
    from app.adapters.nli.hf_nli import HFNLIProvider
    from app.adapters.repositories.memory import InMemoryMessageRepo
    from app.services.concession_service import ConcessionService
    from app.services.message_service import MessageService
    from app.settings import settings

    repo = InMemoryMessageRepo()
    nli = HFNLIProvider()

    if os.environ.get('OPENAI_API_KEY'):
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
    )

    return MessageService(
        parser=parse_topic_side,
        repo=repo,
        llm=llm,
        concession_service=concession_service,
    )


@pytest.fixture()
def client(service):
    """
    Your original fixture, with a tiny tweak: accept *args/**kwargs so
    FastAPI can pass Request if your dependency signature expects it.
    """
    from app.factories import get_service
    from app.main import app

    # Some FastAPI deps are called with the Request param;
    # make the override tolerant to that.
    app.dependency_overrides[get_service] = lambda: service
    try:
        with TestClient(app) as c:
            # Optional: also reset app.state in-mem stores so other code paths use the same fresh instances
            app.state.inmem_repo = service.repo
            app.state.inmem_debate_store = service.debate_store
            yield c
    finally:
        app.dependency_overrides.clear()
