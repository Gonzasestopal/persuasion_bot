# tests/unit/test_di_wiring.py
import pytest

from app.domain.parser import parse_topic_side
from app.infra.service import get_concession_singleton, get_service
from app.services.concession_service import ConcessionService
from app.services.message_service import MessageService

pytestmark = pytest.mark.unit


class DummyRepo:
    pass


class DummyDebateStore:
    pass


class DummyNLI:
    pass


class DummyLLM:
    pass


class DummyTopicChecker:
    """Accept any type; just a sentinel for the DI param."""

    pass


def test_get_concession_singleton_returns_service_and_wires():
    repo = DummyRepo()  # not used here, but handy if you extend
    store = DummyDebateStore()
    nli = DummyNLI()
    llm = DummyLLM()

    svc = get_concession_singleton(
        debate_store=store,
        nli=nli,
        llm=llm,
    )
    assert isinstance(svc, ConcessionService)
    # identity checks: uses exactly the same instances we passed in
    assert svc.debate_store is store
    assert svc.nli is nli
    assert svc.llm is llm


def test_get_service_returns_message_service_and_wires_everything():
    repo = DummyRepo()
    store = DummyDebateStore()
    nli = DummyNLI()
    llm = DummyLLM()
    topic_checker = DummyTopicChecker()

    # Build a ConcessionService the same way FastAPI DI would provide
    concession = get_concession_singleton(
        debate_store=store,
        nli=nli,
        llm=llm,
    )

    ms = get_service(
        repo=repo,
        llm=llm,
        concession=concession,
        topic_checker=topic_checker,  # accepted but not stored by MessageService
        debate_store=store,
    )

    assert isinstance(ms, MessageService)
    # parser should be the module-level parse function you wire
    assert ms.parser is parse_topic_side

    # identity checks for injected deps actually used by MessageService
    assert ms.repo is repo
    assert ms.llm is llm
    assert ms.concession_service is concession
    assert ms.debate_store is store

    # Ensure we didn't accidentally stash the topic_checker on the service
    assert not hasattr(ms, 'topic_checker')


def test_get_service_accepts_alternate_instances_without_sharing():
    """
    Demonstrates that get_service will honor whatever instances it is given,
    not recreating or sharing hidden singletons behind your back.
    """
    repo1, repo2 = DummyRepo(), DummyRepo()
    store1, store2 = DummyDebateStore(), DummyDebateStore()
    nli1, llm1, llm2 = DummyNLI(), DummyLLM(), DummyLLM()
    topic_checker1 = DummyTopicChecker()

    concession1 = get_concession_singleton(debate_store=store1, nli=nli1, llm=llm1)
    ms1 = get_service(
        repo=repo1,
        llm=llm1,
        concession=concession1,
        topic_checker=topic_checker1,
        debate_store=store1,
    )

    concession2 = get_concession_singleton(
        debate_store=store2, nli=DummyNLI(), llm=llm2
    )
    ms2 = get_service(
        repo=repo2,
        llm=llm2,
        concession=concession2,
        topic_checker=DummyTopicChecker(),
        debate_store=store2,
    )

    # Distinct objects across invocations
    assert ms1 is not ms2
    assert ms1.repo is repo1 and ms2.repo is repo2
    assert ms1.llm is llm1 and ms2.llm is llm2
    assert ms1.debate_store is store1 and ms2.debate_store is store2
    assert (
        ms1.concession_service is concession1 and ms2.concession_service is concession2
    )
