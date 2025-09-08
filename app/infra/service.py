from fastapi import Depends

from app.domain.parser import parse_topic_side
from app.domain.ports.llm import LLMPort
from app.domain.ports.nli import NLIPort
from app.infra.db import get_repo
from app.infra.debate_store import get_state_store
from app.infra.llm import (
    get_judge_singleton,
    get_llm_singleton,
    get_topic_checker_singleton,
)
from app.infra.nli import get_nli_singleton
from app.services.concession_service import ConcessionService
from app.services.debate_orchestrator import MessageService


def get_concession_singleton(
    nli: NLIPort = Depends(get_nli_singleton),
    llm: LLMPort = Depends(get_llm_singleton),
    judge: LLMPort = Depends(get_judge_singleton),
) -> ConcessionService:
    return ConcessionService(llm=llm, nli=nli, judge=judge)


def get_service(
    repo=Depends(get_repo),
    llm: LLMPort = Depends(get_llm_singleton),
    concession: ConcessionService = Depends(get_concession_singleton),
    topic_checker: LLMPort = Depends(get_topic_checker_singleton),
    debate_store=Depends(get_state_store),
) -> MessageService:
    return MessageService(
        parser=parse_topic_side,
        repo=repo,
        llm=llm,
        concession_service=concession,
        topic_checker=topic_checker,
        debate_store=debate_store,
    )
