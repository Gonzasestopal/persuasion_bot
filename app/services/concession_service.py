# app/services/concession_service.py
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.adapters.nli.hf_nli import HFNLIProvider
from app.domain.models import Message
from app.domain.ports.llm import LLMPort
from app.domain.ports.scoring import ScoreJudgePort, ScoreVerdict
from app.services.scoring import (
    RunningScores,
    build_context_footer,
    build_score_footer,
    deterministic_verdict_from_eval,
    features_from_last_eval,
    join_footers,
    judge_last_two_messages,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass(frozen=True)
class _NLIConfig:
    model_name: str = 'roberta-large-mnli'
    entailment_threshold: float = 0.65
    contradiction_threshold: float = 0.70
    max_length: int = 512
    max_claims_per_turn: int = 3


class ConcessionService:
    """
    Thin orchestrator: maps history, calls scoring helpers, updates in-memory
    aggregates per conversation, and appends ephemeral footers to debate LLM.
    """

    def __init__(
        self,
        llm: LLMPort,
        nli: Optional[HFNLIProvider] = None,
        config: _NLIConfig = _NLIConfig(),
        score_judge: Optional[ScoreJudgePort] = None,
    ) -> None:
        self.llm = llm
        self.nli = nli or HFNLIProvider()
        self.config = config
        self.entailment_threshold = config.entailment_threshold
        self.contradiction_threshold = config.contradiction_threshold
        self.score_judge = score_judge

        # in-memory running scores keyed by conversation_id
        self._scores: Dict[int, RunningScores] = {}

    async def analyze_conversation(
        self,
        messages: List[Message],
        side: str,
        conversation_id: int,
        topic: str,
    ) -> str:
        mapped = self._map_history(messages)

        # Score only the last assistant→user pair
        last_eval = judge_last_two_messages(
            mapped,
            side=side,
            topic=topic,
            nli=self.nli,
            entailment_threshold=self.entailment_threshold,
            contradiction_threshold=self.contradiction_threshold,
        )

        context_footer: Optional[str] = None
        if last_eval:
            # features → score_judge (optional) → deterministic fallback
            features = features_from_last_eval(
                last_eval,
                side=side,
                entailment_threshold=self.entailment_threshold,
                contradiction_threshold=self.contradiction_threshold,
            )

            verdict: Optional[ScoreVerdict] = None
            if self.score_judge:
                try:
                    verdict = await self.score_judge.score(features=features)
                except Exception:
                    logger.exception('score_judge failed; using deterministic fallback')

            if verdict is None:
                verdict = deterministic_verdict_from_eval(
                    last_eval,
                    entailment_threshold=self.entailment_threshold,
                    contradiction_threshold=self.contradiction_threshold,
                )

            # update in-memory running totals
            rs = self._scores.setdefault(conversation_id, RunningScores())
            rs.update(
                align=verdict['alignment'],
                ts=last_eval['thesis_scores'],
                ps=last_eval['scores'],
            )

            # build ephemeral footers
            footer_context = build_context_footer(
                {
                    'alignment': verdict['alignment'],
                    'concession': verdict['concession'],
                    'reason': verdict['reason'],
                    'scores': last_eval['scores'],
                    'thesis_scores': last_eval['thesis_scores'],
                    'topic': topic,
                }
            )
            footer_score = build_score_footer(rs)
            context_footer = join_footers(footer_context, footer_score)

        # call debate LLM with injected footer(s)
        reply = await self.llm.debate(messages=messages, context_footer=context_footer)
        return reply.strip()

    @staticmethod
    def _map_history(messages: List[Message]) -> List[dict]:
        return [
            {'role': ('assistant' if m.role == 'bot' else 'user'), 'content': m.message}
            for m in messages
        ]
