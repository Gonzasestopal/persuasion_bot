# app/services/concession_service.py
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from app.adapters.nli.hf_nli import HFNLIProvider
from app.domain.models import Message
from app.domain.ports.llm import LLMPort
from app.domain.ports.scoring import ScoreJudgePort, ScoreVerdict
from app.services.scoring import (
    RunningScores,
    build_context_signal,
    build_score_signal,
    deterministic_verdict_from_eval,
    features_from_last_eval,
    judge_last_two_messages,
    make_scoring_system_message,
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


class Side(str, Enum):
    PRO = 'PRO'
    CON = 'CON'


class ConcessionService:
    """
    Thin orchestrator: judge the last assistant→user pair, update running
    in-memory aggregates, and pass hidden <SCORING> system signals to the LLM.
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
        self._scores: Dict[int, RunningScores] = {}

    async def analyze_conversation(
        self,
        messages: List[Message],
        side: Side,
        conversation_id: int,
        topic: str,
    ) -> str:
        side = Side(side.upper())
        mapped = self._map_history(messages)

        last_eval = judge_last_two_messages(
            mapped,
            side=side.value,
            topic=topic,
            nli=self.nli,
            entailment_threshold=self.entailment_threshold,
            contradiction_threshold=self.contradiction_threshold,
        )

        scoring_system_msg: Optional[str] = None

        if last_eval:
            features = features_from_last_eval(
                last_eval,
                side=side.value,
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

            # update in-memory aggregates
            rs = self._scores.setdefault(conversation_id, RunningScores())
            rs.update(
                align=verdict['alignment'],
                ts=last_eval['thesis_scores'],
                ps=last_eval['scores'],
            )

            # build hidden signals
            ctx_sig = build_context_signal(
                {
                    'alignment': verdict['alignment'],
                    'concession': verdict['concession'],
                    'reason': verdict['reason'],
                    'scores': last_eval['scores'],
                    'thesis_scores': last_eval['thesis_scores'],
                    'topic': topic,
                }
            )
            agg_sig = build_score_signal(rs)
            scoring_system_msg = make_scoring_system_message(ctx_sig, agg_sig)

        side_tag = f'<STANCE side="{side}" topic="{topic}"/>'

        # LLM sees signals as an extra system message; MUST NOT show to user
        reply = await self.llm.debate(
            messages=messages,
            scoring_system_msg=scoring_system_msg,
            stance_system_msg=side_tag,
        )
        return reply.strip()

    @staticmethod
    def _map_history(messages: List[Message]) -> List[dict]:
        return [
            {'role': ('assistant' if m.role == 'bot' else 'user'), 'content': m.message}
            for m in messages
        ]
