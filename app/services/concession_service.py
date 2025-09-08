# app/services/concession_service.py
import logging
from dataclasses import asdict
from typing import List, Optional

from app.domain.concession_policy import DebateState
from app.domain.enums import Stance
from app.domain.models import Message
from app.domain.nli.config import NLIConfig
from app.domain.nli.scoring import ScoringConfig
from app.domain.ports.llm import LLMPort
from app.domain.ports.nli import NLIPort
from app.services.end_renderer import EndRenderer
from app.services.judge_payload_builder import JudgePayloadBuilder
from app.services.novelty_guard import NoveltyGuard
from app.utils.text import (
    sanitize_end_markers,
    trunc,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ConcessionService:
    """
    Build NLI evidence payload from conversation.
    """

    # Exclude soft-ack sentences from claim extraction

    def __init__(
        self,
        llm: LLMPort,
        nli: NLIPort,
        judge: LLMPort,
        nli_config: Optional[NLIConfig] = None,
        scoring: Optional[ScoringConfig] = None,
        *,
        llm_judge_min_confidence: float = 0.70,
    ) -> None:
        self.llm = llm
        self.nli = nli
        self.judge = judge
        self.nli_config = nli_config or NLIConfig()
        self.scoring = scoring or ScoringConfig()
        self.llm_judge_min_confidence = float(llm_judge_min_confidence)
        self.payloads = JudgePayloadBuilder(self.nli, self.scoring)
        self.novelty = NoveltyGuard(self.scoring)
        self.ender = EndRenderer(self.llm)

    # ----------------------------- public API -----------------------------
    async def analyze_conversation(
        self,
        messages: List[Message],
        stance: Stance,
        conversation_id: int,
        topic: str,
        state: DebateState,
    ) -> str:
        # sync assistant turns from transcript to avoid drift
        turns_in_transcript = self._count_assistant_turns(messages)
        if state.assistant_turns != turns_in_transcript:
            logger.debug(
                '[sync] correcting assistant_turns: state=%d -> transcript=%d',
                state.assistant_turns,
                turns_in_transcript,
            )
            state.assistant_turns = turns_in_transcript

        topic_clean: str = state.topic
        logger.debug(
            '[analyze] conv_id=%s stance=%s topic=%s | turns=%d msgs=%d',
            conversation_id,
            state.stance,
            trunc(topic_clean),
            state.assistant_turns,
            len(messages),
        )

        # If already ended, render via end-only path
        if state.match_concluded:
            logger.debug('[analyze] already ENDED → debate_aware_end render')
            reply_ended = await self._render_end_via_llm(messages, state)
            return sanitize_end_markers(reply_ended).strip()

        mapped = self._map_history(messages)
        logger.debug('[analyze] mapped_history=%d', len(mapped))

        # judge payload
        payload, user_txt, bot_txt = self.payloads.build(
            mapped, state.stance, state.topic, state
        )

        if payload is not None:
            try:
                payload_dict = (
                    payload.to_dict()
                    if hasattr(payload, 'to_dict')
                    else asdict(payload)
                )
                decision = await self.judge.nli_judge(payload=payload_dict)

                # store judge outputs
                state.set_judge(
                    accept=decision.accept,
                    reason=decision.reason,
                    confidence=decision.confidence,
                )

                # ----------------------------- NOVELTY GUARD APPLIED HERE -----------------------------
                if decision.accept:
                    # confidence gate
                    if decision.confidence < self.llm_judge_min_confidence:
                        state.set_judge(
                            accept=False,
                            reason='low_confidence',
                            confidence=decision.confidence,
                        )
                    else:
                        # novelty gate
                        latest_user_idx = self._last_index(mapped, 'user')
                        novelty = (
                            self.novelty.compute(mapped, latest_user_idx)
                            if latest_user_idx is not None
                            else 1.0
                        )
                        if novelty < float(getattr(self.scoring, 'novelty_min', 0.25)):
                            state.set_judge(
                                accept=False,
                                reason='novelty_reject_duplicate',
                                confidence=decision.confidence,
                            )
                        else:
                            state.positive_judgements += 1
            except Exception as e:
                logger.warning('[judge] scoring failed: %s', e)
        else:
            logger.debug('[judge] skipped: insufficient context for payload')

        # end policy (projected on THIS reply)
        will_hit_cap_now = (
            state.assistant_turns + 1
        ) >= state.policy.max_assistant_turns
        meets_positive_threshold = (
            state.positive_judgements >= state.policy.required_positive_judgements
        )

        if meets_positive_threshold or will_hit_cap_now:
            state.match_concluded = True
            if not state.end_reason:
                state.end_reason = state.last_judge_reason_label or (
                    'policy_threshold_reached'
                    if meets_positive_threshold
                    else 'max_turns_reached'
                )
            logger.debug(
                '[end] ENDED this reply | reason=%s | positives=%d/%d | turns(projected)=%d/%d',
                state.end_reason,
                state.policy.required_positive_judgements,
                state.policy.required_positive_judgements,
                state.assistant_turns + 1,
                state.policy.max_assistant_turns,
            )

        # render
        if state.match_concluded:
            reply = await self.ender.render(messages, state)
        else:
            reply = await self.llm.debate_aware(messages=messages, state=state)

        reply = sanitize_end_markers(reply).strip()

        if not state.match_concluded:
            state.assistant_turns = turns_in_transcript + 1

        logger.debug(
            '[analyze] returning reply (len=%d) | ended=%s',
            len(reply),
            state.match_concluded,
        )
        return reply, state

    # ----------------------------- helpers -----------------------------
    @staticmethod
    def _map_history(messages: List[Message]) -> List[dict]:
        return [
            {'role': ('assistant' if m.role == 'bot' else 'user'), 'content': m.message}
            for m in messages
        ]

    @staticmethod
    def _last_index(
        convo: List[dict],
        role: str,
        predicate=None,
        before: Optional[int] = None,
    ) -> Optional[int]:
        end = (before if before is not None else len(convo)) - 1
        for i in range(end, -1, -1):
            if convo[i].get('role') != role:
                continue
            if predicate and not predicate(convo[i]):
                continue
            return i
        return None

    @staticmethod
    def _count_assistant_turns(messages: List[Message]) -> int:
        return sum(1 for m in messages if m.role == 'bot')
