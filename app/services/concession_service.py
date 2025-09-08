# app/services/concession_service.py
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.domain.enums import Stance
from app.domain.models import Message
from app.domain.nli.config import NLIConfig
from app.domain.nli.judge_payload import NLIJudgePayload
from app.domain.nli.scoring import ScoringConfig
from app.domain.ports.debate_store import DebateStorePort
from app.domain.ports.llm import LLMPort
from app.domain.ports.nli import NLIPort
from app.domain.verdicts import after_end_message, build_verdict
from app.nli.ops import agg_max
from app.utils.text import (
    drop_questions,
    normalize_spaces,
    round3,
    sanitize_end_markers,
    trunc,
    word_count,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ConcessionService:
    """
    Assumptions:
      - Upstream topic gate guarantees a normalized debate-ready `topic`.
      - Upstream also sets the final `stance` the bot must take.
      - This service computes NLI evidence and delegates the *final* decision
        to the adapter's LLM-based NLI judge (primary & only judge).
    """

    ACK_PREFIXES = (
        'thanks',
        'thank you',
        'i appreciate',
        'good point',
        'fair point',
        'i see',
        'understand',
    )

    def __init__(
        self,
        llm: LLMPort,
        nli: Optional[NLIPort] = None,
        nli_config: Optional[NLIConfig] = None,
        scoring: Optional[ScoringConfig] = None,
        debate_store: Optional[DebateStorePort] = None,
        *,
        llm_judge_min_confidence: float = 0.70,
    ) -> None:
        self.llm = llm
        self.nli = nli
        self.debate_store = debate_store
        self.nli_config = nli_config or NLIConfig()
        self.scoring = scoring or ScoringConfig()

        # LLM judge confidence gate (applied here if you want to gate effects)
        self.llm_judge_min_confidence = float(llm_judge_min_confidence)

    # ----------------------------- public API -----------------------------

    async def analyze_conversation(
        self,
        messages: List[Message],
        stance: Stance,  # kept for compatibility; state is authoritative
        conversation_id: int,
        topic: str,  # kept for compatibility; state is authoritative
    ) -> str:
        state = self.debate_store.get(conversation_id)
        if state is None:
            raise RuntimeError(
                f'DebateState missing for conversation_id={conversation_id}'
            )

        topic_clean: str = getattr(state, 'topic', topic)

        logger.debug(
            '[analyze] conv_id=%s stance=%s topic=%s | turns=%d msgs=%d',
            conversation_id,
            stance,
            trunc(topic_clean),
            getattr(state, 'assistant_turns', 0),
            len(messages),
        )

        if getattr(state, 'match_concluded', False):
            logger.debug(
                '[analyze] match already concluded → sending after_end_message'
            )
            return after_end_message(state=state)

        mapped = self._map_history(messages)
        logger.debug(
            '[analyze] mapped_history: %d entries (u/a roles preserved)', len(mapped)
        )

        # ---- Compute NLI evidence & call the LLM Judge (primary) ----
        judge_concession = False

        payload, user_txt, bot_txt = self._build_llm_judge_payload(
            conversation=mapped,
            stance=stance,
            topic=topic_clean,
        )
        if payload is not None:
            # Adapter owns the prompt + result dataclass; we accept object or dict.
            decision = await self.llm.nli_judge(payload=payload.to_dict())

            verdict = getattr(decision, 'verdict', None) or (
                isinstance(decision, dict) and decision.get('verdict')
            )
            concession = getattr(decision, 'concession', None)
            if concession is None and isinstance(decision, dict):
                concession = bool(decision.get('concession', False))
            confidence = getattr(decision, 'confidence', None)
            if confidence is None and isinstance(decision, dict):
                try:
                    confidence = float(decision.get('confidence', 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
            reason = getattr(decision, 'reason', None) or (
                isinstance(decision, dict) and decision.get('reason') or ''
            )

            logger.debug(
                '[llm_judge] verdict=%s concession=%s conf=%.2f reason=%s',
                verdict,
                concession,
                confidence or 0.0,
                reason,
            )

            # Optional effect gate by confidence (kept minimal here)
            if bool(concession) and (
                confidence is None or confidence >= self.llm_judge_min_confidence
            ):
                judge_concession = True

            # Record concession counts if any
            if judge_concession:
                state.positive_judgements += 1
                logger.debug(
                    "[concession] conv_id=%s +1 (total=%s) | reason=%s | user='%s' | bot='%s'",
                    conversation_id,
                    state.positive_judgements,
                    (reason or 'llm_judge'),
                    trunc(user_txt, 80),
                    trunc(bot_txt, 80),
                )
        else:
            logger.debug(
                '[llm_judge] skipped: insufficient context to compute evidence'
            )

        # Short-circuit if match should end before generating a reply
        if getattr(state, 'maybe_conclude', lambda: False)():
            state.match_concluded = True
            self.debate_store.save(conversation_id=conversation_id, state=state)
            logger.debug('[analyze] match concluded after judge → build_verdict')
            return build_verdict(state=state)

        # Generate reply
        reply = await self.llm.debate(messages=messages, state=state)
        reply = sanitize_end_markers(reply).strip()

        # Update turn count and (maybe) conclude once
        state.assistant_turns += 1
        logger.debug('[analyze] assistant_turns=%d', state.assistant_turns)
        if getattr(state, 'maybe_conclude', lambda: False)():
            state.match_concluded = True
            self.debate_store.save(conversation_id=conversation_id, state=state)
            logger.debug('[analyze] match concluded after reply → build_verdict')
            return build_verdict(state=state)

        self.debate_store.save(conversation_id=conversation_id, state=state)
        logger.debug('[analyze] returning reply (len=%d)', len(reply))
        return reply

    # ----------------------------- LLM Judge payload builder -----------------------------

    def _build_llm_judge_payload(
        self,
        *,
        conversation: List[dict],
        stance: Stance,
        topic: str,
    ) -> Tuple[Optional[NLIJudgePayload], str, str]:
        """
        Build the evidence payload expected by the adapter's LLM Judge.

        Returns: (payload_or_None, user_text, bot_text)
                 payload_or_None is None if context is insufficient.
        """
        if not conversation:
            logger.debug('[payload] empty conversation')
            return None, '', ''

        user_idx = self._last_index(conversation, role='user')
        if user_idx is None:
            logger.debug('[payload] no user message found')
            return None, '', ''

        min_asst_words = getattr(self.scoring, 'min_assistant_words', 10)
        bot_idx = self._last_index(
            conversation,
            role='assistant',
            predicate=lambda m: word_count(m.get('content', '')) >= min_asst_words,
            before=user_idx,
        )
        if bot_idx is None:
            logger.debug(
                '[payload] no assistant message meeting min words=%d found',
                min_asst_words,
            )
            return None, '', ''

        user_txt = conversation[user_idx]['content']
        bot_txt = conversation[bot_idx]['content']
        user_wc = word_count(user_txt)
        user_clean = normalize_spaces(user_txt)

        # Thesis & scores
        thesis = (topic or '').strip()
        if not thesis:
            logger.debug('[payload] empty thesis')
            return None, '', ''

        # Thesis-level NLI
        self_scores = self.nli.bidirectional_scores(thesis, user_clean)
        thesis_agg = agg_max(self_scores)
        on_topic = self._on_topic_from_scores(self_scores)
        logger.debug(
            '[payload] thesis_agg=%s | on_topic=%s', round3(thesis_agg), on_topic
        )

        # Sentence scan (max contradiction)
        max_sent_contra, _max_ent, _scores_at_max = self._max_contra_self_vs_sentences(
            thesis, user_txt
        )
        logger.debug('[payload] max_sent_contra=%.3f', max_sent_contra)

        # Pairwise best: extract candidate claims from bot and score vs user
        claims = self._extract_claims(bot_txt)
        if claims:
            claim_scores = self._claim_scores(claims, user_clean)
            # choose by highest contradiction signal
            best_claim, _ent, best_contra, best_rel, best_pair_scores = max(
                claim_scores, key=lambda t: t[2]
            )
            pair_agg = agg_max(best_pair_scores)
            logger.debug(
                "[payload] best_claim='%s' | best_rel=%.3f best_contra=%.3f | pair_agg=%s",
                trunc(best_claim, 80),
                best_rel,
                best_contra,
                round3(pair_agg),
            )
        else:
            pair_agg = {'entailment': 0.0, 'contradiction': 0.0, 'neutral': 1.0}
            logger.debug(
                '[payload] no claims found in assistant reply; using neutral pair_agg'
            )

        payload = NLIJudgePayload(
            topic=thesis,
            stance=('pro' if stance == Stance.PRO else 'con'),
            user_text=user_txt,
            bot_text=bot_txt,
            thesis_scores={
                'entailment': float(thesis_agg.get('entailment', 0.0)),
                'contradiction': float(thesis_agg.get('contradiction', 0.0)),
                'neutral': float(thesis_agg.get('neutral', 0.0)),
            },
            pair_best={
                'entailment': float(pair_agg.get('entailment', 0.0)),
                'contradiction': float(pair_agg.get('contradiction', 0.0)),
                'neutral': float(pair_agg.get('neutral', 0.0)),
            },
            max_sent_contra=float(max_sent_contra),
            on_topic=bool(on_topic),
            user_wc=int(user_wc),
        )
        return payload, user_txt, bot_txt

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

    def _extract_claims(self, bot_txt: str) -> List[str]:
        if not bot_txt:
            return []
        parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', bot_txt) if p.strip()]
        claims: List[str] = []
        for s in parts:
            if s.endswith('?'):
                continue
            s2 = drop_questions(s).strip()
            if not s2:
                continue
            s2_l = s2.lower()
            if any(s2_l.startswith(prefix) for prefix in self.ACK_PREFIXES):
                continue
            if not s2.endswith(('.', '!')):
                s2 += '.'
            if len(s2.split()) >= 3:
                claims.append(s2)
        return claims

    def _claim_scores(
        self, claims: List[str], user_clean: str
    ) -> List[Tuple[str, float, float, float, Dict[str, Dict[str, float]]]]:
        out: List[Tuple[str, float, float, float, Dict[str, Dict[str, float]]]] = []
        for c in claims:
            sc = self.nli.bidirectional_scores(c, user_clean)
            agg = agg_max(sc)
            ent = float(agg.get('entailment', 0.0))
            con = float(agg.get('contradiction', 0.0))
            neu = float(agg.get('neutral', 1.0))
            rel = max(ent, con, 1.0 - neu)
            out.append((c, ent, con, rel, sc))
        return out

    def _on_topic_from_scores(self, thesis_scores: Dict[str, Dict[str, float]]) -> bool:
        ph = thesis_scores.get('p_to_h', {}) or {}
        hp = thesis_scores.get('h_to_p', {}) or {}

        def has_signal(d: Dict[str, float]) -> bool:
            ent = float(d.get('entailment', 0.0))
            con = float(d.get('contradiction', 0.0))
            neu = float(d.get('neutral', 1.0))
            return (max(ent, con) >= self.scoring.topic_signal_min) or (
                neu <= self.scoring.topic_neu_max
            )

        on = has_signal(ph) or has_signal(hp)
        logger.debug('[topic] on_topic=%s | agg=%s', on, round3(agg_max(thesis_scores)))
        return on

    def _max_contra_self_vs_sentences(
        self, self_thesis: str, user_txt: str
    ) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
        if not user_txt or not self_thesis:
            logger.debug('[sent_scan] empty inputs')
            return 0.0, 0.0, {}

        parts = [p.strip() for p in re.split(r'(?<=[.!?])\s+', user_txt) if p.strip()]
        sentences: List[str] = []
        for s in parts:
            s2 = drop_questions(s).strip()
            if not s2 or s.endswith('?'):
                continue
            if not s2.endswith(('.', '!', '?')):
                s2 += '.'
            sentences.append(s2)

        max_contra = 0.0
        ent_at_max = 0.0
        scores_at_max: Dict[str, Dict[str, float]] = {}
        for s in sentences:
            sc = self.nli.bidirectional_scores(self_thesis, s)
            agg = agg_max(sc)
            ent = float(agg.get('entailment', 0.0))
            con = float(agg.get('contradiction', 0.0))
            if con >= max_contra:
                max_contra = con
                ent_at_max = ent
                scores_at_max = sc

        logger.debug(
            '[sent_scan] sentences=%d | max_contra=%.3f ent_at_max=%.3f',
            len(sentences),
            max_contra,
            ent_at_max,
        )
        return max_contra, ent_at_max, scores_at_max
