# app/services/concession_service.py
import logging
import re
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from app.domain.enums import Stance
from app.domain.models import Message
from app.domain.nli.config import NLIConfig
from app.domain.nli.judge_payload import NLIJudgePayload
from app.domain.nli.scoring import ScoringConfig
from app.domain.ports.debate_store import DebateStorePort
from app.domain.ports.llm import LLMPort
from app.domain.ports.nli import NLIPort
from app.domain.verdicts import after_end_message
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
    - Topic gate upstream ensured normalized `topic` and final `stance`.
    - This service computes NLI evidence and asks the LLM Judge for the FINAL decision.
    - The judge returns: accept|ended|reason|assistant_reply|confidence.
    - No scoreboard rendering here; if judge ends, we return its reply (e.g. "<DEBATE_ENDED>").
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

    STANCE_BANNERS = (
        'i will gladly take the pro stance',
        'i will gladly take the con stance',
        'i will defend the proposition as stated',
        'defenderé la proposición tal como está',
        'defenderei a proposição como está',
        'tomaré el lado pro',
        'tomaré el lado con',
        'tomarei o lado pro',
        'tomarei o lado con',
    )

    def __init__(
        self,
        llm: LLMPort,
        nli: NLIPort,
        judge: LLMPort,
        debate_store: DebateStorePort,
        nli_config: Optional[NLIConfig] = None,
        scoring: Optional[ScoringConfig] = None,
        *,
        llm_judge_min_confidence: float = 0.70,
    ) -> None:
        if llm is None:
            raise ValueError('ConcessionService requires an LLMPort')
        if nli is None:
            raise ValueError('ConcessionService requires an NLIPort')
        if debate_store is None:
            raise ValueError('ConcessionService requires a DebateStorePort')

        self.llm = llm
        self.nli = nli
        self.judge = judge
        self.debate_store = debate_store
        self.nli_config = nli_config or NLIConfig()
        self.scoring = scoring or ScoringConfig()
        self.llm_judge_min_confidence = float(llm_judge_min_confidence)

    # ----------------------------- public API -----------------------------

    async def analyze_conversation(
        self,
        messages: List[Message],
        stance: Stance,
        conversation_id: int,
        topic: str,
    ) -> str:
        state = self.debate_store.get(conversation_id)
        if state is None:
            raise RuntimeError(
                f'DebateState missing for conversation_id={conversation_id}'
            )

        topic_clean: str = state.topic

        logger.debug(
            '[analyze] conv_id=%s stance=%s topic=%s | turns=%d msgs=%d',
            conversation_id,
            stance.value,
            trunc(topic_clean),
            state.assistant_turns,
            len(messages),
        )

        if state.match_concluded:
            logger.debug('[analyze] match already concluded → after_end_message')
            return after_end_message(state=state)

        mapped = self._map_history(messages)
        logger.debug('[analyze] mapped_history=%d', len(mapped))

        # ---- Build evidence and call LLM judge ----
        payload, user_txt, bot_txt = self._build_llm_judge_payload(
            conversation=mapped,
            stance=stance,
            topic=topic_clean,
            state=state,
        )

        print(payload)

        if payload is not None:
            payload_dict = (
                payload.to_dict() if hasattr(payload, 'to_dict') else asdict(payload)
            )
            print('ok')
            decision = await self.judge.nli_judge(payload=payload_dict)
            print('rip')
            accept = bool(decision.accept)
            ended = bool(decision.ended)
            reason = decision.reason or 'llm_judge'
            confidence = float(decision.confidence)
            assistant_reply = (decision.assistant_reply or '').strip()

            logger.debug(
                '[llm_judge] accept=%s ended=%s conf=%.2f reason=%s',
                accept,
                ended,
                confidence,
                reason,
            )

            # confidence-gated tally
            if accept and confidence >= self.llm_judge_min_confidence:
                state.positive_judgements += 1
                logger.debug(
                    "[concession] conv_id=%s +1 (total=%s) reason=%s | user='%s' | bot='%s'",
                    conversation_id,
                    state.positive_judgements,
                    reason,
                    trunc(user_txt, 80),
                    trunc(bot_txt, 80),
                )

            if ended:
                state.match_concluded = True
                self.debate_store.save(conversation_id=conversation_id, state=state)
                logger.debug('[analyze] concluded via judge → returning judge reply')
                return assistant_reply or '<DEBATE_ENDED>'

            if assistant_reply:
                state.assistant_turns += 1
                self.debate_store.save(conversation_id=conversation_id, state=state)
                logger.debug(
                    '[analyze] returning judge reply (len=%d) | turns=%d',
                    len(assistant_reply),
                    state.assistant_turns,
                )
                return sanitize_end_markers(assistant_reply).strip()

            logger.debug(
                '[llm_judge] empty assistant_reply → falling back to generator'
            )
        else:
            logger.debug(
                '[llm_judge] skipped: insufficient context to compute evidence'
            )

        # ---- Fallback: generate with the assistant ----
        # NOTE: adapter.debate(...) currently takes only messages
        reply = await self.llm.debate(messages=messages)
        reply = sanitize_end_markers(reply).strip()

        state.assistant_turns += 1
        self.debate_store.save(conversation_id=conversation_id, state=state)
        logger.debug('[analyze] returning generated reply (len=%d)', len(reply))
        return reply

    # ----------------------------- LLM Judge payload builder -----------------------------

    def _build_llm_judge_payload(
        self,
        *,
        conversation: List[dict],
        stance: Stance,
        topic: str,
        state,
    ) -> Tuple[Optional[NLIJudgePayload], str, str]:
        """
        Build the evidence payload expected by the adapter's LLM Judge.
        Returns: (payload_or_None, user_text, bot_text)
        """
        if not conversation:
            logger.debug('[payload] empty conversation')
            return None, '', ''

        user_idx = self._last_index(conversation, role='user')
        if user_idx is None:
            logger.debug('[payload] no user message found')
            return None, '', ''

        min_asst_words = self.scoring.min_assistant_words
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

        thesis = topic.strip()
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

        # Pairwise best
        claims = self._extract_claims(bot_txt)
        if claims:
            claim_scores = self._claim_scores(claims, user_clean)
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
            stance='pro',
            language=state.lang,
            turn_index=state.assistant_turns,
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
            policy={
                'required_positive_judgements': state.policy.required_positive_judgements,
                'max_assistant_turns': state.policy.max_assistant_turns,
            },
            progress={
                'positive_judgements': state.positive_judgements,
                'assistant_turns': state.assistant_turns,
            },
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
        skipped_banners = 0
        for s in parts:
            if s.endswith('?'):
                continue
            s2 = drop_questions(s).strip()
            if not s2:
                continue
            s2_l = s2.lower()

            if any(s2_l.startswith(prefix) for prefix in self.ACK_PREFIXES):
                continue
            if any(b in s2_l for b in self.STANCE_BANNERS):
                skipped_banners += 1
                continue

            if not s2.endswith(('.', '!')):
                s2 += '.'
            if len(s2.split()) >= 3:
                claims.append(s2)

        if skipped_banners:
            logger.debug('[claims] skipped_banners=%d', skipped_banners)
        logger.debug('[claims] extracted=%d', len(claims))
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
