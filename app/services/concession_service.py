# app/services/concession_service.py
import logging
import re
import string
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

from app.domain.enums import Stance
from app.domain.mappings import END_REASON_MAP
from app.domain.models import Message
from app.domain.nli.config import NLIConfig
from app.domain.nli.judge_payload import NLIJudgePayload
from app.domain.nli.scoring import ScoringConfig
from app.domain.ports.debate_store import DebateStorePort
from app.domain.ports.llm import LLMPort
from app.domain.ports.nli import NLIPort
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

# ----------------------------- Spanish question helpers -----------------------------
SPANISH_Q_WORDS = (
    '¿',
    'como',
    'cómo',
    'que',
    'qué',
    'por que',
    'por qué',
    'cuando',
    'cuándo',
    'donde',
    'dónde',
    'cual',
    'cuál',
    'cuales',
    'cuáles',
    'quien',
    'quién',
    'quienes',
    'quiénes',
)


def _looks_like_question(s: str) -> bool:
    s2 = (s or '').strip().lower()
    if not s2:
        return False
    return s2.startswith('¿') or any(s2.startswith(w + ' ') for w in SPANISH_Q_WORDS)


def _ends_with_strong_punct(s: str) -> bool:
    return s.endswith(('.', '!', '?', '…'))


def _strip_trailing_fragment(parts: List[str]) -> List[str]:
    # if the original text doesn't end with strong punctuation, drop last fragment (often a cut question)
    return parts[:-1] if parts and not _ends_with_strong_punct(parts[-1]) else parts


class ConcessionService:
    """
    Flow:
      1) Build NLI evidence payload from conversation.
      2) Call judge.nli_judge(...) → {accept, confidence, reason, ...}.
      3) Server tallies accepts and decides if debate ends (projected on THIS reply).
      4) Ask the LLM to write the visible reply. If ENDED, call the end-only renderer
         (debate_aware_end) so the LLM itself outputs the short, no-question finale.
    """

    # Exclude soft-ack sentences from claim extraction
    ACK_PREFIXES = (
        'thanks',
        'thank you',
        'i appreciate',
        'good point',
        'fair point',
        'i see',
        'understand',
    )

    # Exclude stance/meta banners from claim extraction
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
        payload, user_txt, bot_txt = self._build_llm_judge_payload(
            conversation=mapped,
            stance=state.stance,
            topic=topic_clean,
            state=state,
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
                    novelty_min = float(getattr(self.scoring, 'novelty_min', 0.25))
                    try:
                        user_idx = self._last_index(mapped, role='user')
                        novelty = (
                            self._latest_user_novelty(mapped, user_idx)
                            if user_idx is not None
                            else 1.0
                        )
                    except Exception:
                        novelty = 1.0  # fail-open on novelty calc

                    if decision.confidence < self.llm_judge_min_confidence:
                        logger.debug(
                            '[judge] ACCEPT dropped by confidence gate (%.2f < %.2f) | reason=%s',
                            decision.confidence,
                            self.llm_judge_min_confidence,
                            decision.reason,
                        )
                    elif novelty < novelty_min:
                        # Do not count as a positive judgment if the user repeated themselves.
                        state.set_judge(
                            accept=False,
                            reason='novelty_reject_duplicate',
                            confidence=decision.confidence,
                        )
                        logger.debug(
                            '[novelty] REJECT: novelty=%.3f < %.3f (duplicate/near-duplicate user turn) | user="%s"',
                            novelty,
                            novelty_min,
                            trunc(user_txt, 80),
                        )
                    else:
                        state.positive_judgements += 1
                        logger.debug(
                            "[judge] +ACCEPT (total=%s) reason=%s conf=%.2f novelty=%.3f | user='%s' | bot='%s'",
                            state.positive_judgements,
                            decision.reason,
                            decision.confidence,
                            novelty,
                            trunc(user_txt, 80),
                            trunc(bot_txt, 80),
                        )
                else:
                    logger.debug(
                        '[judge] REJECT reason=%s conf=%.2f',
                        decision.reason,
                        decision.confidence,
                    )
                # -------------------------------------------------------------------------------
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
                state.positive_judgements,
                state.policy.required_positive_judgements,
                state.assistant_turns + 1,
                state.policy.max_assistant_turns,
            )
            self.debate_store.save(conversation_id=conversation_id, state=state)

        # render
        if state.match_concluded:
            reply = await self._render_end_via_llm(messages, state)
        else:
            reply = await self.llm.debate_aware(messages=messages, state=state)

        reply = sanitize_end_markers(reply).strip()

        if not state.match_concluded:
            state.assistant_turns = turns_in_transcript + 1

        self.debate_store.save(conversation_id=conversation_id, state=state)
        logger.debug(
            '[analyze] returning reply (len=%d) | ended=%s',
            len(reply),
            state.match_concluded,
        )
        return reply

    # ----------------------------- LLM end rendering -----------------------------
    async def _render_end_via_llm(self, messages: List[Message], state) -> str:
        """
        Ask the model to render the end message itself (no server-side override).
        Falls back to debate_aware if debate_aware_end isn't implemented.
        """
        prompt_vars = self._end_prompt_vars(state)
        try:
            logger.debug('[render_end] using debate_aware_end | vars=%s', prompt_vars)
            return await self.llm.debate_aware_end(
                messages=messages,
                prompt_vars=prompt_vars,
                temperature=0.2,
                max_tokens=80,
                stop=None,
            )
        except AttributeError:
            logger.debug(
                '[render_end] debate_aware_end unavailable; falling back to debate_aware'
            )
            return await self.llm.debate_aware(
                messages=messages,
                state=state,
                temperature=0.2,
                max_tokens=80,
                stop=None,
            )

    @staticmethod
    def _end_prompt_vars(state) -> Dict[str, str]:
        """
        Build variables for END_SYSTEM_PROMPT using a single-language map
        (no locale branching). Falls back gracefully.
        """
        reason_label = state.last_judge_reason_label or 'unspecified_reason'
        end_reason = (
            END_REASON_MAP.get(reason_label)
            or state.end_reason
            or reason_label.replace('_', ' ')
        )

        return {
            'LANGUAGE': (state.lang or 'en').lower(),
            'TOPIC': state.topic,
            'DEBATE_STATUS': 'ENDED',
            'END_REASON': end_reason,
            'JUDGE_REASON_LABEL': reason_label,
            'JUDGE_CONFIDENCE': f'{state.last_judge_confidence:.2f}',
        }

    # ----------------------------- LLM Judge payload builder -----------------------------
    def _build_llm_judge_payload(
        self,
        *,
        conversation: List[dict],
        stance: Stance,
        topic: str,
        state,
    ) -> Tuple[Optional[NLIJudgePayload], str, str]:
        if not conversation:
            logger.debug('[payload] empty conversation')
            return None, '', ''

        user_idx = self._last_index(conversation, role='user')
        if user_idx is None:
            logger.debug('[payload] no user message found')
            return None, '', ''

        # min assistant words to pick a bot turn
        min_asst_words = getattr(self.scoring, 'min_assistant_words', 8)
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

        # Thesis NLI
        self_scores = self.nli.bidirectional_scores(thesis, user_clean)
        thesis_agg = agg_max(self_scores)
        on_topic = self._on_topic_from_scores(self_scores)
        logger.debug(
            '[payload] thesis_agg=%s | on_topic=%s', round3(thesis_agg), on_topic
        )

        # Sentence scan
        max_sent_contra, _ent, _scores_at_max = self._max_contra_self_vs_sentences(
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
            logger.debug('[payload] no claims found; using neutral pair_agg')

        payload = NLIJudgePayload(
            topic=thesis,
            stance=stance,
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

    @staticmethod
    def _count_assistant_turns(messages: List[Message]) -> int:
        return sum(1 for m in messages if m.role == 'bot')

    def _extract_claims(self, bot_txt: str) -> List[str]:
        """
        Extracts declarative, substantive claims from the assistant's prior turn,
        excluding questions (incl. Spanish openers) and meta banners.
        Also guards against truncated final fragments.
        """
        if not bot_txt:
            return []
        raw_parts = [
            p.strip() for p in re.split(r'(?<=[.!?…])\s+', bot_txt) if p.strip()
        ]
        parts = _strip_trailing_fragment(raw_parts)

        claims: List[str] = []
        skipped_banners = 0
        for s in parts:
            if s.endswith('?') or _looks_like_question(s):
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

    @staticmethod
    def _tokenize_norm(s: str) -> List[str]:
        # Alphanumeric tokens, lowercased, punctuation stripped
        if not s:
            return []
        s = s.lower()
        s = s.translate(str.maketrans('', '', string.punctuation + '¿¡“”"…—–-'))
        tokens = re.findall(r'\b\w+\b', s)
        return [t for t in tokens if len(t) > 2 and t not in STOP_ALL]

    @staticmethod
    def _char_ngrams(s: str, n: int = 3) -> set:
        if not s:
            return set()
        s2 = s.lower()
        s2 = re.sub(r'\s+', ' ', s2).strip()
        s2 = s2.translate(
            str.maketrans('', '', ' \t\n\r' + string.punctuation + '¿¡“”"…—–-')
        )
        if len(s2) < n:
            return {s2} if s2 else set()
        return {s2[i : i + n] for i in range(len(s2) - n + 1)}

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union if union else 0.0

    def _novelty_score(self, current: str, previous_texts: List[str]) -> float:
        """
        Returns novelty in [0,1]. 1.0 = completely new, 0.0 = duplicated.
        Uses a max-similarity penalty against any prior user message.
        """
        if not current:
            return 0.0
        if not previous_texts:
            return 1.0

        cur_ngrams = self._char_ngrams(current, n=3)
        cur_tokens = set(self._tokenize_norm(current))

        max_sim = 0.0
        for prev in previous_texts:
            prev_ngrams = self._char_ngrams(prev, n=3)
            prev_tokens = set(self._tokenize_norm(prev))

            jacc = self._jaccard(cur_ngrams, prev_ngrams)
            tok_sim = (
                len(cur_tokens & prev_tokens) / max(len(cur_tokens), len(prev_tokens))
                if (cur_tokens or prev_tokens)
                else 1.0
            )
            sim = 0.65 * jacc + 0.35 * tok_sim
            if sim > max_sim:
                max_sim = sim

        novelty = 1.0 - max_sim
        return max(0.0, min(1.0, novelty))

    def _latest_user_novelty(self, convo: List[dict], latest_user_idx: int) -> float:
        """
        Compute novelty of the latest user message vs all *prior* user messages.
        """
        latest_txt = convo[latest_user_idx].get('content', '')
        prev_users = [
            m.get('content', '')
            for i, m in enumerate(convo[:latest_user_idx])
            if m.get('role') == 'user' and m.get('content')
        ]
        return self._novelty_score(latest_txt, prev_users)
