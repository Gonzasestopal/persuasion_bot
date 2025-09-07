# app/services/concession_service.py
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.domain.enums import Stance
from app.domain.models import Message
from app.domain.nli.config import NLIConfig
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
    Judge logic compatible with your unit tests:
    - Strong thesis contradiction (≥ strict_contra_threshold) -> concede (reason 'thesis_opposition_soft'),
      even if the user is short (overrides min words).
    - Pairwise contradiction fallback -> concede if on-topic & user long enough (reason 'pairwise_opposition_soft').
    - Thesis support -> SAME (reason 'thesis_support'), non-conceding.
    - Off-topic blocks pairwise concession.
    - Topic gate is computed from canonical self-thesis vs user.
    - Includes sentence-level scan for thesis contradiction (max across sentences).
    """

    # De-emphasize pure acknowledgments as “claims”
    ACK_PREFIXES = (
        "you're right",
        'you are right',
        'indeed',
        'i agree',
        'good point',
        'correct',
        'fair point',
        'exactly',
    )

    def __init__(
        self,
        llm: LLMPort,
        nli: Optional[NLIPort] = None,
        nli_config: Optional[NLIConfig] = None,
        scoring: Optional[ScoringConfig] = None,
        debate_store: Optional[DebateStorePort] = None,
    ) -> None:
        self.nli_config = nli_config or NLIConfig()
        self.scoring = scoring or ScoringConfig()
        self.nli = nli
        self.llm = llm
        self.debate_store = debate_store

    # ----------------------------- public API -----------------------------

    async def analyze_conversation(
        self, messages: List[Message], stance: Stance, conversation_id: int, topic: str
    ) -> str:
        state = self.debate_store.get(conversation_id)
        if state is None:
            raise RuntimeError(
                f'DebateState missing for conversation_id={conversation_id}'
            )

        logger.debug(
            '[analyze] conv_id=%s stance=%s topic=%s',
            conversation_id,
            stance.value,
            trunc(topic),
        )

        if state.match_concluded:
            return after_end_message(state=state)

        mapped = [
            {'role': ('assistant' if m.role == 'bot' else 'user'), 'content': m.message}
            for m in messages
        ]

        out = self.judge_last_two_messages(
            conversation=mapped, stance=stance, topic=topic
        )

        if out and out.get('concession'):
            state.positive_judgements += 1
            logger.debug(
                "[concession] conv_id=%s +1 concession (total=%s) | reason=%s | user='%s' | bot='%s'",
                conversation_id,
                state.positive_judgements,
                out.get('reason'),
                trunc(out.get('user_text_sample', ''), 80),
                trunc(out.get('bot_text_sample', ''), 80),
            )
            self.debate_store.save(conversation_id=conversation_id, state=state)

        if getattr(state, 'maybe_conclude', lambda: False)():
            state.match_concluded = True
            self.debate_store.save(conversation_id=conversation_id, state=state)
            return build_verdict(state=state)

        reply = await self.llm.debate(messages=messages, state=state)
        reply = sanitize_end_markers(reply)

        state.assistant_turns += 1
        if getattr(state, 'maybe_conclude', lambda: False)():
            state.match_concluded = True
            self.debate_store.save(conversation_id=conversation_id, state=state)
            return build_verdict(state=state)

        self.debate_store.save(conversation_id=conversation_id, state=state)
        return reply.strip()

    # ----------------------------- judge -----------------------------

    def judge_last_two_messages(
        self, conversation: List[dict], stance: Stance, topic: str
    ) -> Optional[Dict[str, Any]]:
        if not conversation:
            return None

        # last user
        user_idx = next(
            (
                i
                for i in range(len(conversation) - 1, -1, -1)
                if conversation[i].get('role') == 'user'
            ),
            None,
        )
        if user_idx is None:
            return None

        # most recent substantive assistant before that (>=10 words)
        bot_idx = next(
            (
                i
                for i in range(user_idx - 1, -1, -1)
                if conversation[i].get('role') == 'assistant'
                and word_count(conversation[i].get('content', '')) >= 10
            ),
            None,
        )
        if bot_idx is None:
            return None

        user_txt = conversation[user_idx]['content']
        bot_txt = conversation[bot_idx]['content']
        user_wc = word_count(user_txt)
        user_clean = normalize_spaces(user_txt)

        # Extract claims from assistant reply (used for pairwise)
        claims = self._extract_claims(bot_txt)

        # Canonical thesis sentences for stance (self/opp)
        clean_topic = self._clean_topic_for_nli(topic)
        canon_self = self._canonical_stance(clean_topic, stance)
        canon_opp = self._canonical_stance(
            clean_topic, Stance.CON if stance == Stance.PRO else Stance.PRO
        )

        # --- 1) Pairwise (claims vs user), pick strongest contradiction
        if claims:
            claim_scores = self._claim_scores(claims, user_clean)
            best_by_contra = max(claim_scores, key=lambda t: t[2])
            best_claim, best_entail, best_contra, best_related, best_pair_scores = (
                best_by_contra
            )
        else:
            best_claim = ''
            best_entail = 0.0
            best_contra = 0.0
            best_related = 0.0
            best_pair_scores = {'p_to_h': {}, 'h_to_p': {}}

        relatedness_min = getattr(self.scoring, 'relatedness_min', 0.35)
        engaged = best_related >= relatedness_min
        logger.debug(
            "[claims] n=%d | best_contra=%.3f '%s'",
            len(claims),
            best_contra,
            trunc(best_claim, 60),
        )

        # --- 2) Thesis-level scores (canonical self vs user)
        self_scores = self.nli.bidirectional_scores(canon_self, user_clean)
        self_agg = agg_max(self_scores)

        # Topic gate: from thesis scores
        on_topic = self._on_topic_from_scores(self_scores)

        # Sentence-level scan: max contradiction across user sentences
        max_sent_contra, max_sent_ent, max_sent_scores = (
            self._max_contra_self_vs_sentences(canon_self, user_txt)
        )

        # --- 3) Opposite stance support checks
        opp_scores = self.nli.bidirectional_scores(canon_opp, user_clean)
        opp_agg = agg_max(opp_scores)

        support_min = getattr(self.scoring, 'support_min', 0.50)
        contra_min = getattr(self.scoring, 'strict_contra_threshold', 0.55)
        pair_soft = getattr(self.scoring, 'contradiction_threshold', 0.55)
        margin_min = getattr(self.scoring, 'margin_min', 0.15)
        min_user_words = getattr(self.scoring, 'min_user_words', 8)

        self_ent = float(self_agg.get('entailment', 0.0))
        self_con = float(self_agg.get('contradiction', 0.0))
        opp_ent = float(opp_agg.get('entailment', 0.0))
        opp_con = float(opp_agg.get('contradiction', 0.0))

        # User supports opposite if: entail opposite OR contradict self (with margin)
        opp_supported = (
            opp_ent >= support_min and (opp_ent - opp_con) >= margin_min
        ) or (self_con >= contra_min and (self_con - self_ent) >= margin_min)
        # User supports our stance?
        self_supported = (self_ent >= support_min) and (
            (self_ent - self_con) >= margin_min
        )

        logger.debug(
            '[topic] on_topic=%s | agg=%s',
            on_topic,
            round3(self_agg),
        )
        logger.debug(
            "[rel] best_claim_relatedness=%.3f (min=%.3f) | best_claim='%s'",
            best_related,
            relatedness_min,
            trunc(best_claim, 60),
        )

        # ------------------- Decision branches (ordered) -------------------

        # A) Strong thesis contradiction → concede (overrides min words)
        thesis_strong_contra = max(self_con, max_sent_contra) >= contra_min
        if thesis_strong_contra and on_topic:
            return self._mk_result(
                stance=stance,
                alignment='OPPOSITE',
                concession=True,
                reason='thesis_opposition_soft',
                pair_scores=max_sent_scores or self_scores,
                thesis_scores=self_scores,
                user_txt=user_txt,
                bot_txt=bot_txt,
                topic=clean_topic,
            )

        # B) SAME if thesis supported (non-conceding)
        if on_topic and self_supported:
            return self._mk_result(
                stance=stance,
                alignment='SAME',
                concession=False,
                reason='thesis_support',
                pair_scores=self_scores,
                thesis_scores=self_scores,
                user_txt=user_txt,
                bot_txt=bot_txt,
                topic=clean_topic,
            )

        # C) Too short blocks other concessions (except A)
        if user_wc < min_user_words:
            return self._mk_result(
                stance=stance,
                alignment='UNKNOWN',
                concession=False,
                reason='too_short',
                pair_scores=best_pair_scores or self_scores,
                thesis_scores=self_scores,
                user_txt=user_txt,
                bot_txt=bot_txt,
                topic=clean_topic,
            )

        # D) Off-topic blocks pairwise fallback
        if not on_topic:
            return self._mk_result(
                stance=stance,
                alignment='UNKNOWN',
                concession=False,
                reason='off_topic',
                pair_scores=best_pair_scores or self_scores,
                thesis_scores=self_scores,
                user_txt=user_txt,
                bot_txt=bot_txt,
                topic=clean_topic,
            )

        # E) Pairwise contradiction fallback (soft) → concede
        if engaged and best_contra >= pair_soft:
            return self._mk_result(
                stance=stance,
                alignment='OPPOSITE',
                concession=True,
                reason='pairwise_opposition_soft',
                pair_scores=best_pair_scores,
                thesis_scores=self_scores,
                user_txt=user_txt,
                bot_txt=bot_txt,
                topic=clean_topic,
            )

        # F) Otherwise unknown / underdetermined
        return self._mk_result(
            stance=stance,
            alignment='UNKNOWN',
            concession=False,
            reason='underdetermined',
            pair_scores=best_pair_scores or self_scores,
            thesis_scores=self_scores,
            user_txt=user_txt,
            bot_txt=bot_txt,
            topic=clean_topic,
        )

    # ----------------------------- helpers -----------------------------

    ACK_PREFIXES = (
        "you're right",
        'you are right',
        'indeed',
        'i agree',
        'good point',
        'correct',
        'fair point',
        'exactly',
    )

    @staticmethod
    def _clean_topic_for_nli(topic: str) -> str:
        """
        Remove accidental meta like 'Language: EN', 'Side: PRO', 'Topic: ...'
        Keep only the proposition's first sentence.
        """
        s = topic
        s = re.sub(r'\b(Language|Idioma)\s*:\s*[A-Za-z]{2}\b\.?', '', s, flags=re.I)
        s = re.sub(r'\b(Side|Lado)\s*:\s*(PRO|CON)\b\.?', '', s, flags=re.I)
        s = re.sub(r'\b(Topic|Tema)\s*:\s*', '', s, flags=re.I)
        s = s.strip().strip('.')
        s = s.split('.')[0].strip()
        return s

    @staticmethod
    def _topic_statement(clean_topic: str) -> str:
        return clean_topic.rstrip('.') + '.'

    def _extract_claims(self, bot_txt: str) -> List[str]:
        """
        Extract assertive sentences from the assistant's last reply.
        Drop questions and pure acknowledgments.
        """
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

    @staticmethod
    def _clean_topic_for_nli(topic: str) -> str:
        s = topic or ''
        s = re.sub(r'\b(Language|Idioma)\s*:\s*[A-Za-z]{2}\b\.?', '', s, flags=re.I)
        s = re.sub(r'\b(Side|Lado)\s*:\s*(PRO|CON)\b\.?', '', s, flags=re.I)
        s = re.sub(r'\b(Topic|Tema)\s*:\s*', '', s, flags=re.I)
        s = re.sub(r'^\s*i\s+(think|believe)\s+(that\s+)?', '', s, flags=re.I)
        s = re.sub(r'^\s*in\s+my\s+opinion\s*,?\s*', '', s, flags=re.I)
        s = s.strip().strip('.')
        s = s.split('.')[0].strip()
        # unwrap "It is (not the case|false) that X"
        m = re.match(
            r'^it\s+is\s+(?:not\s+the\s+case|false)\s+that\s+(.+)$', s, flags=re.I
        )
        if m:
            s = m.group(1).strip()
        return s

    @staticmethod
    def _polarity_variants(t: str) -> Tuple[str, str]:
        t0 = (t or '').strip().rstrip('.')
        tl = t0.lower()

        # exists ↔ does not exist
        m = re.match(r'^(.+?)\s+do(?:es)?\s+not\s+exist$', tl, flags=re.I)
        if m:
            subj = t0[: t0.lower().rfind(' does not exist')].strip()
            return f'{subj} exists.', f'{subj} does not exist.'
        m = re.match(r"^(.+?)\s+doesn'?t\s+exist$", tl, flags=re.I)
        if m:
            subj = t0[: t0.lower().rfind(" doesn't exist")].strip()
            return f'{subj} exists.', f'{subj} does not exist.'
        m = re.match(r'^(.+?)\s+exists$', tl, flags=re.I)
        if m:
            subj = t0[: t0.lower().rfind(' exists')].strip()
            return f'{subj} exists.', f'{subj} does not exist.'

        # are not / are
        m = re.match(r'^(?P<subj>.+?)\s+are\s+not\s+(?P<pred>.+)$', t0, flags=re.I)
        if m:
            subj, pred = m.group('subj').strip(), m.group('pred').strip()
            return f'{subj} are {pred}.', f'{subj} are not {pred}.'
        m = re.match(r'^(?P<subj>.+?)\s+are\s+(?P<pred>.+)$', t0, flags=re.I)
        if m:
            subj, pred = m.group('subj').strip(), m.group('pred').strip()
            return f'{subj} are {pred}.', f'{subj} are not {pred}.'

        # is not / is
        m = re.match(r'^(?P<subj>.+?)\s+is\s+not\s+(?P<pred>.+)$', t0, flags=re.I)
        if m:
            subj, pred = m.group('subj').strip(), m.group('pred').strip()
            return f'{subj} is {pred}.', f'{subj} is not {pred}.'
        m = re.match(r'^(?P<subj>.+?)\s+is\s+(?P<pred>.+)$', t0, flags=re.I)
        if m:
            subj, pred = m.group('subj').strip(), m.group('pred').strip()
            return f'{subj} is {pred}.', f'{subj} is not {pred}.'

        # fallback
        pos = t0 + '.'
        neg = f'It is not the case that {t0}.'
        return pos, neg

    @staticmethod
    def _canonical_stance(topic_clean: str, bot_stance: Stance) -> str:
        pos, neg = ConcessionService._polarity_variants(topic_clean)
        return pos if bot_stance == Stance.PRO else neg

    # ----------------------------- pack result -----------------------------

    @staticmethod
    def _mk_result(
        stance: Stance,
        alignment: str,
        concession: bool,
        reason: str,
        pair_scores: Dict[str, Dict[str, float]],
        thesis_scores: Dict[str, Dict[str, float]],
        user_txt: str,
        bot_txt: str,
        topic: str,
    ) -> Dict[str, Any]:
        return {
            'passed_stance': stance.value,
            'alignment': alignment,
            'concession': concession,
            'reason': reason,
            'reasons': [reason],
            'scores': agg_max(pair_scores),
            'thesis_scores': agg_max(thesis_scores),
            'user_text_sample': user_txt,
            'bot_text_sample': bot_txt,
            'topic': topic,
        }

    def _max_contra_self_vs_sentences(
        self, canon_self: str, user_txt: str
    ) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
        """
        Scan user sentences against the canonical self-thesis and return:
          (max_contradiction, entailment_at_that_sentence, scores_for_that_sentence)
        If no sentences are found, returns (0.0, 0.0, {}).
        """
        if not user_txt or not canon_self:
            return 0.0, 0.0, {}

        # Split into sentences; drop pure questions
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
            sc = self.nli.bidirectional_scores(canon_self, s)
            agg = agg_max(sc)
            ent = float(agg.get('entailment', 0.0))
            con = float(agg.get('contradiction', 0.0))
            if con >= max_contra:
                max_contra = con
                ent_at_max = ent
                scores_at_max = sc

        return max_contra, ent_at_max, scores_at_max
