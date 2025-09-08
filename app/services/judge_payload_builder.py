from typing import List, Optional, Tuple

from app.domain.enums import Stance
from app.domain.nli.judge_payload import NLIJudgePayload
from app.domain.nli.scoring import ScoringConfig
from app.domain.ports.nli import NLIPort
from app.nli.ops import agg_max
from app.utils.text import normalize_spaces, word_count

from .text_mining import (
    extract_claims,
    max_contra_self_vs_sentences,
    on_topic_from_scores,
)


class JudgePayloadBuilder:
    def __init__(self, nli: NLIPort, scoring: ScoringConfig):
        self.nli = nli
        self.scoring = scoring

    def build(
        self, conversation: List[dict], stance: Stance, topic: str, state
    ) -> Tuple[Optional[NLIJudgePayload], str, str]:
        # — find latest user + a prior assistant with min words
        user_idx = self._last_index(conversation, 'user')
        if user_idx is None:
            return None, '', ''
        min_asst_words = getattr(self.scoring, 'min_assistant_words', 8)
        bot_idx = self._last_index(
            conversation,
            'assistant',
            predicate=lambda m: word_count(m.get('content', '')) >= min_asst_words,
            before=user_idx,
        )
        if bot_idx is None or not topic.strip():
            return None, '', ''

        user_txt = conversation[user_idx]['content']
        bot_txt = conversation[bot_idx]['content']
        user_clean = normalize_spaces(user_txt)

        # thesis scores
        self_scores = self.nli.bidirectional_scores(topic, user_clean)
        thesis_agg = agg_max(self_scores)
        on_topic = on_topic_from_scores(self_scores, self.scoring)

        # sentence scan
        max_sent_contra, _ent, _scores = max_contra_self_vs_sentences(
            self.nli, topic, user_txt
        )

        # pairwise best vs assistant claims
        claims = extract_claims(bot_txt)
        if claims:
            best_pair = self._best_pair(claims, user_clean)
        else:
            best_pair = {'entailment': 0.0, 'contradiction': 0.0, 'neutral': 1.0}

        payload = NLIJudgePayload(
            topic=topic.strip(),
            stance=stance,
            language=state.lang,
            turn_index=state.assistant_turns,
            user_text=user_txt,
            bot_text=bot_txt,
            thesis_scores={
                k: float(thesis_agg.get(k, 0.0))
                for k in ('entailment', 'contradiction', 'neutral')
            },
            pair_best={
                k: float(best_pair.get(k, 0.0))
                for k in ('entailment', 'contradiction', 'neutral')
            },
            max_sent_contra=float(max_sent_contra),
            on_topic=bool(on_topic),
            user_wc=int(word_count(user_txt)),
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

    # --- helpers (local/private) ---
    @staticmethod
    def _last_index(convo, role, predicate=None, before=None):
        end = (before if before is not None else len(convo)) - 1
        for i in range(end, -1, -1):
            m = convo[i]
            if m.get('role') != role:
                continue
            if predicate and not predicate(m):
                continue
            return i
        return None

    def _best_pair(self, claims, user_clean):
        best = {'entailment': 0.0, 'contradiction': 0.0, 'neutral': 1.0}
        best_contra = -1.0
        for c in claims:
            sc = self.nli.bidirectional_scores(c, user_clean)
            agg = agg_max(sc)
            contra = float(agg.get('contradiction', 0.0))
            if contra >= best_contra:
                best_contra = contra
                best = agg
        return best
