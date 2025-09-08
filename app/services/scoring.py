# app/services/scoring.py
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.adapters.nli.hf_nli import HFNLIProvider
from app.domain.ports.scoring import ScoreFeatures, ScoreVerdict

logger = logging.getLogger(__name__)

SENT_SPLIT_RX = re.compile(r'(?<=[.!?])\s+')
IS_QUESTION_RX = re.compile(r'\?\s*$')


# ----- Running aggregates (purely in-memory) -----
@dataclass
class RunningScores:
    turns: int = 0
    opp: int = 0
    same: int = 0
    unk: int = 0
    tE_ema: float = 0.0
    tC_ema: float = 0.0
    pE_ema: float = 0.0
    pC_ema: float = 0.0

    def update(
        self,
        *,
        align: str,
        ts: Dict[str, float],
        ps: Dict[str, float],
        alpha: float = 0.3,
    ) -> None:
        self.turns += 1
        if align == 'OPPOSITE':
            self.opp += 1
        elif align == 'SAME':
            self.same += 1
        else:
            self.unk += 1

        tE = float(ts.get('entailment', 0.0))
        tC = float(ts.get('contradiction', 0.0))
        pE = float(ps.get('entailment', 0.0))
        pC = float(ps.get('contradiction', 0.0))

        self.tE_ema = alpha * tE + (1 - alpha) * self.tE_ema
        self.tC_ema = alpha * tC + (1 - alpha) * self.tC_ema
        self.pE_ema = alpha * pE + (1 - alpha) * self.pE_ema
        self.pC_ema = alpha * pC + (1 - alpha) * self.pC_ema


# ----- Helpers that used to live in ConcessionService -----
def drop_questions(text: str) -> str:
    sents = [s.strip() for s in re.split(SENT_SPLIT_RX, text) if s.strip()]
    sents = [s for s in sents if not re.search(IS_QUESTION_RX, s)]
    return ' '.join(sents) if sents else text


def bot_thesis(topic: str, bot_stance: str) -> str:
    t = topic.strip().rstrip('.')
    if bot_stance.upper() == 'PRO':
        return f'{t}.'
    return f'It is not true that {t}.'


def nli_confident(
    scores: Dict[str, float], *, pmin: float = 0.75, margin: float = 0.15
) -> bool:
    vals = sorted((float(v) for v in scores.values()), reverse=True)
    if not vals:
        return False
    if len(vals) == 1:
        return vals[0] >= pmin
    return vals[0] >= pmin and (vals[0] - vals[1]) >= margin


def alignment_and_scores_topic_aware(
    nli: HFNLIProvider,
    bot_text: str,
    user_text: str,
    bot_stance: str,
    topic: str,
    *,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """
    Returns:
      align: 'OPPOSITE' | 'SAME' | 'UNKNOWN'
      pair_scores: best of {user->bot, bot->user}
      thesis_scores: NLI(user_text, thesis(topic, stance))
    """
    bot_clean = drop_questions(bot_text)

    s_u2b = nli.score(user_text, bot_clean)
    s_b2u = nli.score(bot_clean, user_text)

    pair_scores = (
        s_u2b
        if max(s_u2b['entailment'], s_u2b['contradiction'])
        >= max(s_b2u['entailment'], s_b2u['contradiction'])
        else s_b2u
    )

    th = bot_thesis(topic, bot_stance)
    thesis_scores = nli.score(user_text, th)

    ent = thesis_scores['entailment']
    contr = thesis_scores['contradiction']

    if contr >= contradiction_threshold and contr > ent:
        align = 'OPPOSITE'
    elif ent >= entailment_threshold and ent > contr:
        align = 'SAME'
    else:
        align = 'UNKNOWN'

    return align, pair_scores, thesis_scores


def latest_idx(
    conv: List[dict], role: str, *, before_idx: Optional[int] = None
) -> Optional[int]:
    for i in range(len(conv) - 1, -1, -1):
        if before_idx is not None and i >= before_idx:
            continue
        if conv[i].get('role') == role:
            return i
    return None


def latest_valid_assistant_before(
    conv: List[dict], before_idx: int, *, min_words: int = 10
) -> Optional[int]:
    for i in range(before_idx - 1, -1, -1):
        m = conv[i]
        if m.get('role') != 'assistant':
            continue
        words = [w for w in m.get('content', '').split() if w.isalpha()]
        if len(words) >= min_words:
            return i
    return None


def judge_last_two_messages(
    conversation: List[dict],
    *,
    stance: str,
    topic: str,
    nli: HFNLIProvider,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> Optional[Dict[str, any]]:
    if not conversation:
        return None
    user_idx = latest_idx(conversation, 'user')
    if user_idx is None:
        return None
    bot_idx = latest_valid_assistant_before(conversation, user_idx)
    if bot_idx is None:
        return None

    user_txt = conversation[user_idx]['content']
    bot_txt = conversation[bot_idx]['content']

    align, pair_scores, thesis_scores = alignment_and_scores_topic_aware(
        nli,
        bot_txt,
        user_txt,
        stance,
        topic,
        entailment_threshold=entailment_threshold,
        contradiction_threshold=contradiction_threshold,
    )
    return {
        'passed_stance': stance,
        'alignment': align,
        'scores': pair_scores,
        'thesis_scores': thesis_scores,
        'user_text_sample': user_txt,
        'bot_text_sample': bot_txt,
        'topic': topic,
    }


def features_from_last_eval(
    ev: Dict[str, any],
    *,
    stance: str,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> ScoreFeatures:
    ts = ev['thesis_scores']
    ps = ev['scores']
    user_len = len(ev.get('user_text_sample', '') or '')
    thesis_ok = nli_confident(ts)
    pair_ok = nli_confident(ps) and user_len >= 30
    return {
        'entailment_threshold': entailment_threshold,
        'contradiction_threshold': contradiction_threshold,
        'pmin': 0.75,
        'margin': 0.15,
        'min_user_len': 30,
        'thesis_entailment': float(ts['entailment']),
        'thesis_contradiction': float(ts['contradiction']),
        'pair_entailment': float(ps['entailment']),
        'pair_contradiction': float(ps['contradiction']),
        'pair_confident': bool(pair_ok),
        'thesis_confident': bool(thesis_ok),
        'stance': stance,
        'user_len': user_len,
    }


def deterministic_verdict_from_eval(
    ev: Dict[str, any], *, entailment_threshold: float, contradiction_threshold: float
) -> ScoreVerdict:
    ts = ev['thesis_scores']
    ps = ev['scores']
    thesis_ok = nli_confident(ts)
    pair_ok = nli_confident(ps) and len(ev.get('user_text_sample', '') or '') >= 30
    ent = ts['entailment']
    contr = ts['contradiction']

    if contr >= contradiction_threshold and contr > ent and thesis_ok:
        return {
            'alignment': 'OPPOSITE',
            'concession': True,
            'reason': 'thesis_opposition',
            'confidence': 0.8,
        }
    if ent >= entailment_threshold and ent > contr:
        return {
            'alignment': 'SAME',
            'concession': False,
            'reason': 'same_stance',
            'confidence': 0.8,
        }
    if ps['contradiction'] >= contradiction_threshold and pair_ok:
        return {
            'alignment': 'OPPOSITE',
            'concession': True,
            'reason': 'pairwise_opposition',
            'confidence': 0.7,
        }
    return {
        'alignment': 'UNKNOWN',
        'concession': False,
        'reason': 'underdetermined',
        'confidence': 0.5,
    }


# ----- Footers -----
def build_context_footer(ev: Optional[dict]) -> Optional[str]:
    if not ev:
        return None
    ts = ev.get('thesis_scores', {})
    ps = ev.get('scores', {})

    def g(d, k):
        try:
            return float(d.get(k, 0.0))
        except Exception:
            return 0.0

    topic = (ev.get('topic') or '').strip().replace('\n', ' ').replace('"', "'")
    return (
        f'[Context] align={ev.get("alignment", "UNKNOWN")} '
        f'| concession={str(ev.get("concession", False)).lower()} '
        f'| reason={ev.get("reason", "underdetermined")} '
        f'| tE={g(ts, "entailment"):.2f} tC={g(ts, "contradiction"):.2f} '
        f'| pE={g(ps, "entailment"):.2f} pC={g(ps, "contradiction"):.2f} '
        f'| topic="{topic}"'
    )


def build_score_footer(rs: RunningScores) -> str:
    return (
        f'[Score] turns={rs.turns} | opp={rs.opp} same={rs.same} unk={rs.unk} '
        f'| tE_ema={rs.tE_ema:.2f} tC_ema={rs.tC_ema:.2f} '
        f'| pE_ema={rs.pE_ema:.2f} pC_ema={rs.pC_ema:.2f}'
    )


def join_footers(*parts: Optional[str]) -> Optional[str]:
    lines = [p for p in parts if p]
    return '\n'.join(lines) if lines else None
