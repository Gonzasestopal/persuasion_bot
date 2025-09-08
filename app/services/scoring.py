# app/services/scoring.py
import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.adapters.nli.hf_nli import HFNLIProvider
from app.domain.ports.scoring import ScoreFeatures, ScoreVerdict
from app.domain.scoring import ContextSignal, ScoreSignal

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
        alpha: float = 0.40,
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


# ----- Core helpers -----
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
    scores: Dict[str, float], *, pmin: float = 0.70, margin: float = 0.10
) -> bool:
    """
    Softer confidence: lower pmin and margin.
    True if top prob >= pmin and (top - second) >= margin.
    """
    vals = sorted((float(v) for v in scores.values()), reverse=True)
    if not vals:
        return False
    if len(vals) == 1:
        return vals[0] >= pmin
    return vals[0] >= pmin and (vals[0] - vals[1]) >= margin


def _soft_label_from_scores(
    ent: float,
    contr: float,
    *,
    ent_thr: float,
    contr_thr: float,
    soft_band: float = 0.08,  # 8% slack band
) -> str:
    """
    Soft band around thresholds with guard against the opposite side being strong.
    """
    # Prefer contradiction if near/above threshold and entailment is not strong
    if contr >= contr_thr or (contr >= contr_thr - soft_band and ent < 0.55):
        return 'OPPOSITE'
    # Prefer entailment if near/above threshold and contradiction is not strong
    if ent >= ent_thr or (ent >= ent_thr - soft_band and contr < 0.55):
        return 'SAME'
    return 'UNKNOWN'


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
    side: str,
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
        side,
        topic,
        entailment_threshold=entailment_threshold,
        contradiction_threshold=contradiction_threshold,
    )
    return {
        'passed_stance': side,
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
    side: str,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> ScoreFeatures:
    ts = ev['thesis_scores']
    ps = ev['scores']
    user_len = len(ev.get('user_text_sample', '') or '')
    # Keep for telemetry; not used to block soft cases
    return {
        'entailment_threshold': entailment_threshold,
        'contradiction_threshold': contradiction_threshold,
        'pmin': 0.66,  # softer
        'margin': 0.06,  # softer
        'min_user_len': 8,  # softer
        'thesis_entailment': float(ts.get('entailment', 0.0)),
        'thesis_contradiction': float(ts.get('contradiction', 0.0)),
        'pair_entailment': float(ps.get('entailment', 0.0)),
        'pair_contradiction': float(ps.get('contradiction', 0.0)),
        'pair_confident': True,  # don’t hard-gate; logging only
        'thesis_confident': True,  # don’t hard-gate; logging only
        'side': side,
        'user_len': user_len,
    }


def _soft_label_from_scores(
    ent: float,
    contr: float,
    *,
    ent_thr: float,
    contr_thr: float,
    soft_band: float = 0.08,  # 8% slack band
) -> str:
    """
    Soft band around thresholds with guard against the opposite side being strong.
    """
    # Prefer contradiction if near/above threshold and entailment is not strong
    if contr >= contr_thr or (contr >= contr_thr - soft_band and ent < 0.55):
        return 'OPPOSITE'
    # Prefer entailment if near/above threshold and contradiction is not strong
    if ent >= ent_thr or (ent >= ent_thr - soft_band and contr < 0.55):
        return 'SAME'
    return 'UNKNOWN'


def deterministic_verdict_from_eval(
    ev: Dict[str, any], *, entailment_threshold: float, contradiction_threshold: float
) -> ScoreVerdict:
    """
    SOFT++ verdicting:
      - Wide soft bands around thresholds
      - Minimal separation requirements
      - Pairwise carry with tiny user texts
      - No nli_confident gating for soft cases
    """
    ts = ev['thesis_scores']  # user -> thesis
    ps = ev['scores']  # user <-> bot

    ent = float(ts.get('entailment', 0.0))
    con = float(ts.get('contradiction', 0.0))
    p_ent = float(ps.get('entailment', 0.0))
    p_con = float(ps.get('contradiction', 0.0))

    user_len = len(ev.get('user_text_sample', '') or '')

    # Very soft bands
    thesis_band = 0.12  # accept within 12 points below threshold
    pair_band = 0.10  # pairwise carry within 10 points
    weak_guard = 0.60  # accept soft side if the other side < 0.60
    sep_guard = 0.00  # no separation needed beyond being >= other (ultra soft)
    min_user_len_pair = 8  # allow short texts to carry

    # ---- Thesis-driven (prefer contradiction first for clashes) ----
    # OPPOSITE if contradiction is above (thr - band) and at least not worse than entailment,
    # and entailment isn't strongly high.
    if (
        con >= (contradiction_threshold - thesis_band)
        and (con - ent) >= sep_guard
        and ent < weak_guard
    ):
        conf = 0.88 if con >= contradiction_threshold else 0.78
        return {
            'alignment': 'OPPOSITE',
            'concession': True,
            'reason': 'thesis_opposition_soft',
            'confidence': conf,
        }

    # SAME if entailment is above (thr - band) and at least not worse than contradiction,
    # and contradiction isn't strongly high.
    if (
        ent >= (entailment_threshold - thesis_band)
        and (ent - con) >= sep_guard
        and con < weak_guard
    ):
        conf = 0.88 if ent >= entailment_threshold else 0.78
        return {
            'alignment': 'SAME',
            'concession': False,
            'reason': 'same_stance_soft',
            'confidence': conf,
        }

    # ---- Pairwise carry when thesis inconclusive ----
    if user_len >= min_user_len_pair and p_con >= (contradiction_threshold - pair_band):
        return {
            'alignment': 'OPPOSITE',
            'concession': True,
            'reason': 'pairwise_opposition_soft',
            'confidence': 0.72,
        }

    # ---- Fallback ----
    return {
        'alignment': 'UNKNOWN',
        'concession': False,
        'reason': 'underdetermined',
        'confidence': 0.50,
    }


# ----- Signals (internal, hidden) -----
def build_context_signal(ev: Optional[dict]) -> Optional[ContextSignal]:
    if not ev:
        return None
    ts = ev.get('thesis_scores', {}) or {}
    ps = ev.get('scores', {}) or {}

    def g(d: Dict[str, float], k: str) -> float:
        try:
            return float(d.get(k, 0.0))
        except Exception:
            return 0.0

    topic = (ev.get('topic') or '').strip().replace('\n', ' ').replace('"', "'")
    return ContextSignal(
        align=ev.get('alignment', 'UNKNOWN'),
        concession=bool(ev.get('concession', False)),
        reason=ev.get('reason', 'underdetermined'),
        tE=g(ts, 'entailment'),
        tC=g(ts, 'contradiction'),
        pE=g(ps, 'entailment'),
        pC=g(ps, 'contradiction'),
        topic=topic,
    )


def build_score_signal(rs: RunningScores) -> ScoreSignal:
    return ScoreSignal(
        turns=rs.turns,
        opp=rs.opp,
        same=rs.same,
        unk=rs.unk,
        tE_ema=rs.tE_ema,
        tC_ema=rs.tC_ema,
        pE_ema=rs.pE_ema,
        pC_ema=rs.pC_ema,
    )


def make_scoring_system_message(
    ctx: Optional[ContextSignal], agg: Optional[ScoreSignal]
) -> Optional[str]:
    if not ctx and not agg:
        return None
    payload: Dict[str, dict] = {}
    if ctx:
        payload['context'] = ctx.to_dict()
    if agg:
        payload['score'] = agg.to_dict()
    return f'<SCORING>{json.dumps(payload, ensure_ascii=False)}</SCORING>'
