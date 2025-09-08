import logging
import re
from typing import Any, Dict, List, Tuple

from app.nli.ops import agg_max
from app.utils.text import (
    ACK_PREFIXES,
    STANCE_BANNERS,
    drop_questions,
    looks_like_question,
    round3,
    strip_trailing_fragment,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---------- internal utils ----------


def _coerce_text(x: Any) -> str:
    """Return a safe unicode string for regex/splitting."""
    if x is None:
        return ''
    if isinstance(x, str):
        return x
    if isinstance(x, bytes):
        try:
            return x.decode('utf-8', 'ignore')
        except Exception:
            return ''
    try:
        return str(x)
    except Exception:
        return ''


_SENT_SPLIT = re.compile(r'(?<=[.!?…])\s+')


def _split_sentences(s: str) -> List[str]:
    s = _coerce_text(s).strip()
    if not s:
        return []
    return [p.strip() for p in _SENT_SPLIT.split(s) if p and p.strip()]


# ---------- public helpers ----------


def extract_claims(bot_txt: Any) -> List[str]:
    bot_txt = _coerce_text(bot_txt)
    if not bot_txt:
        return []

    raw_parts = _split_sentences(bot_txt)
    parts = strip_trailing_fragment(raw_parts)

    claims: List[str] = []
    skipped_banners = 0
    for s in parts:
        if s.endswith('?') or looks_like_question(s):
            continue
        s2 = drop_questions(s).strip()
        if not s2:
            continue
        s2_l = s2.lower()
        if any(s2_l.startswith(prefix) for prefix in ACK_PREFIXES):
            continue
        if any(b in s2_l for b in STANCE_BANNERS):
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


def claim_scores(
    nli, claims: List[str], user_clean: Any
) -> List[Tuple[str, float, float, float, Dict[str, Dict[str, float]]]]:
    user_clean = _coerce_text(user_clean)
    out: List[Tuple[str, float, float, float, Dict[str, Dict[str, float]]]] = []
    for c in claims:
        c = _coerce_text(c)
        if not c:
            continue
        sc = nli.bidirectional_scores(c, user_clean)
        agg = agg_max(sc)
        ent = float(agg.get('entailment', 0.0))
        con = float(agg.get('contradiction', 0.0))
        neu = float(agg.get('neutral', 1.0))
        rel = max(ent, con, 1.0 - neu)
        out.append((c, ent, con, rel, sc))
    return out


def on_topic_from_scores(thesis_scores: Dict[str, Dict[str, float]], scoring) -> bool:
    ph = thesis_scores.get('p_to_h', {}) or {}
    hp = thesis_scores.get('h_to_p', {}) or {}

    def has_signal(d: Dict[str, float]) -> bool:
        ent = float(d.get('entailment', 0.0))
        con = float(d.get('contradiction', 0.0))
        neu = float(d.get('neutral', 1.0))
        return (max(ent, con) >= scoring.topic_signal_min) or (
            neu <= scoring.topic_neu_max
        )

    on = has_signal(ph) or has_signal(hp)
    logger.debug('[topic] on_topic=%s | agg=%s', on, round3(agg_max(thesis_scores)))
    return on


def max_contra_self_vs_sentences(
    nli, self_thesis: Any, user_txt: Any
) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    self_thesis = _coerce_text(self_thesis)
    user_txt = _coerce_text(user_txt)

    if not user_txt or not self_thesis:
        logger.debug('[sent_scan] empty inputs')
        return 0.0, 0.0, {}

    parts = _split_sentences(user_txt)
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
        sc = nli.bidirectional_scores(self_thesis, s)
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
