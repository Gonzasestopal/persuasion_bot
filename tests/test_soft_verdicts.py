from app.services.scoring import (
    deterministic_verdict_from_eval,
    nli_confident,
)

ENT_THR = 0.70
CON_THR = 0.70


def _ev(
    *,
    t_ent: float,
    t_con: float,
    t_neu: float = 0.0,
    p_ent: float = 0.0,
    p_con: float = 0.0,
    p_neu: float = 0.0,
    user_text_len: int = 40,
    topic: str = 'gods are human best friend',
) -> dict:
    """Build a minimal evaluation dict shaped like judge_last_two_messages output."""
    # Ensure values look like probabilities but we don't strictly normalize.
    thesis_scores = {
        'entailment': float(t_ent),
        'contradiction': float(t_con),
        'neutral': float(t_neu),
    }
    pair_scores = {
        'entailment': float(p_ent),
        'contradiction': float(p_con),
        'neutral': float(p_neu),
    }
    return {
        'alignment': 'UNKNOWN',  # placeholder, not used by the function
        'scores': pair_scores,
        'thesis_scores': thesis_scores,
        'user_text_sample': 'x' * user_text_len,
        'bot_text_sample': 'previous assistant turn…',
        'topic': topic,
    }


# ---------- nli_confident soft settings ----------


def test_nli_confident_uses_softer_thresholds_and_margin():
    # Top = 0.70, next = 0.58 (margin 0.12) -> True with pmin=0.70, margin=0.10
    scores = {'entailment': 0.58, 'contradiction': 0.12, 'neutral': 0.70}
    assert nli_confident(scores, pmin=0.70, margin=0.10) is True

    # Top = 0.69 (below pmin) -> False
    scores = {'entailment': 0.69, 'contradiction': 0.20, 'neutral': 0.11}
    assert nli_confident(scores, pmin=0.70, margin=0.10) is False

    # Top = 0.72, next = 0.65 (margin 0.07 < 0.10) -> False
    scores = {'entailment': 0.72, 'contradiction': 0.65, 'neutral': 0.05}
    assert nli_confident(scores, pmin=0.70, margin=0.10) is False


# ---------- Soft band around thresholds (thesis-based) ----------


def test_soft_opposite_when_contradiction_just_below_threshold_and_entailment_low():
    # Contradiction 0.69 (within 0.08 band of 0.70), entailment low.
    # Thesis OK because neutral is clearly highest (0.80) with margin >= 0.10.
    ev = _ev(t_ent=0.20, t_con=0.69, t_neu=0.80)
    verdict = deterministic_verdict_from_eval(
        ev, entailment_threshold=ENT_THR, contradiction_threshold=CON_THR
    )
    assert verdict['alignment'] == 'OPPOSITE'
    assert verdict['reason'] == 'thesis_opposition'
    assert verdict['concession'] is True
    assert verdict['confidence'] >= 0.80


def test_soft_same_when_entailment_just_below_threshold_and_contradiction_low():
    # Entailment 0.66 (within 0.08 of 0.70) and contradiction low -> SAME
    ev = _ev(t_ent=0.66, t_con=0.30, t_neu=0.72)
    verdict = deterministic_verdict_from_eval(
        ev, entailment_threshold=ENT_THR, contradiction_threshold=CON_THR
    )
    assert verdict['alignment'] == 'SAME'
    assert verdict['reason'] == 'same_stance'
    assert verdict['concession'] is False
    assert verdict['confidence'] >= 0.80


# ---------- Pairwise carry-over when thesis inconclusive ----------


def test_pairwise_opposition_carries_when_thesis_inconclusive_and_pair_contra_near_threshold():
    # Thesis inconclusive (both below); pairwise contradiction 0.66 (within 0.06 of 0.70),
    # pairwise is confident and user text length >= 20 -> OPPOSITE via pairwise.
    ev = _ev(
        t_ent=0.40,
        t_con=0.45,
        t_neu=0.50,
        p_ent=0.10,
        p_con=0.66,
        p_neu=0.80,
        user_text_len=40,
    )
    verdict = deterministic_verdict_from_eval(
        ev, entailment_threshold=ENT_THR, contradiction_threshold=CON_THR
    )
    assert verdict['alignment'] == 'OPPOSITE'
    assert verdict['reason'] == 'pairwise_opposition'
    assert verdict['concession'] is True
    assert verdict['confidence'] >= 0.70


def test_pairwise_opposition_does_not_fire_if_user_text_too_short():
    # Same pairwise scores as above, but user text is < 20 chars, so pair_ok=False.
    ev = _ev(
        t_ent=0.40,
        t_con=0.45,
        t_neu=0.50,
        p_ent=0.10,
        p_con=0.66,
        p_neu=0.80,
        user_text_len=18,
    )
    verdict = deterministic_verdict_from_eval(
        ev, entailment_threshold=ENT_THR, contradiction_threshold=CON_THR
    )
    assert verdict['alignment'] == 'UNKNOWN'
    assert verdict['reason'] == 'underdetermined'
    assert verdict['concession'] is False
    assert verdict['confidence'] == 0.50


# ---------- Guard: truly ambiguous stays UNKNOWN ----------


def test_ambiguous_low_scores_remain_unknown():
    ev = _ev(
        t_ent=0.45,
        t_con=0.48,
        t_neu=0.52,
        p_ent=0.40,
        p_con=0.44,
        p_neu=0.51,
        user_text_len=50,
    )
    verdict = deterministic_verdict_from_eval(
        ev, entailment_threshold=ENT_THR, contradiction_threshold=CON_THR
    )
    assert verdict['alignment'] == 'UNKNOWN'
    assert verdict['reason'] == 'underdetermined'
    assert verdict['concession'] is False
    assert verdict['confidence'] == 0.50
