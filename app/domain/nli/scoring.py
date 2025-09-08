from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringConfig:
    # Core contradiction thresholds
    contradiction_threshold: float = 0.55
    strict_contra_threshold: float = 0.55  # ↓ from 0.85 (used as contra_min)

    min_assistant_words: int = 8

    # Soft contradiction gates
    contradiction_threshold_soft: float = 0.48
    min_delta_con_neu: float = 0.08
    eps_contra_vs_neu: float = 0.03

    # Allow soft-OK if entailment isn’t high
    max_ent_for_soft: float = 0.50
    margin_ec_soft: float = 0.02
    soft_negation_discount: float = 0.10

    # SAME-stance (optional legacy paths)
    min_ent_for_same: float = 0.70
    margin_ent_vs_neu: float = 0.25
    max_contra_for_same: float = 0.40
    margin_ec: float = 0.02
    eps_ent: float = 0.20

    # Sentence-level probe
    sentence_probe_min: float = 0.45

    # Topic gate
    topic_signal_min: float = 0.35
    topic_neu_max: float = 0.70

    # Engagement / length gates
    min_user_words: int = 8
    relatedness_min: float = 0.35  # ↓ from 0.40 (slightly easier to engage)

    # NEW: stance-level support gates used by ConcessionService
    support_min: float = 0.50  # ↓ from 0.55
    margin_min: float = 0.15  # ↓ from 0.20
