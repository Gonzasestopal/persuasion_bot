from typing import Optional, TypedDict


class ScoreFeatures(TypedDict, total=False):
    # Static policy knobs
    entailment_threshold: float
    contradiction_threshold: float
    pmin: float  # for nli_confident-like checks
    margin: float
    min_user_len: int

    # Live numeric signals (from your existing eval)
    thesis_entailment: float
    thesis_contradiction: float
    pair_entailment: float
    pair_contradiction: float
    pair_confident: bool
    thesis_confident: bool


class ScoreVerdict(TypedDict, total=False):
    alignment: str  # "OPPOSITE" | "SAME" | "UNKNOWN"
    concession: bool
    reason: str  # snake_case label
    confidence: float  # 0..1


class ScoreJudgePort:
    async def score(self, *, features: ScoreFeatures) -> Optional[ScoreVerdict]:
        raise NotImplementedError
