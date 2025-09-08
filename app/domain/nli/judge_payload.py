from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class NLIJudgePayload:
    # Debate context
    topic: str  # normalized thesis the assistant defends
    stance: str  # "pro" | "con"
    user_text: str  # last user message
    bot_text: str  # assistant message the user replied to

    # NLI evidence
    thesis_scores: Dict[
        str, float
    ]  # {"entailment":..., "contradiction":..., "neutral":...}
    pair_best: Dict[str, float]  # best claim-vs-user aggregate scores
    max_sent_contra: float  # max contradiction vs any user sentence
    on_topic: bool  # whether thesis vs user is on-topic
    user_wc: int  # user word count

    def to_dict(self) -> Dict[str, Any]:
        """Compact dict ready for JSON serialization."""
        return asdict(self)
