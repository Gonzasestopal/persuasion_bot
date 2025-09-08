from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class NLIResult:
    # Debate context
    topic: str  # normalized thesis the assistant is defending
    stance: str  # "pro" | "con"
    user_text: str  # last user message
    bot_text: str  # assistant's previous message

    # NLI evidence
    thesis_scores: Dict[
        str, float
    ]  # e.g. {"entailment": 0.22, "contradiction": 0.71, "neutral": 0.07}
    pair_best: Dict[str, float]  # best pairwise claim-vs-user scores
    max_sent_contra: float  # max contradiction across user sentences
    on_topic: bool  # whether thesis vs. user is on-topic
    user_wc: int  # user word count

    def to_dict(self) -> Dict:
        """Convert to plain dict ready for JSON serialization."""
        return asdict(self)
