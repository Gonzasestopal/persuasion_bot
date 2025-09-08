from dataclasses import dataclass


@dataclass
class TopicResult:
    is_valid: bool
    normalized: str
    raw: str
    normalized_stance: str
    reason: str = None


@dataclass
class JudgeResult:
    accept: bool
    ended: bool
    reason: str  # short snake_case reason from the judge
    assistant_reply: (
        str  # the assistant's next turn (language-locked, ≤80 words, 1 question)
    )
    confidence: float  # 0..1
