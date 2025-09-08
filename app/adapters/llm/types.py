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
    verdict: str  # "SAME" | "OPPOSITE" | "UNKNOWN"
    concession: bool
    confidence: float  # 0..1
    reason: str  # short snake_case
