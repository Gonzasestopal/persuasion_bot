from dataclasses import dataclass


@dataclass
class TopicResult:
    is_valid: bool
    normalized: str
    raw: str
    normalized_stance: str
    reason: str = None
