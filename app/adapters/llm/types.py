from dataclasses import dataclass


@dataclass
class TopicResult:
    is_valid: bool
    normalized: str
    reason: str = None
