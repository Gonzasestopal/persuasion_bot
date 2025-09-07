from dataclasses import dataclass


@dataclass
class TopicResult:
    reason: str
    valid: bool
