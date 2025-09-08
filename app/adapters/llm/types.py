from dataclasses import dataclass
from typing import Dict

from app.domain.nli.reasons import JudgeReason


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
    reason: JudgeReason  # short snake_case reason from the judge
    confidence: float  # 0..1
    metrics: Dict[
        str, float
    ]  # optional detail: {"defended_contra":0.72,"defended_ent":0.22,"max_sent_contra":0.81}
