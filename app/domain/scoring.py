from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class ContextSignal:
    align: str = 'UNKNOWN'  # OPPOSITE | SAME | UNKNOWN
    concession: bool = False
    reason: str = 'underdetermined'
    tE: float = 0.0
    tC: float = 0.0
    pE: float = 0.0
    pC: float = 0.0
    topic: str = ''

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ScoreSignal:
    turns: int = 0
    opp: int = 0
    same: int = 0
    unk: int = 0
    tE_ema: float = 0.0
    tC_ema: float = 0.0
    pE_ema: float = 0.0
    pC_ema: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)
