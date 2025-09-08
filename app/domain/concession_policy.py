from dataclasses import asdict, dataclass, field
from typing import Dict, Literal

from app.domain.enums import Stance
from app.settings import settings


@dataclass(frozen=True)
class ConcessionPolicy:
    required_positive_judgements: int = settings.REQUIRED_POSITIVE_JUDGEMENTS
    max_assistant_turns: int = 3  # hard cap on assistant turns


DebateStatus = Literal['ONGOING', 'ENDED']


@dataclass
class DebateState:
    # Server-authoritative core
    topic: str  # normalized thesis string
    stance: Stance  # Stance.PRO or Stance.CON

    # Language lock (set after first assistant turn)
    lang: str = 'en'
    lang_locked: bool = False

    # Counters / status
    assistant_turns: int = 0
    positive_judgements: int = 0  # count of ACCEPTs so far (from the judge)
    match_concluded: bool = False

    # Policy
    policy: ConcessionPolicy = field(default_factory=ConcessionPolicy)
    assistant_turns: int = 0
    positive_judgements: int = 0
    match_concluded: bool = False
    lang_locked: bool = False  # once True, never auto-change

    end_reason: str = ''
    ended_by: str = ''  # e.g. 'judge', 'policy:max_turns', 'policy:points'

    def maybe_conclude(self) -> bool:
        """
        End the debate if:
          - user has reached required_positive_judgements, OR
          - assistant hit max_assistant_turns, OR
          - already marked concluded.
        """
        if self.match_concluded:
            return True
        if self.positive_judgements >= self.policy.required_positive_judgements:
            return True
        if self.assistant_turns >= self.policy.max_assistant_turns:
            return True
        return False

    # ---------- Mutation helpers ----------
    def mark_concluded(self, *, reason: str, by: str) -> None:
        self.match_concluded = True
        self.end_reason = reason
        self.ended_by = by

    # ---------- Prompt wiring ----------
    def to_prompt_vars(self) -> Dict[str, str]:
        """
        Map state to placeholders consumed by your AWARE system prompt.
        (If your prompt previously used USER_POINT/USER_POINT_REASON, remove them.)
        """
        return {
            'STANCE': self.stance.value,  # "pro" | "con"
            'DEBATE_STATUS': self.debate_status,  # "ONGOING" | "ENDED"
            'TURN_INDEX': str(self.assistant_turns),
            'LANGUAGE': self.lang,  # e.g., "en"
            'TOPIC': self.topic,
        }

    # Handy for logging/debugging
    def asdict(self) -> Dict:
        return asdict(self)
