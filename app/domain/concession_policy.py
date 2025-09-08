from dataclasses import asdict, dataclass, field
from typing import Dict, Literal

from app.domain.enums import Stance
from app.settings import settings


@dataclass(frozen=True)
class ConcessionPolicy:
    required_positive_judgements: int = settings.REQUIRED_POSITIVE_JUDGEMENTS
    max_assistant_turns: int = 9  # hard cap on assistant turns


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
    last_reason: str = ''  # last judge reason (snake_case)

    last_judge_accept: bool = False
    last_judge_reason_label: str = (
        ''  # snake_case from judge, e.g. "on_topic_clear_contradiction"
    )
    last_judge_confidence: float = 0.0  # 0..1

    # UI flags
    show_telemetry: bool = True  # if True, show banner in replies

    @property
    def debate_status(self) -> DebateStatus:
        return 'ENDED' if self.match_concluded else 'ONGOING'

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

    def mark_concluded(self) -> None:
        self.match_concluded = True

    def set_judge(self, *, accept: bool, reason: str, confidence: float) -> None:
        self.last_judge_accept = bool(accept)
        self.last_judge_reason_label = (reason or '').strip()
        try:
            self.last_judge_confidence = float(confidence)
        except Exception:
            self.last_judge_confidence = 0.0

    # ---------- Prompt wiring ----------
    def to_prompt_vars(self) -> Dict[str, str]:
        return {
            'STANCE': self.stance,
            'DEBATE_STATUS': self.debate_status,
            'TURN_INDEX': str(self.assistant_turns),
            'LANGUAGE': self.lang,
            'TOPIC': self.topic,
            'JUDGE_ACCEPT': 'accept' if self.last_judge_accept else 'reject',
            'JUDGE_REASON_LABEL': self.last_judge_reason_label,
            'JUDGE_CONFIDENCE': f'{self.last_judge_confidence:.2f}',
            'END_REASON': self.end_reason or self.last_judge_reason_label or '',
            'POSITIVE_JUDGEMENTS': str(self.positive_judgements),
            'REQUIRED_POSITIVE_JUDGEMENTS': str(
                self.policy.required_positive_judgements
            ),
            'MAX_ASSISTANT_TURNS': str(self.policy.max_assistant_turns),
        }

    # Handy for logging/debugging
    def asdict(self) -> Dict:
        return asdict(self)
