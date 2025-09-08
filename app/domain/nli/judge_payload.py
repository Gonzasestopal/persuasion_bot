# app/domain/nli/judge_payload.py
from dataclasses import dataclass
from typing import Any, Dict, Literal

StanceStr = Literal['pro', 'con']
LangStr = Literal['en', 'es', 'pt']

NLITriple = Dict[str, float]  # {"entailment": x, "contradiction": y, "neutral": z}
PolicyDict = Dict[
    str, int
]  # {"required_positive_judgements": n, "max_assistant_turns": m}
ProgressDict = Dict[str, int]  # {"positive_judgements": a, "assistant_turns": b}


@dataclass
class NLIJudgePayload:
    # Core
    topic: str  # normalized thesis the assistant defends
    stance: StanceStr  # "pro" or "con" relative to topic
    language: LangStr  # locked language for replies
    turn_index: int  # assistant turns so far (0-based)

    # Conversation slice
    user_text: str  # user's last message
    bot_text: str  # assistant's previous message

    # NLI evidence
    thesis_scores: NLITriple  # aggregated thesis vs user scores
    pair_best: NLITriple  # best claim-vs-user scores
    max_sent_contra: float  # max contradiction vs any user sentence
    on_topic: bool
    user_wc: int

    # Policy and progress
    policy: PolicyDict  # required_positive_judgements, max_assistant_turns
    progress: ProgressDict  # current positive_judgements, assistant_turns

    # Serialize for the judge system prompt
    def to_dict(self) -> Dict[str, Any]:
        def clamp01(x: float) -> float:
            try:
                v = float(x)
            except Exception:
                return 0.0
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return v

        ts = {
            'entailment': clamp01(self.thesis_scores['entailment']),
            'contradiction': clamp01(self.thesis_scores['contradiction']),
            'neutral': clamp01(self.thesis_scores['neutral']),
        }
        pb = {
            'entailment': clamp01(self.pair_best['entailment']),
            'contradiction': clamp01(self.pair_best['contradiction']),
            'neutral': clamp01(self.pair_best['neutral']),
        }

        return {
            'topic': self.topic.strip(),
            'stance': self.stance,
            'language': self.language,
            'turn_index': int(self.turn_index),
            'user_text': self.user_text,
            'bot_text': self.bot_text,
            'nli': {
                'thesis_scores': ts,
                'pair_best': pb,
                'max_sent_contra': clamp01(self.max_sent_contra),
                'on_topic': bool(self.on_topic),
                'user_wc': int(self.user_wc),
            },
            'policy': {
                'required_positive_judgements': int(
                    self.policy['required_positive_judgements']
                ),
                'max_assistant_turns': int(self.policy['max_assistant_turns']),
            },
            'progress': {
                'positive_judgements': int(self.progress['positive_judgements']),
                'assistant_turns': int(self.progress['assistant_turns']),
            },
        }
