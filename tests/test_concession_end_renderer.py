# tests/unit/test_concession_end_renderer.py
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pytest

from app.domain.concession_policy import DebateState
from app.domain.enums import Stance
from app.domain.nli.config import NLIConfig
from app.domain.nli.scoring import ScoringConfig
from app.services.concession_service import ConcessionService

# ----- Fakes -----


class FakeDebateLLM:
    async def debate_aware(self, messages, state=None, **_):
        return 'fake-llm-reply'

    async def debate_aware_end(self, messages, prompt_vars: Dict[str, str], **_):
        return (
            f'{prompt_vars.get("END_REASON", "Explained end reason")}.\n'
            f'Reason: {prompt_vars.get("JUDGE_REASON_LABEL", "unspecified reason").replace("_", " ")} '
            f'(conf {prompt_vars.get("JUDGE_CONFIDENCE", "0.00")})'
        )


@dataclass
class _JudgeDecision:
    accept: bool
    confidence: float
    reason: str
    metrics: Dict[str, float]


class FakeJudgeLLM:
    def __init__(self, decision: Dict[str, Any] = None):
        self._raw = decision or {
            'accept': False,
            'confidence': 0.0,
            'reason': 'unused_in_this_test',
            'metrics': {},
        }

    async def nli_judge(self, *, payload: Dict[str, Any]) -> _JudgeDecision:
        return _JudgeDecision(
            accept=bool(self._raw.get('accept', False)),
            confidence=float(self._raw.get('confidence', 0.0)),
            reason=str(self._raw.get('reason', '')),
            metrics=dict(self._raw.get('metrics', {})),
        )


class FakeNLI:
    def bidirectional_scores(self, premise: str, hypothesis: str):
        # Not used in ended path
        return {
            'p_to_h': {},
            'h_to_p': {},
            'agg_max': {'entailment': 0, 'neutral': 1, 'contradiction': 0},
        }


def _msgs_for(pairs: List[Tuple[str, str]]):
    class Msg:
        def __init__(self, role, message):
            self.role = role
            self.message = message

    out = []
    for a, u in pairs:
        out.append(Msg('bot', a))
        out.append(Msg('user', u))
    return out


class DummyState(DebateState):
    def __init__(self):
        self.positive_judgements = 0
        self.assistant_turns = 0
        self.match_concluded = True  # force ended path
        self.lang = 'en'
        self.lang_locked = True
        self.topic = 'Sample thesis'
        self.stance = Stance.PRO

        class _Policy:
            required_positive_judgements = 1
            max_assistant_turns = 5

        self.policy = _Policy()

        self.last_judge_accept = True
        self.last_judge_reason_label = 'strict_thesis_contradiction'
        self.last_judge_confidence = 0.87
        self.end_reason = 'Policy threshold reached'

    def set_judge(self, *, accept: bool, reason: str, confidence: float):
        self.last_judge_accept = bool(accept)
        self.last_judge_reason_label = (reason or '').strip()
        try:
            self.last_judge_confidence = float(confidence)
        except Exception:
            self.last_judge_confidence = 0.0


@pytest.mark.asyncio
async def test_end_renderer_two_lines_reason_and_conf():
    state = DummyState()
    llm = FakeDebateLLM()
    svc = ConcessionService(
        llm=llm,
        nli=FakeNLI(),
        judge=FakeJudgeLLM(),
        nli_config=NLIConfig(),
        scoring=ScoringConfig(),
    )

    messages = _msgs_for([('Assistant long enough.', 'User final turn.')])

    reply, out_state = await svc.analyze_conversation(
        messages=messages,
        stance=Stance.PRO,
        conversation_id=123,
        topic=state.topic,
        state=state,
    )

    lines = [ln for ln in reply.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert lines[0].lower().startswith('policy threshold reached')
    assert lines[1] == 'Reason: strict thesis contradiction (conf 0.87)'
