# tests/unit/test_concession_judge_confidence.py
from dataclasses import dataclass
from typing import Any, Dict

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
        # Not used here
        return 'ended.\nReason: x (conf 0.00)'


@dataclass
class _JudgeDecision:
    accept: bool
    confidence: float
    reason: str
    metrics: Dict[str, float]


class FakeJudgeLLM:
    def __init__(self, decision: Dict[str, Any]):
        self._raw = decision
        self.last_payload = None

    async def nli_judge(self, *, payload: Dict[str, Any]) -> _JudgeDecision:
        self.last_payload = payload
        return _JudgeDecision(
            accept=bool(self._raw.get('accept', False)),
            confidence=float(self._raw.get('confidence', 0.0)),
            reason=str(self._raw.get('reason', '')),
            metrics=dict(self._raw.get('metrics', {})),
        )


def mk_dir(ent: float, neu: float, contra: float):
    return {'entailment': ent, 'neutral': neu, 'contradiction': contra}


def mk_bidir(ph, hp):
    agg = {
        k: max(ph.get(k, 0.0), hp.get(k, 0.0))
        for k in ('entailment', 'neutral', 'contradiction')
    }
    return {'p_to_h': ph, 'h_to_p': hp, 'agg_max': agg}


class FakeNLI:
    def __init__(self, thesis_pkg):
        self.thesis_pkg = thesis_pkg
        self.calls = 0

    def bidirectional_scores(self, premise: str, hypothesis: str):
        self.calls += 1
        return self.thesis_pkg


def _msgs_one_turn():
    class Msg:
        def __init__(self, role, message):
            self.role = role
            self.message = message

    return [
        Msg('bot', 'Assistant long enough to count.'),
        Msg('user', 'User argument text here.'),
    ]


class DummyState(DebateState):
    def __init__(self):
        self.positive_judgements = 0
        self.assistant_turns = 0
        self.match_concluded = False
        self.lang = 'en'
        self.lang_locked = True
        self.topic = 'Sample thesis'
        self.stance = Stance.CON

        class _Policy:
            required_positive_judgements = 1
            max_assistant_turns = 5

        self.policy = _Policy()

        self.last_judge_accept = False
        self.last_judge_reason_label = ''
        self.last_judge_confidence = 0.0
        self.end_reason = ''

    def set_judge(self, *, accept: bool, reason: str, confidence: float):
        self.last_judge_accept = bool(accept)
        self.last_judge_reason_label = (reason or '').strip()
        try:
            self.last_judge_confidence = float(confidence)
        except Exception:
            self.last_judge_confidence = 0.0


@pytest.mark.asyncio
async def test_confidence_gate_blocks_low_confidence_accept():
    thesis_pkg = mk_bidir(mk_dir(0.2, 0.7, 0.1), mk_dir(0.2, 0.7, 0.1))
    nli = FakeNLI(thesis_pkg)

    judge = FakeJudgeLLM(
        decision={
            'accept': True,
            'confidence': 0.30,
            'reason': 'weak_signal',
            'metrics': {},
        }
    )
    llm = FakeDebateLLM()
    state = DummyState()

    svc = ConcessionService(
        llm=llm,
        nli=nli,
        judge=judge,
        nli_config=NLIConfig(),
        scoring=ScoringConfig(),
        llm_judge_min_confidence=0.70,
    )

    reply, out_state = await svc.analyze_conversation(
        messages=_msgs_one_turn(),
        stance=Stance.CON,
        conversation_id=456,
        topic=state.topic,
        state=state,
    )

    assert out_state.positive_judgements == 0  # blocked
    assert reply == 'fake-llm-reply'  # normal path (not ended)


@pytest.mark.asyncio
async def test_high_confidence_accept_increments_and_may_end():
    thesis_pkg = mk_bidir(mk_dir(0.2, 0.6, 0.2), mk_dir(0.25, 0.55, 0.2))
    nli = FakeNLI(thesis_pkg)

    judge = FakeJudgeLLM(
        decision={
            'accept': True,
            'confidence': 0.92,
            'reason': 'strict_thesis_contradiction',
            'metrics': {'defended_contra': 0.9, 'defended_ent': 0.1},
        }
    )
    llm = FakeDebateLLM()
    state = DummyState()

    svc = ConcessionService(
        llm=llm,
        nli=nli,
        judge=judge,
        nli_config=NLIConfig(),
        scoring=ScoringConfig(),
        llm_judge_min_confidence=0.70,
    )

    reply, out_state = await svc.analyze_conversation(
        messages=_msgs_one_turn(),
        stance=Stance.PRO,
        conversation_id=789,
        topic=state.topic,
        state=state,
    )

    assert out_state.positive_judgements == 1
    # May still be ongoing if policy requires more; by default required_positive_judgements=1 → ended path
    if out_state.match_concluded:
        # End renderer used — two lines; but our FakeDebateLLM returns generic ended format only
        assert 'Reason: ' in reply
    else:
        assert reply == 'fake-llm-reply'
