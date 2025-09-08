# tests/unit/test_concession_novelty_guard.py
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

    async def nli_judge(self, *, payload: Dict[str, Any]) -> _JudgeDecision:
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
        self.assistant_turns = 1  # service will sync to transcript length
        self.match_concluded = False
        self.lang = 'en'
        self.lang_locked = True
        self.topic = 'Sample thesis'
        self.stance = Stance.PRO

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
async def test_novelty_guard_blocks_duplicate_user_turn_when_accept():
    # require high novelty so duplicates get rejected
    scoring = ScoringConfig()
    setattr(scoring, 'novelty_min', 0.80)

    thesis_pkg = mk_bidir(mk_dir(0.2, 0.6, 0.2), mk_dir(0.2, 0.6, 0.2))
    nli = FakeNLI(thesis_pkg)

    judge = FakeJudgeLLM(
        decision={
            'accept': True,
            'confidence': 0.95,
            'reason': 'strict_thesis_contradiction',
            'metrics': {'defended_contra': 0.9, 'defended_ent': 0.1},
        }
    )
    llm = FakeDebateLLM()
    state = DummyState()

    earlier_user = (
        'Productivity rises with flexible hours and fewer office interruptions.'
    )
    repeated_user = (
        'Productivity rises with flexible hours and fewer office interruptions.'
    )

    messages = _msgs_for(
        [
            ('Assistant intro with enough words.', earlier_user),
            ('Assistant second turn with enough words.', repeated_user),
        ]
    )

    svc = ConcessionService(
        llm=llm,
        nli=nli,
        judge=judge,
        nli_config=NLIConfig(),
        scoring=scoring,
    )

    reply, out_state = await svc.analyze_conversation(
        messages=messages,
        stance=Stance.PRO,
        conversation_id=321,
        topic=state.topic,
        state=state,
    )

    assert out_state.positive_judgements == 0  # novelty guard blocked it
    assert reply == 'fake-llm-reply'  # still ongoing
