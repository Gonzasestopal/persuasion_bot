# tests/integration/test_novelty_guard.py
import pytest

from app.domain.concession_policy import DebateState
from app.domain.enums import Stance
from app.domain.models import Message

pytestmark = pytest.mark.integration


class _FakeJudgeAccept:
    """Async fake judge that always ACCEPTs with high confidence."""

    class _Result:
        def __init__(self):
            self.accept = True
            self.confidence = 0.95
            self.reason = 'strict_thesis_contradiction'
            self.metrics = {'defended_contra': 0.9, 'max_sent_contra': 0.9}

    async def nli_judge(self, *, payload):
        return self._Result()


def _mk_state(topic: str, stance: Stance, lang: str = 'en'):
    st = DebateState(topic=topic, stance=stance, lang=lang)
    return st


@pytest.mark.asyncio
async def test_novelty_guard_blocks_duplicate_accept(service):
    """
    Exact duplicate user turn must be blocked by the novelty guard,
    leaving positives at 1 after two ACCEPTs from the judge.
    """
    conv_id = 901
    topic = 'Dogs are the best human companion'
    stance = Stance.PRO

    state = _mk_state(topic, stance, lang='en')
    service.debate_store.save(conversation_id=conv_id, state=state)

    # Force ACCEPTs to isolate novelty behavior
    service.concession_service.judge = _FakeJudgeAccept()
    # (Optional) make the guard a bit stricter for safety

    u1 = "Dogs are costly and trigger allergies; they're not the best companions."

    messages_round1 = [
        Message(
            role='bot',
            message='Dogs provide unique loyalty and aid in therapy; this makes them unparalleled companions.',
        ),
        Message(role='user', message=u1),
    ]

    # First analyze: should increment positives -> 1
    _ = await service.concession_service.analyze_conversation(
        messages=messages_round1,
        stance=stance,
        conversation_id=conv_id,
        topic=topic,
        state=state,
    )
    st1 = service.debate_store.get(conv_id)
    assert st1.positive_judgements == 1

    # Second analyze with an **exact duplicate** user turn
    messages_round2 = messages_round1 + [Message(role='user', message=u1)]
    _ = await service.concession_service.analyze_conversation(
        messages=messages_round2,
        stance=stance,
        conversation_id=conv_id,
        topic=topic,
        state=state,
    )
    st2 = service.debate_store.get(conv_id)

    assert st2.positive_judgements == 1, (
        'Novelty guard failed: exact duplicate user turn should not add another positive_judgement'
    )
    # Bonus check: guard reason surfaced
    assert (st2.last_judge_reason_label or '').startswith('novelty_reject'), (
        f'Expected novelty reject label, got {st2.last_judge_reason_label!r}'
    )


@pytest.mark.asyncio
async def test_novelty_guard_allows_new_argument(service):
    """
    A substantially new user argument must pass novelty and increment positives to 2.
    """
    conv_id = 902
    topic = 'Dogs are the best human companion'
    stance = Stance.PRO

    state = _mk_state(topic, stance, lang='en')
    service.debate_store.save(conversation_id=conv_id, state=state)

    service.concession_service.judge = _FakeJudgeAccept()

    messages_round1 = [
        Message(
            role='bot',
            message='Dogs provide unique loyalty and aid in therapy; this makes them unparalleled companions.',
        ),
        Message(
            role='user',
            message="Dogs are costly and trigger allergies; they're not the best companions.",
        ),
    ]

    # First ACCEPT -> positives = 1
    _ = await service.concession_service.analyze_conversation(
        messages=messages_round1,
        stance=stance,
        conversation_id=conv_id,
        topic=topic,
        state=state,
    )
    st1 = service.debate_store.get(conv_id)
    assert st1.positive_judgements == 1

    # Second user brings *new* line of attack (maintenance/time/public safety)
    messages_round2 = messages_round1 + [
        Message(
            role='user',
            message=(
                'Beyond money/allergies, dogs demand daily training and exercise; '
                "lack of training leads to bites and public-safety incidents. That undermines them as 'best' companions."
            ),
        )
    ]

    _ = await service.concession_service.analyze_conversation(
        messages=messages_round2,
        stance=stance,
        conversation_id=conv_id,
        topic=topic,
        state=state,
    )
    st2 = service.debate_store.get(conv_id)
    assert st2.positive_judgements == 2, (
        'New argument should pass novelty and be tallied'
    )
