from typing import Any, Dict, List

import pytest

from app.adapters.repositories.memory_debate_store import InMemoryDebateStore
from app.domain.enums import Stance
from app.domain.nli.config import NLIConfig
from app.domain.nli.scoring import ScoringConfig
from app.services.concession_service import ConcessionService

pytestmark = pytest.mark.unit


# ---------------------------- Fakes / Helpers ------------------------------


@pytest.fixture
def store():
    return InMemoryDebateStore()


class FakeLLM:
    """
    Minimal LLM stub:
      - debate(...) returns a static reply (service calls it after judging)
      - nli_judge(payload=...) returns a configured decision dict
    """

    def __init__(self, decision: Dict[str, Any] | None = None):
        # Default: concession with high confidence
        self.decision = decision or {
            'verdict': 'OPPOSITE',
            'concession': True,
            'confidence': 0.87,
            'reason': 'thesis_opposition_strong',
        }
        self.last_payload: Dict[str, Any] | None = None

    async def debate(self, messages, state=None):
        return 'fake-llm-reply'

    async def nli_judge(self, *, payload: Dict[str, Any]):
        self.last_payload = payload  # capture for assertions
        return self.decision


def mk_dir(ent: float, neu: float, contra: float):
    # single direction scores
    return {'entailment': ent, 'neutral': neu, 'contradiction': contra}


def mk_bidir(ph, hp):
    # bidirectional package; service will agg with agg_max
    agg = {
        k: max(ph.get(k, 0.0), hp.get(k, 0.0))
        for k in ('entailment', 'neutral', 'contradiction')
    }
    return {'p_to_h': ph, 'h_to_p': hp, 'agg_max': agg}


class FakeNLI:
    """
    Returns scripted bidirectional scores in sequence for each call to bidirectional_scores.
    If 'per_sentence' is provided, those are consumed first (for sentence scan).
    After sequence exhausted, repeats the last package if repeat_last=True.
    """

    def __init__(self, sequence, repeat_last=True, per_sentence=None):
        self.seq = list(sequence)
        self.repeat_last = repeat_last
        self.per_sentence = list(per_sentence) if per_sentence else None
        self._ps_idx = 0
        self._last_pkg = None

    def bidirectional_scores(self, premise, hypothesis):
        if self.per_sentence is not None and self._ps_idx < len(self.per_sentence):
            pkg = self.per_sentence[self._ps_idx]
            self._ps_idx += 1
            self._last_pkg = pkg
            return pkg

        if self.seq:
            pkg = self.seq.pop(0)
            self._last_pkg = pkg
            return pkg

        if self.repeat_last and self._last_pkg is not None:
            return self._last_pkg

        raise AssertionError('FakeNLI: no more scripted scores')


class DummyState:
    def __init__(self):
        self.positive_judgements = 0
        self.assistant_turns = 0
        self.match_concluded = False
        self.lang = 'en'

    def maybe_conclude(self):
        # conclude once we record a concession (used in analyze_conversation test)
        return self.positive_judgements >= 1


def make_msgs():
    class Msg:
        def __init__(self, role, message):
            self.role = role
            self.message = message

    bot = Msg(
        'bot',
        'The assistant presents a clear argument with solid evidence and several complete sentences to satisfy length.',
    )
    user = Msg(
        'user',
        'Here is the user’s sufficiently long reply that easily passes any length threshold used by the service.',
    )
    return [bot, user]


# ------------------------------- Tests -------------------------------------


@pytest.mark.asyncio
async def test_analyze_conversation_concession_and_conclude(store):
    """
    LLM judge returns a concession with high confidence:
    - service increments positive_judgements
    - state concludes (DummyState.maybe_conclude)
    - analyze_conversation returns verdict string (build_verdict path)
    """
    nli = FakeNLI(sequence=[mk_bidir(mk_dir(0.2, 0.6, 0.2), mk_dir(0.2, 0.6, 0.2))])
    llm = FakeLLM(
        decision={
            'verdict': 'OPPOSITE',
            'concession': True,
            'confidence': 0.9,
            'reason': 'thesis_opposition_strong',
        }
    )
    svc = ConcessionService(
        llm=llm,
        nli=nli,
        nli_config=NLIConfig(),
        scoring=ScoringConfig(),
        debate_store=store,
    )

    conv_id = 42
    store.save(conv_id, DummyState())
    msgs = make_msgs()

    out = await svc.analyze_conversation(
        messages=msgs,
        stance=Stance.PRO,
        conversation_id=conv_id,
        topic='Remote work is more productive',
    )

    # Concluded → returns verdict string
    assert isinstance(out, str)
    assert store.get(conv_id).positive_judgements == 1
    # Payload sanity
    assert llm.last_payload is not None
    assert llm.last_payload['topic'] == 'Remote work is more productive'
    assert llm.last_payload['stance'] == 'pro'
    assert isinstance(llm.last_payload['thesis_scores'], dict)


@pytest.mark.asyncio
async def test_analyze_conversation_no_concession_high_confidence(store):
    """
    LLM judge returns SAME / no concession with high confidence:
    - service SHOULD NOT increment positive_judgements
    - service proceeds to generate a reply (not concluding)
    """
    nli = FakeNLI(sequence=[mk_bidir(mk_dir(0.3, 0.5, 0.2), mk_dir(0.35, 0.45, 0.2))])
    llm = FakeLLM(
        decision={
            'verdict': 'SAME',
            'concession': False,
            'confidence': 0.88,
            'reason': 'thesis_support',
        }
    )
    svc = ConcessionService(
        llm=llm,
        nli=nli,
        nli_config=NLIConfig(),
        scoring=ScoringConfig(),
        debate_store=store,
    )

    conv_id = 7
    store.save(conv_id, DummyState())
    msgs = make_msgs()

    out = await svc.analyze_conversation(
        messages=msgs,
        stance=Stance.CON,
        conversation_id=conv_id,
        topic='Dogs are the best companions',
    )

    assert isinstance(out, str)  # reply path
    assert store.get(conv_id).positive_judgements == 0
    assert llm.last_payload is not None
    assert llm.last_payload['stance'] == 'con'


@pytest.mark.asyncio
async def test_analyze_conversation_concession_low_confidence_gate(store):
    """
    LLM judge returns concession but with low confidence:
    - service should *not* increment positive_judgements because of min confidence gate
    """
    nli = FakeNLI(sequence=[mk_bidir(mk_dir(0.1, 0.8, 0.1), mk_dir(0.1, 0.8, 0.1))])
    llm = FakeLLM(
        decision={
            'verdict': 'OPPOSITE',
            'concession': True,
            'confidence': 0.25,  # below default min_conf=0.70
            'reason': 'weak_signal',
        }
    )
    svc = ConcessionService(
        llm=llm,
        nli=nli,
        nli_config=NLIConfig(),
        scoring=ScoringConfig(),
        debate_store=store,
        llm_judge_min_confidence=0.70,
    )

    conv_id = 9
    store.save(conv_id, DummyState())
    msgs = make_msgs()

    out = await svc.analyze_conversation(
        messages=msgs, stance=Stance.PRO, conversation_id=conv_id, topic='X'
    )

    assert isinstance(out, str)
    assert store.get(conv_id).positive_judgements == 0  # gated


def test_payload_builder_sentence_scan_and_on_topic_flags():
    """
    Validate that the payload builder computes:
      - thesis_agg
      - max_sent_contra from sentence scanning
      - on_topic heuristic
      - pair_best aggregation (from best contradicting claim)
    """
    # First sentence neutral; second sentence strongly contradictory
    sent1 = mk_dir(0.20, 0.70, 0.10)
    sent2 = mk_dir(0.10, 0.05, 0.85)

    thesis_neutral = mk_bidir(sent1, sent1)
    thesis_contra = mk_bidir(sent2, sent2)

    # FakeNLI will first be used to compute thesis-level scores (once), and also in sentence scan
    # Here we want the sentence scan to hit both, resulting in max_sent_contra≈0.85
    nli = FakeNLI(
        sequence=[thesis_neutral], per_sentence=[thesis_neutral, thesis_contra]
    )

    svc = ConcessionService(
        llm=FakeLLM(),
        nli=nli,
        nli_config=NLIConfig(),
        scoring=ScoringConfig(),
        debate_store=InMemoryDebateStore(),
    )

    conv = [
        {
            'role': 'assistant',
            'content': 'Assistant long message with multiple sentences to meet min words.',
        },
        {
            'role': 'user',
            'content': 'Primera oración neutral. Sin embargo, esto contradice claramente la tesis.',
        },
    ]

    payload, user_txt, bot_txt = svc._build_llm_judge_payload(
        conversation=conv,
        stance=Stance.PRO,
        topic='El universo requiere un creador',
    )

    assert payload is not None
    d = payload.to_dict()
    assert d['topic'] == 'El universo requiere un creador'
    assert d['stance'] == 'pro'
    assert 0.0 <= d['thesis_scores']['entailment'] <= 1.0
    assert 0.80 <= d['max_sent_contra'] <= 0.90  # catches the strong second sentence
    assert isinstance(d['pair_best'], dict)
    assert 'user_wc' in d
    assert isinstance(user_txt, str) and isinstance(bot_txt, str)


def test_payload_builder_pair_best_from_claims():
    """
    Ensure pair_best aggregates from the most contradictory assistant claim.
    """
    # Pairwise scoring sequence: first claim neutral, second claim contradicts
    pair_neutral = mk_bidir(mk_dir(0.20, 0.70, 0.10), mk_dir(0.22, 0.68, 0.10))
    pair_contra = mk_bidir(mk_dir(0.10, 0.25, 0.72), mk_dir(0.12, 0.22, 0.75))

    # Thesis agg (not important here)
    thesis_any = mk_bidir(mk_dir(0.30, 0.60, 0.10), mk_dir(0.32, 0.58, 0.10))
    nli = FakeNLI(sequence=[thesis_any])
    nli.per_sentence = None  # claims path doesn't use per_sentence

    svc = ConcessionService(
        llm=FakeLLM(),
        nli=nli,
        nli_config=NLIConfig(),
        scoring=ScoringConfig(),
        debate_store=InMemoryDebateStore(),
    )

    conv = [
        {
            'role': 'assistant',
            'content': 'First claim is tentative. Second claim clearly rejects the thesis.',
        },
        {
            'role': 'user',
            'content': 'Long enough user text to consider claims and compute pairwise scores.',
        },
    ]

    # Monkeypatch _claim_scores to use our two scripted packages
    def fake_claim_scores(claims: List[str], user_clean: str):
        # Return tuples like (claim, ent, contra, rel, bidir_scores)
        return [
            (claims[0], 0.30, 0.10, 0.30, pair_neutral),
            (claims[1], 0.20, 0.75, 0.75, pair_contra),  # most contradictory
        ]

    svc._claim_scores = fake_claim_scores  # type: ignore

    payload, _, _ = svc._build_llm_judge_payload(
        conversation=conv,
        stance=Stance.CON,
        topic='Thesis T',
    )
    assert payload is not None
    d = payload.to_dict()
    assert 0.70 <= d['pair_best']['contradiction'] <= 0.80  # came from second claim
    assert d['stance'] == 'con'
