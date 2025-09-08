# tests/unit/test_concession_service_unit.py
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest

from app.adapters.repositories.memory_debate_store import InMemoryDebateStore
from app.domain.enums import Stance
from app.domain.nli.config import NLIConfig
from app.domain.nli.scoring import ScoringConfig
from app.services.concession_service import ConcessionService

pytestmark = pytest.mark.unit


# ---------------------------- Helpers de compatibilidad ------------------------------


def _alias(d: dict, *keys):
    """
    Devuelve d[key] usando alias. Si no está al tope, intenta dentro de d['nli'].
    Lanza KeyError si ninguno existe.
    """
    for k in keys:
        if k in d:
            return d[k]
    nli = d.get('nli')
    if isinstance(nli, dict):
        for k in keys:
            if k in nli:
                return nli[k]
    raise KeyError(keys[0])


def _is_procon_or_enum(v):
    """Acepta 'pro'|'con' o instancia de Stance enum."""
    try:
        if isinstance(v, Stance):
            return True
    except Exception:
        pass
    return v in ('pro', 'con')


# ---------------------------- Fakes / Helpers ------------------------------


@dataclass
class _JudgeDecision:
    accept: bool
    confidence: float
    reason: str
    metrics: Dict[str, float]


class FakeDebateLLM:
    """
    Minimal reply LLM stub:
      - debate_aware(...) devuelve un reply estático.
    """

    async def debate_aware(self, messages, state=None, **_):
        return 'fake-llm-reply'

    # Opcional: algunos tests llaman debate_aware_end; mantener default seguro.
    async def debate_aware_end(self, messages, prompt_vars: Dict[str, str], **_):
        # End renderer determinístico.
        return (
            'Debate ended by policy or judge signal; a strong contradiction was detected.\n'
            f'Reason: {prompt_vars.get("JUDGE_REASON_LABEL", "unspecified_reason")} '
            f'(conf {prompt_vars.get("JUDGE_CONFIDENCE", "0.00")})'
        )


class FakeJudgeLLM:
    """
    Minimal judge LLM stub:
      - nli_judge(payload=...) retorna un _JudgeDecision con accept/confidence/reason/metrics
    Soporta tanto el shape nuevo (accept/metrics) como uno legacy (concession).
    """

    def __init__(self, decision: Dict[str, Any] = None):
        # Default: accept (concession) con alta confianza
        self._raw = decision or {
            'accept': True,
            'confidence': 0.9,
            'reason': 'thesis_opposition_strong',
            'metrics': {
                'defended_contra': 0.9,
                'defended_ent': 0.1,
                'max_sent_contra': 0.9,
            },
        }
        self.last_payload: Dict[str, Any] = None

    async def nli_judge(self, *, payload: Dict[str, Any]) -> _JudgeDecision:
        self.last_payload = payload

        # Mapeo flexible para soportar shapes legacy:
        accept = self._raw.get('accept')
        if accept is None:
            # Legacy: concession=True implica accept=True
            accept = bool(self._raw.get('concession', False))

        confidence = float(self._raw.get('confidence', 0.0))
        reason = str(self._raw.get('reason', 'unspecified'))
        metrics = dict(
            self._raw.get('metrics')
            or {
                'defended_contra': 0.0,
                'defended_ent': 0.0,
                'max_sent_contra': 0.0,
            }
        )
        return _JudgeDecision(
            accept=accept, confidence=confidence, reason=reason, metrics=metrics
        )


@pytest.fixture
def store():
    return InMemoryDebateStore()


def mk_dir(ent: float, neu: float, contra: float):
    # single direction scores
    return {'entailment': ent, 'neutral': neu, 'contradiction': contra}


def mk_bidir(ph, hp):
    # bidirectional package; service agregará con agg_max
    agg = {
        k: max(ph.get(k, 0.0), hp.get(k, 0.0))
        for k in ('entailment', 'neutral', 'contradiction')
    }
    return {'p_to_h': ph, 'h_to_p': hp, 'agg_max': agg}


class FakeNLI:
    """
    Devuelve paquetes bidireccionales en secuencia por cada llamada a bidirectional_scores.
    Si 'per_sentence' se provee, esos se consumen primero (para el sentence scan).
    Luego repite el último si repeat_last=True.
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
        self.topic = ''
        self.stance = Stance.PRO

        # Policy mirror (defaults usados por el service)
        class _Policy:
            required_positive_judgements = 1
            max_assistant_turns = 3

        self.policy = _Policy()
        # Info del judge usada por end rendering
        self.last_judge_accept = False
        self.last_judge_reason_label = ''
        self.last_judge_confidence = 0.0
        self.end_reason = ''

    def maybe_conclude(self):
        # concluir una vez que registramos un positive judgement
        return self.positive_judgements >= 1

    # permitir que el service setee info del judge
    def set_judge(self, *, accept: bool, reason: str, confidence: float):
        self.last_judge_accept = bool(accept)
        self.last_judge_reason_label = (reason or '').strip()
        try:
            self.last_judge_confidence = float(confidence)
        except Exception:
            self.last_judge_confidence = 0.0


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
    Judge acepta con alta confianza:
      - service incrementa positive_judgements
      - state concluye
      - service devuelve end-render o normal si la policy requiere 2 accepts
    """
    nli = FakeNLI(sequence=[mk_bidir(mk_dir(0.2, 0.6, 0.2), mk_dir(0.2, 0.6, 0.2))])
    judge = FakeJudgeLLM(
        decision={
            'accept': True,
            'confidence': 0.90,
            'reason': 'thesis_opposition_strong',
            'metrics': {
                'defended_contra': 0.9,
                'defended_ent': 0.1,
                'max_sent_contra': 0.92,
            },
        }
    )
    llm = FakeDebateLLM()

    svc = ConcessionService(
        llm=llm,
        judge=judge,
        nli=nli,
        nli_config=NLIConfig(),
        scoring=ScoringConfig(),
        debate_store=store,
    )

    conv_id = 42
    st = DummyState()
    st.topic = 'Remote work is more productive'
    st.stance = Stance.PRO
    store.save(conv_id, st)
    msgs = make_msgs()

    out = await svc.analyze_conversation(
        messages=msgs,
        stance=Stance.PRO,
        conversation_id=conv_id,
        topic=st.topic,
    )

    assert isinstance(out, str)
    assert store.get(conv_id).positive_judgements == 1
    # Judge payload sanity
    assert judge.last_payload is not None
    assert judge.last_payload['topic'] == st.topic
    assert 'stance' in judge.last_payload
    assert _is_procon_or_enum(judge.last_payload['stance'])
    # tolerar ambos nombres de clave en nivel tope o dentro de 'nli'
    assert isinstance(_alias(judge.last_payload, 'thesis_scores', 'thesis_agg'), dict)


@pytest.mark.asyncio
async def test_analyze_conversation_no_concession_high_confidence(store):
    """
    Judge rechaza (no concession) con alta confianza:
      - no incrementa
      - service devuelve un reply normal
    """
    nli = FakeNLI(sequence=[mk_bidir(mk_dir(0.3, 0.5, 0.2), mk_dir(0.35, 0.45, 0.2))])
    judge = FakeJudgeLLM(
        decision={
            'accept': False,
            'confidence': 0.88,
            'reason': 'thesis_support',
            'metrics': {
                'defended_contra': 0.2,
                'defended_ent': 0.7,
                'max_sent_contra': 0.3,
            },
        }
    )
    llm = FakeDebateLLM()

    svc = ConcessionService(
        llm=llm,
        judge=judge,
        nli=nli,
        nli_config=NLIConfig(),
        scoring=ScoringConfig(),
        debate_store=store,
    )

    conv_id = 7
    st = DummyState()
    st.topic = 'Dogs are the best companions'
    st.stance = Stance.CON
    store.save(conv_id, st)
    msgs = make_msgs()

    out = await svc.analyze_conversation(
        messages=msgs,
        stance=Stance.CON,
        conversation_id=conv_id,
        topic=st.topic,
    )

    assert isinstance(out, str)  # reply path
    assert store.get(conv_id).positive_judgements == 0
    assert judge.last_payload is not None
    assert 'stance' in judge.last_payload
    assert _is_procon_or_enum(judge.last_payload['stance'])


@pytest.mark.asyncio
async def test_analyze_conversation_concession_low_confidence_gate(store):
    """
    Judge acepta pero la confianza está por debajo del umbral:
      - no incrementa debido a llm_judge_min_confidence
    """
    nli = FakeNLI(sequence=[mk_bidir(mk_dir(0.1, 0.8, 0.1), mk_dir(0.1, 0.8, 0.1))])
    judge = FakeJudgeLLM(
        decision={
            'accept': True,
            'confidence': 0.25,  # por debajo de min_conf=0.70
            'reason': 'weak_signal',
            'metrics': {
                'defended_contra': 0.51,
                'defended_ent': 0.30,
                'max_sent_contra': 0.55,
            },
        }
    )
    llm = FakeDebateLLM()

    svc = ConcessionService(
        llm=llm,
        judge=judge,
        nli=nli,
        nli_config=NLIConfig(),
        scoring=ScoringConfig(),
        debate_store=store,
        llm_judge_min_confidence=0.70,
    )

    conv_id = 9
    st = DummyState()
    st.topic = 'X'
    st.stance = Stance.PRO
    store.save(conv_id, st)
    msgs = make_msgs()

    out = await svc.analyze_conversation(
        messages=msgs, stance=Stance.PRO, conversation_id=conv_id, topic=st.topic
    )

    assert isinstance(out, str)
    assert store.get(conv_id).positive_judgements == 0  # gated


def test_payload_builder_sentence_scan_and_on_topic_flags():
    """
    Validar que el builder calcula:
      - thesis_agg
      - max_sent_contra del sentence scanning
      - heurística on_topic
      - pair_best agregado (de la claim más contradictoria)
    """
    # Primera oración neutral; segunda fuertemente contradictoria
    sent1 = mk_dir(0.20, 0.70, 0.10)
    sent2 = mk_dir(0.10, 0.05, 0.85)

    thesis_neutral = mk_bidir(sent1, sent1)
    thesis_contra = mk_bidir(sent2, sent2)

    nli = FakeNLI(
        sequence=[thesis_neutral], per_sentence=[thesis_neutral, thesis_contra]
    )

    svc = ConcessionService(
        llm=FakeDebateLLM(),
        judge=FakeJudgeLLM(),  # no usado en builder, pero requerido por ctor
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
        state=DummyState(),
    )

    assert payload is not None
    d = payload.to_dict()
    assert d['topic'] == 'El universo requiere un creador'
    assert _is_procon_or_enum(d.get('stance'))
    ts = _alias(d, 'thesis_scores', 'thesis_agg')
    assert 0.0 <= ts['entailment'] <= 1.0
    max_contra = _alias(d, 'max_sent_contra')
    assert 0.80 <= max_contra <= 0.90  # strong second sentence
    assert isinstance(_alias(d, 'pair_best', 'pair_agg'), dict)
    assert 'user_wc' in d
    assert isinstance(user_txt, str) and isinstance(bot_txt, str)


def test_payload_builder_pair_best_from_claims():
    """
    Asegurar que pair_best se arma desde la claim más contradictoria del assistant.
    """
    # Secuencia pairwise: primera claim neutral, segunda contradice
    pair_neutral = mk_bidir(mk_dir(0.20, 0.70, 0.10), mk_dir(0.22, 0.68, 0.10))
    pair_contra = mk_bidir(mk_dir(0.10, 0.25, 0.72), mk_dir(0.12, 0.22, 0.75))

    # Thesis agg (no relevante acá)
    thesis_any = mk_bidir(mk_dir(0.30, 0.60, 0.10), mk_dir(0.32, 0.58, 0.10))
    nli = FakeNLI(sequence=[thesis_any])

    svc = ConcessionService(
        llm=FakeDebateLLM(),
        judge=FakeJudgeLLM(),  # no usado en builder
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

    # Monkeypatch _claim_scores para usar nuestros dos paquetes
    def fake_claim_scores(claims: List[str], user_clean: str):
        # Retorna tuplas (claim, ent, contra, rel, bidir_scores)
        return [
            (claims[0], 0.30, 0.10, 0.30, pair_neutral),
            (claims[1], 0.20, 0.75, 0.75, pair_contra),  # la más contradictoria
        ]

    svc._claim_scores = fake_claim_scores  # type: ignore

    payload, _, _ = svc._build_llm_judge_payload(
        conversation=conv,
        stance=Stance.CON,
        topic='Thesis T',
        state=DummyState(),
    )
    assert payload is not None
    d = payload.to_dict()
    pb = _alias(d, 'pair_best', 'pair_agg')
    assert 0.70 <= pb['contradiction'] <= 0.80  # provino de la segunda claim
    assert _is_procon_or_enum(d.get('stance'))
