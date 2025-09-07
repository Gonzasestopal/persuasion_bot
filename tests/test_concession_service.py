import pytest

from app.adapters.repositories.memory_debate_store import InMemoryDebateStore
from app.domain.nli.scoring import ScoringConfig
from app.domain.ports.llm import (
    LLMPort,
)  # only for typing; any object with .debate(...) works
from app.services.concession_service import ConcessionService, NLIConfig, Stance

pytestmark = pytest.mark.unit


# ---------------------------- Fakes / Helpers ------------------------------
@pytest.fixture
def store():
    return InMemoryDebateStore()


class FakeLLM(LLMPort):
    async def debate(self, messages):
        return 'fake-llm-reply'

    async def generate(self):
        return 'i would glady'

    async def check_topic(self, topic):
        return True


def mk_dir(ent: float, neu: float, contra: float):
    # one direction scores
    return {'entailment': ent, 'neutral': neu, 'contradiction': contra}


def mk_bidir(ph, hp):
    # build bidirectional package + agg_max
    agg = {
        k: max(ph.get(k, 0.0), hp.get(k, 0.0))
        for k in ('entailment', 'neutral', 'contradiction')
    }
    return {'p_to_h': ph, 'h_to_p': hp, 'agg_max': agg}


class FakeNLI:
    """
    Devuelve una secuencia preprogramada y, si se agota, reutiliza el último
    paquete (repeat_last=True). Opcionalmente, puede devolver una lista
    específica por oración (per_sentence) que consumirá primero.
    """

    def __init__(self, sequence, repeat_last=True, per_sentence=None):
        self.seq = list(sequence)
        self.repeat_last = repeat_last
        self.per_sentence = list(per_sentence) if per_sentence else None
        self._ps_idx = 0
        self._last_pkg = None

    def bidirectional_scores(self, premise, hypothesis):
        # Si hay paquetes por oración, consúmelos primero (útil para _max_contra_sentence)
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

    # assistant message must have >= 10 words (alphabetic) to be “valid”
    bot = Msg(
        'bot',
        'Presento mi argumento principal. Considera la evidencia empírica disponible y los efectos observados.',
    )
    user = Msg(
        'user',
        'Aquí va la respuesta extensa del usuario que contiene más de treinta palabras para pasar la verificación de longitud.',
    )
    return [bot, user]


# ------------------------------- Tests -------------------------------------


def test_thesis_contradiction_triggers_concession():
    # pair: neutral, thesis: contradiction (strong)
    pair_neutral = mk_bidir(
        mk_dir(0.10, 0.82, 0.08),
        mk_dir(0.12, 0.80, 0.08),
    )
    thesis_contra = mk_bidir(
        mk_dir(0.10, 0.75, 0.80),
        mk_dir(0.10, 0.70, 0.82),
    )
    nli = FakeNLI([pair_neutral, thesis_contra])
    svc = ConcessionService(
        llm=FakeLLM(), nli=nli, nli_config=NLIConfig(), scoring=ScoringConfig()
    )
    conv = [
        {
            'role': 'assistant',
            'content': (
                'el asistente presenta un argumento claro con evidencia solida y varias '
                'oraciones completas para validar adecuadamente la longitud requerida'
            ),
        },
        {
            'role': 'user',
            'content': (
                'respuesta del usuario con suficiente longitud para validar el fallback y '
                'probar el camino de contradiccion de la tesis'
            ),
        },
    ]

    out = svc.judge_last_two_messages(
        conv, Stance.PRO, topic='El trabajo remoto es más productivo'
    )
    assert out['concession'] is True
    assert out['alignment'] == 'OPPOSITE'
    assert out['reason'] == 'thesis_opposition_soft'


def test_thesis_support_same_no_concession():
    # pair: neutral, thesis: support via h→p (arguments imply claim)
    pair_neutral = mk_bidir(
        mk_dir(0.10, 0.82, 0.08),
        mk_dir(0.12, 0.80, 0.08),
    )
    thesis_support = mk_bidir(
        mk_dir(0.20, 0.65, 0.15),
        mk_dir(0.82, 0.50, 0.10),
    )
    nli = FakeNLI([pair_neutral, thesis_support])
    svc = ConcessionService(
        llm=FakeLLM(), nli=nli, nli_config=NLIConfig(), scoring=ScoringConfig()
    )
    conv = [
        {
            'role': 'assistant',
            'content': (
                'el asistente presenta un argumento claro con evidencia solida y varias '
                'oraciones completas para validar adecuadamente la longitud requerida'
            ),
        },
        {
            'role': 'user',
            'content': (
                'respuesta del usuario con suficiente longitud para validar el fallback y '
                'probar el camino de contradiccion de la tesis'
            ),
        },
    ]

    out = svc.judge_last_two_messages(
        conv, Stance.PRO, topic='Los perros son los mejores amigos del ser humano'
    )
    assert out['alignment'] == 'SAME'
    assert out['concession'] is False  # support does NOT count


def test_pairwise_contradiction_fallback():
    # thesis: neutral, pair: contradiction strong → fallback triggers concession
    pair_contra = mk_bidir(
        mk_dir(0.10, 0.20, 0.78),
        mk_dir(0.12, 0.18, 0.80),
    )
    thesis_neutral = mk_bidir(
        mk_dir(0.20, 0.70, 0.10),
        mk_dir(0.22, 0.68, 0.10),
    )
    nli = FakeNLI(
        [pair_contra, thesis_neutral]
    )  # note: pair is computed first in service
    svc = ConcessionService(
        llm=FakeLLM(), nli=nli, nli_config=NLIConfig(), scoring=ScoringConfig()
    )
    # ensure user has >= 30 chars/words; our service checks length >= 30 for fallback
    conv = [
        {
            'role': 'assistant',
            'content': 'el asistente escribe un texto suficientemente largo con más de diez palabras válidas en total.',
        },
        {
            'role': 'user',
            'content': 'este es un texto de usuario bastante largo que supera con facilidad el umbral de longitud exigido por el servicio.',
        },
    ]

    out = svc.judge_last_two_messages(
        conv, Stance.CON, topic='Las redes sociales han mejorado la conexión humana'
    )
    assert out['alignment'] == 'OPPOSITE'
    assert out['reason'] == 'pairwise_opposition_soft'
    assert out['concession'] is True


def test_underdetermined_no_concession():
    # both neutral → unknown
    pair_neutral = mk_bidir(
        mk_dir(0.20, 0.70, 0.10),
        mk_dir(0.22, 0.68, 0.10),
    )
    thesis_neutral = mk_bidir(
        mk_dir(0.25, 0.65, 0.10),
        mk_dir(0.24, 0.66, 0.10),
    )
    nli = FakeNLI([pair_neutral, thesis_neutral])
    svc = ConcessionService(
        llm=FakeLLM(), nli=nli, nli_config=NLIConfig(), scoring=ScoringConfig()
    )
    conv = [
        {
            'role': 'assistant',
            'content': 'texto del asistente con varias oraciones para cumplir el mínimo de palabras requerido.',
        },
        {
            'role': 'user',
            'content': 'texto del usuario sin posicionamiento claro ni relación directa con la tesis.',
        },
    ]

    out = svc.judge_last_two_messages(conv, Stance.PRO, topic='Tema cualquiera')
    assert out['alignment'] == 'UNKNOWN'
    assert out['concession'] is False


@pytest.mark.asyncio
async def test_analyze_conversation_increments_on_contradiction_and_concludes(
    monkeypatch, store
):
    # Make state conclude after first concession
    nli = FakeNLI(
        [
            # pair neutral, thesis contradiction strong
            mk_bidir(mk_dir(0.10, 0.82, 0.08), mk_dir(0.12, 0.80, 0.08)),
            mk_bidir(mk_dir(0.10, 0.75, 0.80), mk_dir(0.10, 0.70, 0.82)),
        ]
    )
    svc = ConcessionService(
        llm=FakeLLM(),
        nli=nli,
        nli_config=NLIConfig(),
        scoring=ScoringConfig(),
        debate_store=store,
    )

    # Inject dummy state that concludes after one positive judgement
    conv_id = 42
    svc.debate_store.save(conv_id, DummyState())

    msgs = make_msgs()
    # The service expects Message objects with .role and .message;
    # our make_msgs returns a simple class that matches that.
    result = await svc.analyze_conversation(
        messages=msgs, stance=Stance.PRO, conversation_id=conv_id, topic='Tema X'
    )

    # Since maybe_conclude() returns True after first concession, analyze_conversation returns verdict string
    assert isinstance(result, str)
    assert svc.debate_store.get(conv_id).positive_judgements == 1


def test_short_user_blocks_concession_on_thesis_contradiction():
    """
    Usuario muy corto → NO conceder, aunque la tesis sea contradicha,
    siempre que la contradicción NO alcance strict_contra_threshold.
    """
    # pair neutral
    pair_neutral = mk_bidir(
        mk_dir(0.10, 0.82, 0.08),
        mk_dir(0.12, 0.80, 0.08),
    )
    # tesis contradicción fuerte pero < 0.90 (p. ej. 0.85)
    thesis_contra = mk_bidir(
        mk_dir(0.10, 0.10, 0.85),
        mk_dir(0.10, 0.15, 0.83),
    )

    nli = FakeNLI([pair_neutral, thesis_contra])
    cfg = ScoringConfig(min_user_words=8, strict_contra_threshold=0.90)
    svc = ConcessionService(llm=FakeLLM(), nli=nli, scoring=cfg)

    conv = [
        # Asistente válido (≥10 palabras alfabéticas)
        {
            'role': 'assistant',
            'content': 'el asistente presenta un argumento claro con evidencia solida y varias oraciones completas',
        },
        # Usuario MUY corto (menos de 8 palabras)
        {'role': 'user', 'content': 'no estoy de acuerdo'},
    ]

    out = svc.judge_last_two_messages(
        conv, Stance.PRO, topic='El trabajo remoto es más productivo'
    )
    assert out['concession'] is False
    assert out['reason'] == 'too_short'
    assert (
        out['alignment'] == 'UNKNOWN'
    )  # la rama too_short deja el alineamiento como UNKNOWN


def test_strong_thesis_contradiction_overrides_min_words():
    """
    Contradicción “extra fuerte” de la tesis (≥ strict_contra_threshold) → concede,
    incluso si el usuario es breve.
    """
    pair_neutral = mk_bidir(
        mk_dir(0.10, 0.82, 0.08),
        mk_dir(0.12, 0.80, 0.08),
    )
    # tesis contradicción ≥ 0.90
    thesis_contra_strong = mk_bidir(
        mk_dir(0.05, 0.05, 0.93),
        mk_dir(0.06, 0.07, 0.92),
    )

    nli = FakeNLI([pair_neutral, thesis_contra_strong])
    cfg = ScoringConfig(min_user_words=8, strict_contra_threshold=0.90)
    svc = ConcessionService(llm=FakeLLM(), nli=nli, scoring=cfg)

    conv = [
        {
            'role': 'assistant',
            'content': 'el asistente presenta un argumento claro con evidencia solida y varias oraciones completas',
        },
        {'role': 'user', 'content': 'no'},
    ]

    out = svc.judge_last_two_messages(
        conv, Stance.CON, topic='Las redes sociales han mejorado la conexión humana'
    )
    assert out['concession'] is True
    assert out['reason'] == 'thesis_opposition_soft'
    assert out['alignment'] == 'OPPOSITE'


def test_short_user_blocks_pairwise_fallback():
    """
    Si la tesis es neutral pero hay contradicción por pares,
    el fallback NO debe activar concesión cuando el usuario es corto.
    """
    # pair contradicción fuerte
    pair_contra = mk_bidir(
        mk_dir(0.10, 0.20, 0.78),
        mk_dir(0.12, 0.18, 0.80),
    )
    # tesis neutral
    thesis_neutral = mk_bidir(
        mk_dir(0.25, 0.65, 0.10),
        mk_dir(0.24, 0.66, 0.10),
    )

    nli = FakeNLI(
        [pair_contra, thesis_neutral]
    )  # pair se evalúa primero en el servicio
    cfg = ScoringConfig(min_user_words=8, strict_contra_threshold=0.90)
    svc = ConcessionService(llm=FakeLLM(), nli=nli, scoring=cfg)

    conv = [
        {
            'role': 'assistant',
            'content': 'el asistente presenta un argumento claro con evidencia solida y varias oraciones completas',
        },
        {'role': 'user', 'content': 'eso no'},
    ]

    out = svc.judge_last_two_messages(conv, Stance.PRO, topic='Tema cualquiera')
    assert out['concession'] is False
    assert out['reason'] == 'too_short'
    assert out['alignment'] == 'UNKNOWN'


def test_off_topic_blocks_pairwise_concession_even_if_pair_contradiction():
    """
    Si la relación con la TESIS es off-topic (muy neutral en ambas direcciones),
    el fallback por pares NO debe conceder, aunque haya contradicción fuerte pairwise.
    """
    # pair: contradicción fuerte (para forzar el fallback si fuera on-topic)
    pair_contra = mk_bidir(
        mk_dir(0.02, 0.18, 0.80),
        mk_dir(0.02, 0.16, 0.82),
    )

    # tesis: off-topic → muy neutral en ambas direcciones y sin señal (ent/contra bajos)
    # TOPIC_NEU_MAX = 0.75 y TOPIC_SIGNAL_MIN = 0.30 en el servicio
    thesis_offtopic = mk_bidir(
        mk_dir(0.04, 0.92, 0.04),
        mk_dir(0.05, 0.90, 0.05),
    )

    nli = FakeNLI([pair_contra, thesis_offtopic])
    svc = ConcessionService(llm=FakeLLM(), nli=nli)

    conv = [
        {
            'role': 'assistant',
            'content': 'el asistente presenta un argumento claro con evidencia solida y varias oraciones completas',
        },
        {
            'role': 'user',
            # suficiente longitud para no activar 'too_short'
            'content': 'hablo sobre recetas de cocina y especias aromaticas nada que ver con productividad laboral remota',
        },
    ]

    out = svc.judge_last_two_messages(
        conv, Stance.PRO, topic='El trabajo remoto es más productivo'
    )
    assert out['concession'] is False
    assert out['alignment'] == 'UNKNOWN'
    assert out['reason'] == 'off_topic'


def test_multilingual_thesis_contradiction_spanish_user():
    """
    Thesis in EN, user reply in ES. Service should still mark contradiction
    (we script NLI to return strong thesis-contradiction).
    """
    # pair: neutral (won't matter)
    pair_neutral = mk_bidir(
        mk_dir(0.10, 0.82, 0.08),
        mk_dir(0.12, 0.80, 0.08),
    )
    # thesis: strong contradiction (agg contradiction >= 0.80)
    thesis_contra = mk_bidir(
        mk_dir(0.10, 0.10, 0.82),  # p→h
        mk_dir(0.10, 0.12, 0.85),  # h→p
    )

    nli = FakeNLI([pair_neutral, thesis_contra])
    svc = ConcessionService(llm=FakeLLM(), nli=nli)

    conv = [
        {
            'role': 'assistant',
            'content': 'the assistant presents a clear argument with solid evidence and several complete sentences to validate length',
        },
        {
            'role': 'user',
            # Spanish, clearly opposing the English thesis; long enough to pass length checks
            'content': (
                'No es cierto que el trabajo remoto sea más productivo que el trabajo en oficina; '
                'las distracciones domésticas y la falta de supervisión reducen el rendimiento.'
            ),
        },
    ]

    out = svc.judge_last_two_messages(
        conv, Stance.PRO, topic='Remote work is more productive than office work'
    )
    assert out['alignment'] == 'OPPOSITE'
    assert out['concession'] is True
    assert out['reason'] == 'thesis_opposition_soft'


def test_multilingual_thesis_contradiction_english_user():
    pair_neutral = mk_bidir(
        mk_dir(0.10, 0.82, 0.08),
        mk_dir(0.12, 0.80, 0.08),
    )
    thesis_contra = mk_bidir(
        mk_dir(0.10, 0.11, 0.83),
        mk_dir(0.10, 0.12, 0.86),
    )
    nli = FakeNLI([pair_neutral, thesis_contra])
    svc = ConcessionService(llm=FakeLLM(), nli=nli)

    conv = [
        {
            'role': 'assistant',
            'content': 'el asistente presenta un argumento claro con evidencia y varias oraciones completas para validar la longitud requerida',
        },
        {
            'role': 'user',
            'content': (
                'It is not true that social media has improved human connection; '
                'it encourages superficial interactions and weakens trust.'
            ),
        },
    ]

    out = svc.judge_last_two_messages(
        conv, Stance.PRO, topic='Las redes sociales han mejorado la conexión humana'
    )
    assert out['alignment'] == 'OPPOSITE'
    assert out['concession'] is True
    assert out['reason'] == 'thesis_opposition_soft'


def test_sentence_splitting_and_max_contra():
    """
    Ensure that text is split into individual sentences and the strongest
    contradiction is detected even if only one sentence is clearly opposing.
    """
    # First sentence: neutral
    sent1 = mk_dir(0.20, 0.70, 0.10)
    # Second sentence: strong contradiction
    sent2 = mk_dir(0.10, 0.05, 0.85)

    # Pack into bidirectional fake results for each sentence
    # We'll have FakeNLI pop these in order of calls
    thesis_neutral = mk_bidir(sent1, sent1)
    thesis_contra = mk_bidir(sent2, sent2)

    nli = FakeNLI([thesis_neutral, thesis_contra])
    svc = ConcessionService(llm=FakeLLM(), nli=nli)

    conv = [
        {
            'role': 'assistant',
            'content': 'The assistant gives a long claim with evidence and examples to ensure word count.',
        },
        {
            'role': 'user',
            'content': (
                'Primera oración es neutral y no contradice de manera explícita. '
                'Sin embargo, no es cierto que el universo requiera un creador; '
                'las leyes físicas pueden surgir naturalmente.'
            ),
        },
    ]

    out = svc.judge_last_two_messages(
        conv, Stance.PRO, topic='The universe requires a creator'
    )
    # Because the second sentence is strongly contradictory, service should catch it
    assert out['alignment'] == 'OPPOSITE'
    assert out['concession'] is True
    assert out['reason'] == 'thesis_opposition_soft'
    # Because the second sentence is strongly contradictory, service should catch it
    assert out['alignment'] == 'OPPOSITE'
    assert out['concession'] is True
    assert out['reason'] == 'thesis_opposition_soft'
