# tests/test_scoring_core.py
import json

import pytest

from app.services.scoring import (
    RunningScores,
    alignment_and_scores_topic_aware,
    bot_thesis,
    build_context_signal,
    build_score_signal,
    deterministic_verdict_from_eval,
    drop_questions,
    features_from_last_eval,
    judge_last_two_messages,
    latest_idx,
    latest_valid_assistant_before,
    make_scoring_system_message,
    nli_confident,
)


# ----------------------------
# Fake NLI with deterministic behavior
# ----------------------------
class _FakeNLI:
    """
    Scores depend ONLY on the premise (first arg).
      - contains 'OPPOSE'   -> strong contradiction (clean margin)
      - contains 'SUPPORT'  -> strong entailment   (clean margin)
      - otherwise           -> underdetermined / mild neutral
    """

    def score(self, premise: str, hypothesis: str):
        p = (premise or '').upper()
        if 'OPPOSE' in p:
            return {'entailment': 0.06, 'neutral': 0.12, 'contradiction': 0.82}
        if 'SUPPORT' in p:
            return {'entailment': 0.83, 'neutral': 0.10, 'contradiction': 0.07}
        return {'entailment': 0.40, 'neutral': 0.45, 'contradiction': 0.15}


ENT_THR = 0.65
CON_THR = 0.70


# ----------------------------
# Small, focused unit tests
# ----------------------------
def test_drop_questions_removes_interrogatives_and_keeps_statements():
    txt = 'Dogs are great. Are cats better? Maybe. What do you think?'
    out = drop_questions(txt)
    # remove sentences ending with '?'
    assert 'Are cats better?' not in out
    assert 'What do you think?' not in out
    assert 'Dogs are great.' in out and 'Maybe.' in out


@pytest.mark.parametrize(
    'topic,stance,expected',
    [
        ('Dogs are best', 'PRO', 'Dogs are best.'),
        ('Dogs are best.', 'PRO', 'Dogs are best.'),
        ('Dogs are best', 'CON', 'It is not true that Dogs are best.'),
    ],
)
def test_bot_thesis(topic, stance, expected):
    assert bot_thesis(topic, stance) == expected


@pytest.mark.parametrize(
    'scores,pmin,margin,ok',
    [
        ({'entailment': 0.8, 'neutral': 0.1, 'contradiction': 0.1}, 0.75, 0.15, True),
        (
            {'entailment': 0.76, 'neutral': 0.72},
            0.75,
            0.05,
            True,
        ),  # top >= pmin and gap >= margin
        ({'entailment': 0.74, 'neutral': 0.2}, 0.75, 0.15, False),  # below pmin
        ({'entailment': 0.8}, 0.75, 0.15, True),  # single value case
        ({}, 0.75, 0.15, False),
    ],
)
def test_nli_confident(scores, pmin, margin, ok):
    assert nli_confident(scores, pmin=pmin, margin=margin) == ok


def test_latest_idx_and_latest_valid_assistant_before():
    conv = [
        {'role': 'assistant', 'content': 'too short'},  # not enough alpha words
        {'role': 'user', 'content': 'Hi'},
        {
            'role': 'assistant',
            'content': 'This has many alpha words indeed for threshold pass okay now good enough',
        },
        {'role': 'user', 'content': 'Final user'},
    ]
    # latest user
    assert latest_idx(conv, 'user') == 3
    # latest valid assistant BEFORE last user (needs >=10 alpha words)
    idx = latest_valid_assistant_before(conv, 3, min_words=10)
    assert idx == 2


def test_alignment_and_scores_topic_aware_paths():
    nli = _FakeNLI()
    topic = 'Dogs are the best human companion'

    # Case 1: OPPOSE → OPPOSITE by thesis contradiction
    a1, pair1, th1 = alignment_and_scores_topic_aware(
        nli,
        bot_text='Dogs help humans in therapy and safety.',  # cleaned
        user_text='I OPPOSE this strongly.',
        bot_stance='PRO',
        topic=topic,
        entailment_threshold=ENT_THR,
        contradiction_threshold=CON_THR,
    )
    assert a1 == 'OPPOSITE'
    assert th1['contradiction'] > th1['entailment']

    # Case 2: SUPPORT → SAME by thesis entailment
    a2, pair2, th2 = alignment_and_scores_topic_aware(
        nli,
        bot_text='Dogs help humans in therapy and safety?',
        user_text='I SUPPORT this claim.',
        bot_stance='PRO',
        topic=topic,
        entailment_threshold=ENT_THR,
        contradiction_threshold=CON_THR,
    )
    assert a2 == 'SAME'
    assert th2['entailment'] > th2['contradiction']

    # Case 3: neutral → UNKNOWN
    a3, pair3, th3 = alignment_and_scores_topic_aware(
        nli,
        bot_text='Dogs are helpful. They reduce loneliness.',
        user_text='This is nuanced and depends.',
        bot_stance='PRO',
        topic=topic,
        entailment_threshold=ENT_THR,
        contradiction_threshold=CON_THR,
    )
    assert a3 == 'UNKNOWN'


@pytest.mark.asyncio
async def test_judge_last_two_messages_and_features_and_deterministic_verdict():
    nli = _FakeNLI()
    topic = 'Dogs are the best human companion'
    stance = 'PRO'

    # Build a conversation with a valid assistant turn (>= 10 alpha words)
    bot_text = (
        'Dogs provide loyalty therapy benefits safety assistance and structure for daily routine '
        'which improves mental health outcomes consistently'
    )
    conv = [
        {'role': 'assistant', 'content': 'short'},  # invalid due to few alpha words
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': bot_text},
        {
            'role': 'user',
            'content': 'I OPPOSE this strongly due to costs and constraints.',
        },
    ]

    ev = judge_last_two_messages(
        conv,
        stance=stance,
        topic=topic,
        nli=nli,
        entailment_threshold=ENT_THR,
        contradiction_threshold=CON_THR,
    )
    assert ev is not None
    assert ev['topic'] == topic
    assert ev['passed_stance'] == stance
    assert 'scores' in ev and 'thesis_scores' in ev
    assert ev['user_text_sample'].startswith('I OPPOSE')

    # features
    feats = features_from_last_eval(
        ev, stance=stance, entailment_threshold=ENT_THR, contradiction_threshold=CON_THR
    )
    # minimal shape checks
    for k in (
        'entailment_threshold',
        'contradiction_threshold',
        'thesis_entailment',
        'thesis_contradiction',
        'pair_entailment',
        'pair_contradiction',
        'pair_confident',
        'thesis_confident',
        'stance',
        'user_len',
    ):
        assert k in feats

    # deterministic verdict: OPPOSE in user's text -> thesis_opposition branch
    verd1 = deterministic_verdict_from_eval(
        ev, entailment_threshold=ENT_THR, contradiction_threshold=CON_THR
    )
    assert verd1['alignment'] == 'OPPOSITE'
    assert verd1['concession'] is True
    assert verd1['reason'] in (
        'thesis_opposition',
        'pairwise_opposition',
    )  # prefer thesis
    assert 0.5 <= verd1['confidence'] <= 1.0

    # Now SAME branch: user SUPPORT
    conv[-1] = {
        'role': 'user',
        'content': 'I SUPPORT this because therapy programs show gains.',
    }
    ev2 = judge_last_two_messages(
        conv,
        stance=stance,
        topic=topic,
        nli=nli,
        entailment_threshold=ENT_THR,
        contradiction_threshold=CON_THR,
    )
    verd2 = deterministic_verdict_from_eval(
        ev2, entailment_threshold=ENT_THR, contradiction_threshold=CON_THR
    )
    assert verd2['alignment'] == 'SAME'
    assert verd2['concession'] is False
    assert verd2['reason'] == 'same_stance'

    # Pairwise opposition branch:
    # Make thesis NOT confident (neutral default), but pair contradiction strong & confident.
    conv[-1] = {'role': 'user', 'content': 'I OPPOSE your framing on pairwise only.'}
    ev3 = judge_last_two_messages(
        conv,
        stance=stance,
        topic=topic,
        nli=nli,
        entailment_threshold=ENT_THR,
        contradiction_threshold=CON_THR,
    )
    # To force pairwise branch we simulate thesis not confident by replacing thesis scores in ev3
    # However with our Fake, thesis already shows OPPOSE -> confident. So emulate neutral thesis:
    ev3_mut = dict(ev3)
    ev3_mut['thesis_scores'] = {
        'entailment': 0.4,
        'neutral': 0.45,
        'contradiction': 0.15,
    }
    verd3 = deterministic_verdict_from_eval(
        ev3_mut, entailment_threshold=ENT_THR, contradiction_threshold=CON_THR
    )
    assert verd3['alignment'] == 'OPPOSITE'
    assert verd3['concession'] is True
    assert verd3['reason'] == 'pairwise_opposition'

    # Unknown branch: neutral text
    conv[-1] = {'role': 'user', 'content': 'This feels nuanced and context dependent.'}
    ev4 = judge_last_two_messages(
        conv,
        stance=stance,
        topic=topic,
        nli=nli,
        entailment_threshold=ENT_THR,
        contradiction_threshold=CON_THR,
    )
    verd4 = deterministic_verdict_from_eval(
        ev4, entailment_threshold=ENT_THR, contradiction_threshold=CON_THR
    )
    assert verd4['alignment'] == 'UNKNOWN'
    assert verd4['concession'] is False
    assert verd4['reason'] == 'underdetermined'


def test_running_scores_update_counters_and_emas():
    rs = RunningScores()
    # first update: opposite with strong contradictions and entailments
    rs.update(
        align='OPPOSITE',
        ts={'entailment': 0.1, 'contradiction': 0.9},
        ps={'entailment': 0.2, 'contradiction': 0.8},
        alpha=0.3,
    )
    assert (rs.turns, rs.opp, rs.same, rs.unk) == (1, 1, 0, 0)
    for val in (rs.tE_ema, rs.tC_ema, rs.pE_ema, rs.pC_ema):
        assert 0.0 <= val <= 1.0

    # second update: same
    prev = (rs.tE_ema, rs.tC_ema, rs.pE_ema, rs.pC_ema)
    rs.update(
        align='SAME',
        ts={'entailment': 0.8, 'contradiction': 0.1},
        ps={'entailment': 0.6, 'contradiction': 0.2},
        alpha=0.3,
    )
    assert (rs.turns, rs.opp, rs.same, rs.unk) == (2, 1, 1, 0)
    # EMAs should move toward new values
    assert rs.tE_ema >= prev[0]
    assert rs.tC_ema <= prev[1]

    # third update: unknown
    rs.update(
        align='UNKNOWN',
        ts={'entailment': 0.5, 'contradiction': 0.5},
        ps={'entailment': 0.5, 'contradiction': 0.5},
        alpha=0.3,
    )
    assert (rs.turns, rs.opp, rs.same, rs.unk) == (3, 1, 1, 1)
    for val in (rs.tE_ema, rs.tC_ema, rs.pE_ema, rs.pC_ema):
        assert 0.0 <= val <= 1.0


def test_build_context_and_score_signal_and_system_message():
    # Context from eval dict (simulate downstream enrichment)
    ev = {
        'alignment': 'OPPOSITE',
        'concession': True,
        'reason': 'thesis_opposition',
        'thesis_scores': {'entailment': 0.08, 'contradiction': 0.85},
        'scores': {'entailment': 0.12, 'contradiction': 0.80},
        'topic': 'Dogs "are" best\nindeed',
    }
    ctx = build_context_signal(ev)
    assert ctx is not None
    ctxd = ctx.to_dict()
    # topic cleaned (no newlines, double quotes replaced)
    assert '"' not in ctxd['topic'] and '\n' not in ctxd['topic']
    assert ctxd['align'] == 'OPPOSITE'
    assert ctxd['concession'] is True
    assert 0.0 <= ctxd['tE'] <= 1.0 and 0.0 <= ctxd['tC'] <= 1.0

    rs = RunningScores(
        turns=3, opp=1, same=1, unk=1, tE_ema=0.2, tC_ema=0.4, pE_ema=0.3, pC_ema=0.5
    )
    agg = build_score_signal(rs)
    aggd = agg.to_dict()
    assert (
        aggd['turns'] == 3
        and aggd['opp'] == 1
        and aggd['same'] == 1
        and aggd['unk'] == 1
    )

    sysmsg = make_scoring_system_message(ctx, agg)
    assert (
        sysmsg is not None
        and sysmsg.startswith('<SCORING>')
        and sysmsg.endswith('</SCORING>')
    )
    # parse inner JSON
    inner = sysmsg[len('<SCORING>') : -len('</SCORING>')]
    data = json.loads(inner)
    assert 'context' in data and 'score' in data
    # quick sanity
    assert data['context']['align'] == 'OPPOSITE'
    assert data['score']['turns'] == 3
