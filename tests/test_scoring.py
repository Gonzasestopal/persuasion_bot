# tests/test_scoring_three_exchanges.py
import pytest

from app.services.scoring import (
    RunningScores,
    build_context_footer,
    build_score_footer,
    deterministic_verdict_from_eval,
    join_footers,
    judge_last_two_messages,
)


# --- Tiny deterministic fake NLI provider ---
class FakeNLI:
    """
    A minimal stand-in for HFNLIProvider.
    We key outcomes off the *premise* (first arg, i.e., user_text in your code).
    Markers in the user_text:
      - "OPPOSE"  -> strong contradiction
      - "SUPPORT" -> strong entailment
      - otherwise -> mostly neutral (weak signal)
    """

    def score(self, premise: str, hypothesis: str):
        p = (premise or '').upper()
        if 'OPPOSE' in p:
            # strong contradiction, clean margin
            return {'entailment': 0.06, 'neutral': 0.12, 'contradiction': 0.82}
        if 'SUPPORT' in p:
            # strong entailment, clean margin
            return {'entailment': 0.83, 'neutral': 0.10, 'contradiction': 0.07}
        # underdetermined
        return {'entailment': 0.40, 'neutral': 0.45, 'contradiction': 0.15}


# --- Test data helpers ---
ENT_THR = 0.65
CON_THR = 0.70
TOPIC = 'Dogs are the best human companion'
STANCE = 'PRO'  # bot defends the topic


def mapped_pair(bot_text: str, user_text: str):
    """Map your Message list to the {role, content} list used by scoring helpers."""
    return [
        {'role': 'assistant', 'content': bot_text},
        {'role': 'user', 'content': user_text},
    ]


@pytest.mark.asyncio
async def test_three_exchanges_running_scores_and_footers():
    nli = FakeNLI()
    rs = RunningScores()

    # --- Exchange 1: user OPPOSES strongly -> expect OPPOSITE (concession True) ---
    conv = mapped_pair(
        bot_text=(
            'Dogs provide unique loyalty, therapy benefits, and safety assistance, '
            'and this makes them unparalleled companions for humans in diverse contexts.'
        ),
        user_text='I OPPOSE this claim because working adults face costs and constraints.',
    )

    last_eval = judge_last_two_messages(
        conv,
        stance=STANCE,
        topic=TOPIC,
        nli=nli,
        entailment_threshold=ENT_THR,
        contradiction_threshold=CON_THR,
    )
    assert last_eval is not None

    verdict = deterministic_verdict_from_eval(
        last_eval,
        entailment_threshold=ENT_THR,
        contradiction_threshold=CON_THR,
    )
    assert verdict['alignment'] == 'OPPOSITE'
    assert verdict['concession'] is True

    rs.update(
        align=verdict['alignment'],
        ts=last_eval['thesis_scores'],
        ps=last_eval['scores'],
    )

    footer1 = join_footers(
        build_context_footer(
            {
                'alignment': verdict['alignment'],
                'concession': verdict['concession'],
                'reason': verdict['reason'],
                'scores': last_eval['scores'],
                'thesis_scores': last_eval['thesis_scores'],
                'topic': TOPIC,
            }
        ),
        build_score_footer(rs),
    )
    assert '[Context]' in footer1
    assert '[Score]' in footer1
    assert 'opp=1' in footer1 and 'turns=1' in footer1

    # --- Exchange 2: user SUPPORTS strongly -> expect SAME (no concession) ---
    conv = mapped_pair(
        bot_text=(
            'Dogs improve social connection, reduce loneliness, and motivate physical activity; '
            'these outcomes translate into better mental health metrics across age groups.'
        ),
        user_text='I SUPPORT this argument with evidence on therapy dogs improving outcomes.',
    )

    last_eval = judge_last_two_messages(
        conv,
        stance=STANCE,
        topic=TOPIC,
        nli=nli,
        entailment_threshold=ENT_THR,
        contradiction_threshold=CON_THR,
    )
    verdict = deterministic_verdict_from_eval(
        last_eval,
        entailment_threshold=ENT_THR,
        contradiction_threshold=CON_THR,
    )
    assert verdict['alignment'] == 'SAME'
    assert verdict['concession'] is False

    rs.update(
        align=verdict['alignment'],
        ts=last_eval['thesis_scores'],
        ps=last_eval['scores'],
    )

    footer2 = join_footers(
        build_context_footer(
            {
                'alignment': verdict['alignment'],
                'concession': verdict['concession'],
                'reason': verdict['reason'],
                'scores': last_eval['scores'],
                'thesis_scores': last_eval['thesis_scores'],
                'topic': TOPIC,
            }
        ),
        build_score_footer(rs),
    )
    assert 'same=1' in footer2
    assert 'turns=2' in footer2

    # --- Exchange 3: underdetermined (nearly neutral) -> expect UNKNOWN ---
    conv = mapped_pair(
        bot_text=(
            'Dogs may require time investment, but training and community programs can mitigate the burden, '
            'preserving benefits such as companionship and routine formation.'
        ),
        user_text='This point seems nuanced; there are pros and cons and it depends a lot.',
    )

    last_eval = judge_last_two_messages(
        conv,
        stance=STANCE,
        topic=TOPIC,
        nli=nli,
        entailment_threshold=ENT_THR,
        contradiction_threshold=CON_THR,
    )
    verdict = deterministic_verdict_from_eval(
        last_eval,
        entailment_threshold=ENT_THR,
        contradiction_threshold=CON_THR,
    )
    assert verdict['alignment'] == 'UNKNOWN'
    assert verdict['concession'] is False

    rs.update(
        align=verdict['alignment'],
        ts=last_eval['thesis_scores'],
        ps=last_eval['scores'],
    )

    footer3 = join_footers(
        build_context_footer(
            {
                'alignment': verdict['alignment'],
                'concession': verdict['concession'],
                'reason': verdict['reason'],
                'scores': last_eval['scores'],
                'thesis_scores': last_eval['thesis_scores'],
                'topic': TOPIC,
            }
        ),
        build_score_footer(rs),
    )

    # --- Final assertions on counters and footer content ---
    assert 'turns=3' in footer3
    assert 'opp=1' in footer3
    assert 'same=1' in footer3
    assert 'unk=1' in footer3

    # EMA sanity checks: after 3 updates, EMAs should be bounded [0,1] and nonnegative
    for val in (rs.tE_ema, rs.tC_ema, rs.pE_ema, rs.pC_ema):
        assert 0.0 <= val <= 1.0

    # Because first two exchanges have strong signals, EMAs should move away from 0
    # We don't pin exact numbers, just ensure they are not ~0.0
    assert rs.tC_ema > 0.15  # we saw a strong contradiction once
    assert rs.tE_ema > 0.15  # we saw a strong entailment once

    # Footers contain both sections
    assert '[Context]' in footer3 and '[Score]' in footer3
