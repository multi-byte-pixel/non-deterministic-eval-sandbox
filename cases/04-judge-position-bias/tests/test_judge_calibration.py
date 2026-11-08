"""Measure the judge before trusting it.

The calibration set has human-labelled pairs with clear quality gaps
(0.1 to 0.35). A usable judge must agree with humans and must not change
its verdict when the answers swap positions.
"""
import pytest

from judge_gate.data import Pair, load_pairs, quality_index
from judge_gate.fake_judge import PairwiseJudge
from judge_gate.gate import judge_both_orders

CAL = load_pairs("judge_calibration.jsonl")


def _stats(profile):
    judge = PairwiseJudge.from_profile(quality_index(CAL), profile)
    outcomes = [judge_both_orders(p, judge) for p in CAL]
    consistency = sum(o != "tie" for o in outcomes) / len(CAL)
    agree = sum(
        (o == "win") == (p.human_quality["candidate"] > p.human_quality["baseline"])
        for o, p in zip(outcomes, CAL) if o != "tie"
    ) / max(1, sum(o != "tie" for o in outcomes))
    return consistency, agree


def test_v2_judge_is_position_consistent_and_agrees_with_humans():
    consistency, agree = _stats("pairwise-v2-rubric")
    assert consistency >= 0.9
    assert agree >= 0.95


def test_v1_judge_is_measurably_position_biased():
    consistency, _ = _stats("pairwise-v1")
    assert consistency < 0.8


@pytest.mark.parametrize("profile", ["pairwise-v1", "pairwise-v2-rubric"])
def test_identical_quality_answers_are_a_tie_not_a_win(profile):
    pair = Pair("canary", "q", "answer one", "answer two", {"baseline": 0.6, "candidate": 0.6})
    judge = PairwiseJudge.from_profile(quality_index([pair]), profile)
    assert judge_both_orders(pair, judge) == "tie"
