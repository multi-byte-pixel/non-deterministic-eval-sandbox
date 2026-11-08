from judge_gate.data import Pair, load_pairs, quality_index
from judge_gate.fake_judge import PairwiseJudge
from judge_gate.gate import evaluate, judge_both_orders


def _judge(pairs, profile):
    return PairwiseJudge.from_profile(quality_index(pairs), profile)


def test_every_pair_is_judged_in_both_orders():
    pairs = load_pairs()
    judge = _judge(pairs, "pairwise-v2-rubric")
    evaluate(pairs, judge)
    assert judge.calls == 2 * len(pairs)


def test_v1_judge_fails_closed_on_release_pairs():
    pairs = load_pairs()
    decision = evaluate(pairs, _judge(pairs, "pairwise-v1"))
    assert not decision.ship
    assert "consistency" in decision.reason


def test_v2_judge_rejects_the_worse_candidate():
    pairs = load_pairs()
    decision = evaluate(pairs, _judge(pairs, "pairwise-v2-rubric"))
    humans_prefer_candidate = sum(p.human_quality["candidate"] > p.human_quality["baseline"] for p in pairs)
    assert decision.consistency >= 0.8
    assert not decision.ship
    assert decision.wins == humans_prefer_candidate


def test_gate_still_ships_a_genuinely_better_candidate():
    pairs = [
        Pair(f"x{i}", "q", f"base {i}", f"cand {i}", {"baseline": 0.5, "candidate": 0.8})
        for i in range(20)
    ]
    assert evaluate(pairs, _judge(pairs, "pairwise-v2-rubric")).ship
