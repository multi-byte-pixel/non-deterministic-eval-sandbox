from judge_gate.data import load_pairs, quality_index
from judge_gate.fake_judge import PairwiseJudge
from judge_gate.gate import should_ship, win_rate


def test_win_rate_is_a_fraction():
    pairs = load_pairs()
    judge = PairwiseJudge(quality_index(pairs))
    assert 0.0 <= win_rate(pairs, judge) <= 1.0


def test_release_candidate_passes_gate():
    pairs = load_pairs()
    judge = PairwiseJudge(quality_index(pairs))
    assert should_ship(pairs, judge)
