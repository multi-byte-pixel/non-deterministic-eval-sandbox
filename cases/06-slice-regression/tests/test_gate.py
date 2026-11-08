from release_gate.gate import evaluate, pass_rate
from release_gate.results import load

BASE = load("baseline_results.jsonl")
CAND = load("candidate_results.jsonl")


def test_overall_rate_still_looks_better():
    # The trap: this is true, and it is not enough.
    assert pass_rate(CAND) > pass_rate(BASE)


def test_candidate_is_blocked_by_safety_slice():
    decision = evaluate(CAND, BASE)
    assert not decision.ship
    assert any(v.startswith("safety_refusal") for v in decision.violations)
    assert not any(v.startswith(("billing", "technical")) for v in decision.violations)


def test_baseline_passes_its_own_gate():
    assert evaluate(BASE, BASE).ship


def test_missing_slice_blocks_release():
    no_safety = [r for r in CAND if r["slice"] != "safety_refusal"]
    decision = evaluate(no_safety, BASE)
    assert not decision.ship
    assert "insufficient data" in decision.violations[0]


def test_uniform_improvement_ships():
    better = [dict(r, passed=True) for r in BASE]
    assert evaluate(better, BASE).ship
