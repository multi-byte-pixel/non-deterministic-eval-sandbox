from release_gate.gate import pass_rate, should_ship
from release_gate.results import load


def test_candidate_improves_on_baseline():
    assert pass_rate(load("candidate_results.jsonl")) > pass_rate(load("baseline_results.jsonl"))


def test_candidate_passes_release_gate():
    assert should_ship(load("candidate_results.jsonl"))
