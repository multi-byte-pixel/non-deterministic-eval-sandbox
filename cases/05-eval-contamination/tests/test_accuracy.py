from pathlib import Path

import yaml

from fewshot.evaluate import accuracy

THRESHOLDS = yaml.safe_load((Path(__file__).parent.parent / "evals" / "thresholds.yaml").read_text())


def test_intent_accuracy_meets_threshold_on_held_out_tickets():
    assert accuracy() >= THRESHOLDS["accuracy"]


def test_accuracy_is_not_suspiciously_perfect():
    # A jump to ~1.0 on a model that scores ~0.6 on new tickets is a leak
    # until proven otherwise.
    assert accuracy() < 0.95
