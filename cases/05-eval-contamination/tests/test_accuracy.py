from pathlib import Path

import yaml

from fewshot.evaluate import accuracy

THRESHOLDS = yaml.safe_load((Path(__file__).parent.parent / "evals" / "thresholds.yaml").read_text())


def test_intent_accuracy_meets_target():
    assert accuracy() >= THRESHOLDS["accuracy"]
