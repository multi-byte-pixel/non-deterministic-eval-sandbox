"""The model in config.yaml is the one being considered for release.

It is evaluated with a fresh cache, so the result reflects real calls.
"""
from harness.cache import ResponseCache
from harness.runner import MIN_ACCURACY, configured_model, run_eval


def test_configured_model_is_actually_called(tmp_path):
    result = run_eval(configured_model(), ResponseCache(tmp_path / "c.json"))
    assert result.model_calls > 0


def test_configured_model_meets_threshold(tmp_path):
    result = run_eval(configured_model(), ResponseCache(tmp_path / "c.json"))
    assert result.accuracy >= MIN_ACCURACY, f"{result.model_id} scored {result.accuracy:.2f}"
