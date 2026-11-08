from harness.cache import ResponseCache
from harness.runner import MIN_ACCURACY, ROOT, configured_model, run_eval

CACHE = ROOT / "data" / "response_cache.json"


def test_eval_is_free_when_cache_is_warm():
    model = configured_model()
    run_eval(model, ResponseCache(CACHE))
    assert model.calls == 0


def test_configured_model_meets_accuracy_threshold():
    assert run_eval(configured_model(), ResponseCache(CACHE)) >= MIN_ACCURACY
