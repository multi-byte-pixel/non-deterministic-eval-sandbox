import pytest

from harness.cache import ResponseCache, StaleCacheError
from harness.fake_models import FakeModel
from harness.runner import MIN_ACCURACY, TEMPLATE, TEMPLATE_VERSION, run_eval


@pytest.fixture
def cache(tmp_path):
    return ResponseCache(tmp_path / "cache.json")


def test_v1_meets_threshold(cache):
    assert run_eval(FakeModel("assistant-v1"), cache).accuracy >= MIN_ACCURACY


def test_v2_regression_is_caught_even_with_a_warm_v1_cache(cache):
    run_eval(FakeModel("assistant-v1"), cache)
    result = run_eval(FakeModel("assistant-v2"), cache)
    assert result.model_calls == 10, "v2 must be called, not served v1 responses"
    assert result.accuracy == pytest.approx(0.6)
    assert result.accuracy < MIN_ACCURACY


def test_second_run_of_same_model_is_cached(cache):
    run_eval(FakeModel("assistant-v1"), cache)
    model = FakeModel("assistant-v1")
    assert run_eval(model, cache).model_calls == 0


@pytest.mark.parametrize("change", [
    {"model_id": "assistant-v2"},
    {"temperature": 0.7},
    {"template_version": "docs-qa-v2"},
])
def test_cache_key_changes_with_every_input_that_changes_output(change):
    base = dict(prompt="p", model_id="assistant-v1", temperature=0.0, template_version=TEMPLATE_VERSION)
    assert ResponseCache.key(**base) != ResponseCache.key(**{**base, **change})


def test_entry_from_another_model_is_refused(cache):
    prompt = TEMPLATE.format(question="x")
    args = dict(model_id="assistant-v2", temperature=0.0, template_version=TEMPLATE_VERSION)
    cache.data[ResponseCache.key(prompt, **args)] = {"response": "r", "model_id": "assistant-v1"}
    with pytest.raises(StaleCacheError):
        cache.get(prompt, **args)
