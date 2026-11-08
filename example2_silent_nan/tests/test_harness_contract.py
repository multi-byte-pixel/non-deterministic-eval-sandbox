"""Contract tests: does our eval harness use its libraries correctly, and does
it refuse to report numbers it didn't really compute?

Each test here maps to one link in the incident's failure chain:

    hallucinated kwarg  -> test_every_kwarg_we_pass_exists_in_ragas_evaluate
    unpinned deps       -> test_direct_dependencies_are_pinned
    swallowed exception -> test_broken_judge_output_raises_instead_of_nan
    zero judge calls    -> test_harness_errors_when_judge_is_never_called
    NaN passes the gate -> test_gate_is_fail_closed

"Contract test" here means a test of the *agreement* between our code and
code we don't own (ragas, langchain). Those libraries change under us. The
contract tests are cheap and offline, and they run on every Dependabot
upgrade PR.
"""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

import pandas as pd
import pytest
import ragas

import evals.run_ragas_eval as harness
from evals.run_ragas_eval import EvalHarnessError, evaluate_kwargs, gate, run_eval
from tests.fakes import GarbageJudge, ScriptedJudge

SAMPLE = {
    "user_input": "When are production deploys frozen?",
    "response": "From 5 PM Friday until 9 AM Monday.",
    "retrieved_contexts": ["Deploys to production are frozen from 5 PM Friday until 9 AM Monday."],
    "reference": "From 5 PM Friday until 9 AM Monday.",
}


# ---------------------------------------------------------------------------
# 1. API contract: no hallucinated parameters
# ---------------------------------------------------------------------------
def test_every_kwarg_we_pass_exists_in_ragas_evaluate():
    """Would have caught ``llm_judge=`` instantly.

    ``inspect.signature`` reads the REAL installed function's parameters.
    LLM coding agents often produce kwargs that sound right but don't exist,
    or that existed in an older version they saw in training data. Checking
    against the signature catches both, offline, in milliseconds.

    (A TypeError would also be raised at runtime, but only if nothing
    swallows it. This test does not depend on the error-handling path being
    correct.)
    """
    params = inspect.signature(ragas.evaluate).parameters
    accepts_var_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    unknown = [k for k in evaluate_kwargs(ScriptedJudge(), []) if k not in params]
    assert not unknown or accepts_var_kwargs, f"ragas.evaluate() has no parameter(s) {unknown}"


def test_direct_dependencies_are_pinned():
    """Every requirement uses ``==``. An unpinned ragas + langchain-community
    is what broke the import in the first place."""
    lines = [
        ln.strip()
        for ln in (Path(__file__).resolve().parent.parent / "requirements.txt").read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    unpinned = [ln for ln in lines if not re.match(r"^[A-Za-z0-9_.\-\[\]]+==[^=]+$", ln)]
    assert not unpinned, f"unpinned requirements: {unpinned}"


# ---------------------------------------------------------------------------
# 2. Harness honesty: errors are errors, never scores
# ---------------------------------------------------------------------------
def test_broken_judge_output_raises_instead_of_nan():
    """With Ragas' default raise_exceptions=False, this judge would yield
    {'faithfulness': nan}. We require an exception instead."""
    with pytest.raises(Exception) as excinfo:
        run_eval([SAMPLE], GarbageJudge())
    # Whatever the exact type (RagasOutputParserException today), the run did
    # not quietly produce a number.
    assert not isinstance(excinfo.value, AssertionError)


def test_harness_errors_when_judge_is_never_called(monkeypatch):
    """Reproduce the incident's signature: scores come back, judge calls == 0.

    We replace ``ragas_evaluate`` with a stub that returns a plausible
    result without ever calling the judge, like the old code path did when
    it never reached Ragas at all. The harness must notice.

    ``monkeypatch`` is pytest's built-in fixture for temporarily replacing
    attributes. It undoes the change automatically after the test.
    """

    class FakeResult:
        def to_pandas(self):
            return pd.DataFrame([{"faithfulness": 1.0}])

    monkeypatch.setattr(harness, "ragas_evaluate", lambda dataset, **kwargs: FakeResult())
    with pytest.raises(EvalHarnessError, match="called 0 time"):
        run_eval([SAMPLE], ScriptedJudge())


def test_harness_rejects_nan_per_sample_scores(monkeypatch):
    """Even if the judge WAS called, a NaN row means that sample wasn't scored."""

    class FakeResult:
        def to_pandas(self):
            return pd.DataFrame([{"faithfulness": float("nan")}])

    def fake_evaluate(dataset, **kwargs):
        for cb in kwargs["callbacks"]:  # pretend the judge ran twice
            if isinstance(cb, harness.JudgeCallCounter):
                cb.calls += 2
        return FakeResult()

    monkeypatch.setattr(harness, "ragas_evaluate", fake_evaluate)
    with pytest.raises(EvalHarnessError, match="Non-finite"):
        run_eval([SAMPLE], ScriptedJudge())


def test_empty_eval_set_is_an_error_not_a_pass():
    with pytest.raises(EvalHarnessError, match="No samples"):
        run_eval([], ScriptedJudge())


def test_judge_call_count_is_observed_through_callbacks():
    """The counter sees the same calls LangSmith would trace."""
    judge = ScriptedJudge()
    result = run_eval([SAMPLE, SAMPLE], judge)
    assert result.judge_calls == judge.calls == 4  # 2 samples x 2 Faithfulness prompts


# ---------------------------------------------------------------------------
# 3. The gate fails closed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "scores, expected_fragment",
    [
        ({"faithfulness": float("nan")}, "not a finite number"),
        ({"faithfulness": float("inf")}, "not a finite number"),
        ({"faithfulness": None}, "not a finite number"),
        ({}, "missing"),
        ({"faithfulness": 0.79}, "0.790 < 0.8"),
    ],
    ids=["nan", "inf", "none", "missing", "below-threshold"],
)
def test_gate_is_fail_closed(scores, expected_fragment):
    failures = gate(scores, {"faithfulness": 0.8})
    assert failures and expected_fragment in failures[0]


def test_gate_passes_exactly_at_threshold():
    assert gate({"faithfulness": 0.8}, {"faithfulness": 0.8}) == []


def test_nan_comparison_semantics():
    """Executable documentation of the root cause. Every comparison with NaN is False."""
    nan = float("nan")
    assert not (nan < 0.8) and not (nan >= 0.8) and not (nan == nan)
    assert math.isnan(nan)
