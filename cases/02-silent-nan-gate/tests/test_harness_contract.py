"""Contract tests: does our eval harness use its libraries correctly, and does
it refuse to report numbers it didn't really compute?

Each test here maps to one link in the incident's failure chain:

    hallucinated kwarg  -> test_every_kwarg_we_pass_exists_in_the_scorer
    unpinned deps       -> test_direct_dependencies_are_pinned
    swallowed exception -> test_broken_judge_output_raises_instead_of_nan
    zero judge calls    -> test_harness_errors_when_judge_is_never_called
    NaN passes the gate -> test_gate_is_fail_closed

"Contract test" here means a test of the *agreement* between our code and
code it depends on (the scorer, langchain). Those change under us. The
contract tests are cheap and offline, and they run on every Dependabot
upgrade PR.
"""

from __future__ import annotations

import inspect
import math
import re
from pathlib import Path

import pytest

import evals.run_faithfulness_eval as harness
from evals.faithfulness import JudgeOutputError, _parse_json, score_samples
from evals.run_faithfulness_eval import EvalHarnessError, gate, run_eval, scorer_kwargs
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
def test_every_kwarg_we_pass_exists_in_the_scorer():
    """Would have caught ``llm_judge=`` instantly.

    ``inspect.signature`` reads the REAL function's parameters. LLM coding
    agents often produce kwargs that sound right but don't exist, or that
    existed in an older version they saw in training data. Checking against
    the signature catches both, offline, in milliseconds.

    (A TypeError would also be raised at runtime, but only if nothing
    swallows it. This test does not depend on the error-handling path being
    correct.)
    """
    params = inspect.signature(score_samples).parameters
    accepts_var_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    unknown = [k for k in scorer_kwargs(ScriptedJudge(), []) if k not in params]
    assert not unknown and not accepts_var_kwargs, f"score_samples() has no parameter(s) {unknown}"


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
    """A judge that replies with prose must produce an error, not {'faithfulness': nan}."""
    with pytest.raises(EvalHarnessError, match="could not be scored") as excinfo:
        run_eval([SAMPLE], GarbageJudge())
    assert isinstance(excinfo.value.__cause__, JudgeOutputError)


def test_harness_errors_when_judge_is_never_called(monkeypatch):
    """Reproduce the incident's signature: scores come back, judge calls == 0.

    We replace the scorer with a stub that returns a plausible result without
    ever calling the judge, like the old code path did when it never reached
    the metric at all. The harness must notice.

    ``monkeypatch`` is pytest's built-in fixture for temporarily replacing
    attributes. It undoes the change automatically after the test.
    """
    monkeypatch.setattr(harness, "score_faithfulness", lambda samples, **kwargs: [{"faithfulness": 1.0}])
    with pytest.raises(EvalHarnessError, match="called 0 time"):
        run_eval([SAMPLE], ScriptedJudge())


def test_harness_rejects_nan_per_sample_scores(monkeypatch):
    """Even if the judge WAS called, a NaN row means that sample wasn't scored."""

    def fake_score(samples, **kwargs):
        for cb in kwargs["callbacks"]:  # pretend the judge ran twice
            if isinstance(cb, harness.JudgeCallCounter):
                cb.calls += 2
        return [{"faithfulness": float("nan")}]

    monkeypatch.setattr(harness, "score_faithfulness", fake_score)
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


@pytest.mark.parametrize(
    "reply",
    ['{"statements": []}', '{"statements": "s"}', "[1, 2]", "not json"],
    ids=["zero-statements", "not-a-list", "not-an-object", "not-json"],
)
def test_malformed_statement_output_raises(reply):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    with pytest.raises(JudgeOutputError):
        score_samples([SAMPLE], judge=FakeListChatModel(responses=[reply]), callbacks=[])


@pytest.mark.parametrize(
    "verdicts",
    ['{"statements": []}', '{"statements": [{"verdict": 2}]}', '{"statements": [{"verdict": true}]}'],
    ids=["count-mismatch", "out-of-range", "bool"],
)
def test_malformed_verdict_output_raises(verdicts):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    judge = FakeListChatModel(responses=['{"statements": ["s"]}', verdicts])
    with pytest.raises(JudgeOutputError):
        score_samples([SAMPLE], judge=judge, callbacks=[])


def test_parser_accepts_only_a_json_fence():
    assert _parse_json('```json\n{"statements": ["s"]}\n```') == {"statements": ["s"]}
    with pytest.raises(JudgeOutputError):
        _parse_json('Here you go: {"statements": ["s"]}')


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
