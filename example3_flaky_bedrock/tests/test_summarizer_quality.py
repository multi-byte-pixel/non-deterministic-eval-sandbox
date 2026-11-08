"""Quality eval for the summarizer: N runs, property checks, rubric judge, pass-rate gate.

This replaces the original test:

    assert summary == "Customer reports login failure after password reset; requests account unlock."

That test was wrong in kind, not in degree. It asserted one exact sample of
a random variable. It passed about 7 runs in 8 (so it got merged), and then
failed about 1 run in 8 in CI. Over 10 CI runs that's a ~74% chance of at
least one red build that nobody can fix, which trains the team to click
"re-run" and ignore red. That's the real cost of a flaky test.

How it was diagnosed: run the suspect test many times and measure, rather
than guess:
    pytest tests/test_summarizer.py --count=25      # pytest-repeat
or  python scripts/flake_rate.py <test-id> -n 25   # adds a confidence interval

What we assert now, per run
===========================
1. **Format**: the output parses and validates as ``TicketSummary``
   (forced tool use + pydantic). A format failure counts as a failed run.
2. **Stable fields**: ``category == "account_access"``, ``priority`` in
   {high, urgent}. For a clear-cut ticket these should not vary. If they
   do, that's a real quality problem worth a failed run.
3. **Invariants**: the summary mentions the core problem (password/login)
   and does NOT contain the customer's name (privacy).
4. **Rubric judge** score >= PASS_SCORE (see evals/judge.py).

What we gate on, across runs
============================
**Pass rate >= MIN_PASS_RATE over N_RUNS**, with a 95% Wilson interval in
the report. No exact text anywhere.

Live vs offline
===============
Offline (default), ``get_client()`` returns FakeBedrockRuntime, which is
non-deterministic on purpose. With BEDROCK_MODE=live it's real Bedrock
(the nightly workflow). Same test code, different noise source.
"""

from __future__ import annotations

import os

from app.bedrock_client import get_client
from app.fake_bedrock import JUDGE_LOW_SCORE_RATE
from app.summarizer import SummaryFormatError, summarize_ticket
from evals.judge import PASS_SCORE, JudgeError, grade_summary
from evals.stats import PassRateReport, probability_gate_fails

TICKET = (
    "Hi, I reset my password this morning and now I can't log in at all. "
    "It says my account is locked. Can you unlock it? Thanks, Dana"
)
CUSTOMER_NAME = "Dana"

# N and the threshold are chosen with probability_gate_fails (see the test
# at the bottom), not by feel. More runs = tighter estimate = more API cost.
N_RUNS = int(os.getenv("EVAL_RUNS", "30"))
MIN_PASS_RATE = float(os.getenv("EVAL_MIN_PASS_RATE", "0.9"))


def check_one_run(client) -> str | None:
    """Run the summarizer + judge once. Return None on success or a failure reason."""
    try:
        s = summarize_ticket(TICKET, client)
    except SummaryFormatError as exc:
        return f"format: {exc}"[:200]

    if s.category != "account_access":
        return f"category={s.category!r}"
    if s.priority not in {"high", "urgent"}:
        return f"priority={s.priority!r}"
    text = s.summary.lower()
    if not ("password" in text or "log in" in text or "login" in text):
        return f"summary misses the core problem: {s.summary!r}"
    if CUSTOMER_NAME.lower() in (s.summary + " ".join(s.action_items)).lower():
        return "privacy: customer name in output"

    try:
        grade = grade_summary(TICKET, s, client)
    except JudgeError as exc:
        # A broken judge is a failed sample, never a pass (see example 2).
        return f"judge error: {exc}"[:200]
    if grade.score < PASS_SCORE:
        return f"judge score {grade.score}: {grade.reasoning}"
    return None


def test_summarizer_pass_rate():
    client = get_client()
    failures = [r for r in (check_one_run(client) for _ in range(N_RUNS)) if r]
    report = PassRateReport(passed=N_RUNS - len(failures), n=N_RUNS, min_pass_rate=MIN_PASS_RATE, failures=failures)
    print("\n" + report.summary())  # visible with `pytest -s` and in CI logs on failure
    assert report.ok, report.summary()


def test_gate_itself_is_not_flaky_against_the_fake():
    """Meta-test: is our GATE flaky?

    With the fake's 1% spurious-low-judge rate, N=30 and a 90% threshold
    (up to 3 failures allowed), the chance of a false alarm is about 0.02%.
    If someone tightens the threshold to 100% "to be safe", this test fails
    and explains why: P(false alarm) would jump to about 26%, and we'd have
    rebuilt the flaky test we just removed.

    Story from writing this PR: the fake first used a 2% noise rate, where
    this formula predicts a 0.3% false-alarm rate. Running the gate 200 times
    (`pytest --count=200`) produced exactly 1 false alarm, matching the
    maths. That's still a test that goes red about once every 300 CI runs
    for no reason, so the rate was lowered. Measure your gate's flakiness;
    don't assume it.
    """
    p_false_alarm = probability_gate_fails(N_RUNS, MIN_PASS_RATE, JUDGE_LOW_SCORE_RATE)
    assert p_false_alarm < 0.01, f"gate would fail {p_false_alarm:.1%} of healthy runs"


def test_gate_can_detect_a_real_regression():
    """And the other side: if 20% of runs went bad, the gate should usually catch it.

    P(miss) is about 12% for a single run of this gate at N=30. Acceptable for a
    nightly that runs every day; raise N for a release gate.
    """
    p_catch = probability_gate_fails(N_RUNS, MIN_PASS_RATE, 0.20)
    assert p_catch > 0.85
