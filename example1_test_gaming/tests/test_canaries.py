"""Canary tests: prove the metric can still FAIL.

Why this file exists
====================
A test that cannot fail is not a test. The "before" state of this repo
proves it: with the retriever mocked to return the answer key, the
faithfulness test passed whether or not the bot was broken.

The way we caught that was a manual **mutation check**: deliberately feed
the pipeline a known-bad answer and confirm the test goes red. It stayed
green, which exposed the problem. This file automates that check
permanently, so it runs on every PR:

  * NEGATIVE canaries: known hallucinated answers MUST score below the
    threshold. If one passes, the metric or harness is broken (gamed,
    mocked out, threshold lowered, judge misconfigured...).
  * POSITIVE canaries: known-good answers MUST pass. Without these, a
    metric that always returns 0 would "pass" the negative canaries.

Together they pin the metric from both sides. This idea comes from
*mutation testing* in classic software testing (tools like mutmut and
PIT inject bugs and check that your suite notices). For LLM evals it
matters even more, because the judge is itself a model and can drift or
be swapped out.

Live-mode note: with EVAL_MODE=live these run against DeepEval's
LLM-judged FaithfulnessMetric too. That is a cheap, valuable smoke test
of your judge: if GPT/Claude-as-judge starts approving "30 days" when
the context says "14 days", you want to know before trusting its scores
on anything else.
"""

from __future__ import annotations

import pytest
from deepeval.test_case import LLMTestCase

from evals.metrics import make_faithfulness_metric
from tests.test_faithfulness import FAITHFULNESS_MIN

REFUND_CONTEXT = [
    "Customers on monthly plans may request a full refund within 14 days of their most recent charge. "
    "Refunds are issued to the original payment method."
]

# Each negative canary is a realistic failure, not random noise. Realistic
# canaries test the metric where it matters: subtle, confident, wrong.
NEGATIVE_CANARIES = {
    "wrong-number": "Customers on monthly plans may request a full refund within 30 days of their most recent charge.",
    "parametric-prior": "Refunds are available within 30 days of purchase.",
    "invented-policy": "Refunds are issued as store credit valid for 12 months.",
}

POSITIVE_CANARIES = {
    "verbatim": "Customers on monthly plans may request a full refund within 14 days of their most recent charge.",
    "two-supported-claims": (
        "Customers on monthly plans may request a full refund within 14 days of their most recent charge. "
        "Refunds are issued to the original payment method."
    ),
}


def _score(answer: str) -> float:
    metric = make_faithfulness_metric(FAITHFULNESS_MIN)
    metric.measure(LLMTestCase(input="refund policy?", actual_output=answer, retrieval_context=REFUND_CONTEXT))
    return metric.score


@pytest.mark.parametrize("name", NEGATIVE_CANARIES)
def test_hallucinated_answer_fails_metric(name):
    score = _score(NEGATIVE_CANARIES[name])
    assert score < FAITHFULNESS_MIN, (
        f"Canary {name!r} scored {score:.2f} >= {FAITHFULNESS_MIN}. The faithfulness metric "
        "is no longer able to detect a hallucination. Do not trust any green eval until this is fixed."
    )


@pytest.mark.parametrize("name", POSITIVE_CANARIES)
def test_grounded_answer_passes_metric(name):
    assert _score(POSITIVE_CANARIES[name]) >= FAITHFULNESS_MIN


def test_mixed_answer_is_penalised_proportionally():
    """One supported + one invented claim should land at 0.5 for the lexical
    metric. This guards the scoring formula itself (supported / total)."""
    answer = POSITIVE_CANARIES["verbatim"] + " " + NEGATIVE_CANARIES["invented-policy"]
    assert _score(answer) == pytest.approx(0.5)
