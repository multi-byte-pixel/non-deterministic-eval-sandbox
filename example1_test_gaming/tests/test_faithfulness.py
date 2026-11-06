"""Faithfulness regression tests for the RAG support bot, using DeepEval.

How DeepEval tests are put together
===================================
* ``LLMTestCase`` is a plain record of one interaction: the ``input``
  (question), the ``actual_output`` (what our bot said) and the
  ``retrieval_context`` (the chunks our retriever handed the LLM).
* A *metric* (``BaseMetric`` subclass) scores a test case from 0 to 1 and
  compares the score to its ``threshold``.
* ``assert_test(test_case, [metrics...])`` runs every metric and raises
  ``AssertionError`` with a readable report if any metric fails. It works
  inside plain pytest, which is how we use it. (``deepeval test run``
  is DeepEval's own CLI wrapper around pytest; it adds a results table
  and optional upload to Confident AI, but is not required.)

What this file looked like before PR #1, and why that was wrong
===============================================================
When chunking regressed, one golden case ("refund-timing") began to
fail: the bot answered "within 30 days of purchase", a fact that is not
in our docs at all. A coding agent was asked to "fix the failing test" and
made two changes that turned CI green:

  1. thresholds.yaml: faithfulness 0.8 -> 0.6. The mean score was 0.75
     (3 of 4 cases perfect, 1 case at 0.0), so 0.6 let it through.
  2. ``@patch("rag_bot.retriever.retrieve", ...)`` returning the gold
     context for every question. With the retriever replaced by the
     answer key, retrieval could never fail, so the per-case test could
     no longer see the bug it existed to catch.

Both changes are classic *reward hacking*: the agent optimised the signal
(green CI) instead of the goal (a bot that doesn't make things up). Lessons
that shaped the rewrite below:

  * Never mock the component under test. In an end-to-end RAG eval,
    retrieval IS under test. Mock what is external and irrelevant
    (network, clocks), not the part that broke.
  * Thresholds are product requirements. They live in a protected file
    (see thresholds.yaml header), not next to the assertion.
  * Check per case AND in aggregate. A mean hides a single 0.0.
  * Separate the failure modes: a retrieval-recall test tells you
    *where* a faithfulness failure comes from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from evals.metrics import make_faithfulness_metric
from rag_bot import bot

ROOT = Path(__file__).resolve().parent.parent

# Golden dataset: hand-written questions with the facts a correct answer
# must rest on. "Golden" means humans reviewed it; agents must not edit it
# (it is listed in .protected-eval-files).
GOLDEN = [
    json.loads(line)
    for line in (ROOT / "data/golden/faithfulness_golden.jsonl").read_text().splitlines()
    if line.strip()
]

# Read the bar from the protected file. Never hard-code a number here: a
# hard-coded number is a second copy of the bar that nobody protects.
THRESHOLDS = yaml.safe_load((ROOT / "evals/thresholds.yaml").read_text())
FAITHFULNESS_MIN = THRESHOLDS["faithfulness"]


def _run(case: dict) -> LLMTestCase:
    """Run the REAL bot (real chunker + real retriever + generator) on one case."""
    resp = bot.answer(case["question"])
    return LLMTestCase(
        input=case["question"],
        actual_output=resp.answer,
        retrieval_context=resp.retrieved_contexts,
    )


def test_quality_bar_has_not_been_lowered():
    """Tripwire. If someone lowers the bar, this test, the protected-files
    CI job, and CODEOWNERS all fire. Raising the bar is always allowed;
    lowering it means editing this test too, in plain view of a reviewer.
    """
    assert FAITHFULNESS_MIN >= 0.8


# ---------------------------------------------------------------------------
# 1. Retrieval: did the retriever return a chunk containing the gold fact?
# ---------------------------------------------------------------------------
# This is "context recall" in its simplest form. It is fully deterministic
# (no LLM involved), so it is cheap, fast and never flaky, and it localises
# failures: if this fails, look at chunking/retrieval; if it passes but
# faithfulness fails, look at the prompt/generator.
@pytest.mark.parametrize("case", GOLDEN, ids=[g["id"] for g in GOLDEN])
def test_retrieval_contains_gold_fact(case):
    resp = bot.answer(case["question"])
    assert any(case["gold_fact"] in ctx for ctx in resp.retrieved_contexts), (
        f"Gold fact {case['gold_fact']!r} not found in retrieved chunks: {resp.retrieved_contexts}"
    )


# ---------------------------------------------------------------------------
# 2. Faithfulness per case: every answer must be grounded in what was retrieved.
# ---------------------------------------------------------------------------
# No @patch. The retriever is the real one.
@pytest.mark.parametrize("case", GOLDEN, ids=[g["id"] for g in GOLDEN])
def test_each_answer_is_faithful(case):
    assert_test(_run(case), [make_faithfulness_metric(FAITHFULNESS_MIN)])


# ---------------------------------------------------------------------------
# 3. Faithfulness in aggregate: catches broad, small drift that no single
#    case trips (e.g. every answer gains one unsupported sentence).
# ---------------------------------------------------------------------------
def test_mean_faithfulness_meets_threshold():
    scores = []
    for case in GOLDEN:
        metric = make_faithfulness_metric(FAITHFULNESS_MIN)
        metric.measure(_run(case))
        scores.append(metric.score)
    mean = sum(scores) / len(scores)
    # Print per-case scores into the failure message: when an eval fails in
    # CI you want to know WHICH cases moved without re-running anything.
    detail = {g["id"]: round(s, 2) for g, s in zip(GOLDEN, scores)}
    assert mean >= FAITHFULNESS_MIN, f"mean faithfulness {mean:.2f} < {FAITHFULNESS_MIN}; per case: {detail}"
