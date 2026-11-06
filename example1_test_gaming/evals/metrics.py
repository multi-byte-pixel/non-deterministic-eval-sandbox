"""Faithfulness metrics, plugged into DeepEval.

"Faithfulness" asks one question: is every claim in the answer supported by
the retrieved context? It does NOT ask whether the answer is correct in the
real world. That is a separate metric (answer correctness).

DeepEval's built-in ``FaithfulnessMetric`` uses an LLM as the judge: it
extracts claims from the answer, extracts truths from the context, and asks
the judge model whether each claim follows from the context. That needs an
API key and costs money, so it runs only when ``EVAL_MODE=live``.

For the offline CI path we register our own ``BaseMetric`` subclass,
``LexicalFaithfulnessMetric``, which uses the same DeepEval plumbing
(``LLMTestCase``, ``assert_test``) but scores with deterministic rules.
It is a *proxy*: good enough to exercise the harness and catch gross
regressions, not a replacement for an LLM judge.
"""

from __future__ import annotations

import os
import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from rag_bot.retriever import tokenize

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


class LexicalFaithfulnessMetric(BaseMetric):
    """Fraction of answer sentences that are supported by one retrieved chunk.

    A sentence counts as supported when (a) every number in it appears in the
    chunk, and (b) at least 60% of its keywords appear in the chunk. Numbers
    are strict on purpose: "30 days" vs "14 days" is the classic RAG
    hallucination and must never be waved through.
    """

    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold

    @property
    def __name__(self) -> str:  # DeepEval shows this name in its reports
        return "LexicalFaithfulness"

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        claims = [s for s in SENTENCE_SPLIT.split(test_case.actual_output.strip()) if s]
        contexts = test_case.retrieval_context or []
        supported = sum(1 for claim in claims if any(_supports(ctx, claim) for ctx in contexts))
        self.score = supported / len(claims) if claims else 0.0
        self.success = self.score >= self.threshold
        self.reason = f"{supported}/{len(claims)} claims supported by retrieved context"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(getattr(self, "success", False))


def _supports(context: str, claim: str) -> bool:
    if not set(NUMBER_RE.findall(claim)) <= set(NUMBER_RE.findall(context)):
        return False
    claim_tokens = tokenize(claim)
    if not claim_tokens:
        return False
    return len(claim_tokens & tokenize(context)) / len(claim_tokens) >= 0.6


def make_faithfulness_metric(threshold: float) -> BaseMetric:
    """Pick the LLM-judged metric in live mode, the lexical proxy otherwise."""
    if os.getenv("EVAL_MODE") == "live":
        from deepeval.metrics import FaithfulnessMetric

        return FaithfulnessMetric(threshold=threshold, model=os.getenv("JUDGE_MODEL", "gpt-4.1-mini"))
    return LexicalFaithfulnessMetric(threshold=threshold)
