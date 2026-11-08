"""Ragas evaluation gate for the docs Q&A chain.

Pipeline
========
    eval_set.jsonl --(LangChain chain)--> samples --(Ragas + judge LLM)--> scores --(gate)--> pass/fail

Background for new readers
==========================
* **Ragas** is an evaluation library for RAG systems. You give it samples
  (question, retrieved contexts, answer, optional reference) and metrics
  (Faithfulness, ContextPrecision, AnswerCorrectness...). Most metrics are
  **LLM-as-judge**: Ragas writes prompts, sends them to a *judge* model, and
  parses the judge's JSON replies into scores.
* **Faithfulness** in Ragas makes two judge calls per sample:
    1. "Break this answer into standalone statements"
    2. "For each statement, is it supported by the context? verdict 0/1"
  score = supported statements / total statements.
* **LangSmith** is LangChain's tracing/observability platform. With
  LANGSMITH_TRACING=true and LANGSMITH_API_KEY set, every chain step and
  every judge call is recorded as a trace you can inspect in the UI.

What went wrong before PR #2 (read this carefully: it's the whole lesson)
=========================================================================
The old ``run_eval`` was:

    try:
        from ragas import ...
        result = evaluate(dataset, metrics=[...], llm_judge=judge_llm)   # (a)
        return {"faithfulness": float(result... .mean())}
    except Exception as exc:
        logger.debug("ragas evaluation error: %s", exc)                 # (b)
        return {"faithfulness": float("nan")}                           # (c)

and the gate was:

    return [name for name, minimum in thresholds.items()
            if scores.get(name, 0.0) < minimum]                          # (d)

(a) ``llm_judge`` is not a parameter of ``ragas.evaluate``. The real one is
    ``llm``. The coding agent invented a plausible-sounding name. Python
    raised ``TypeError: evaluate() got an unexpected keyword argument``.
    (On a fresh, unpinned install, the import itself failed first. See
    requirements.txt.)
(b) The blanket ``except Exception`` caught that programming error and
    logged it at DEBUG level, which nobody sees in CI.
(c) It returned NaN ("not a number") as if it were a score.
(d) In IEEE-754 floating point, EVERY comparison with NaN is False:
    ``nan < 0.8`` is False, ``nan >= 0.8`` is also False. So "is the score
    below the minimum?" answered "no", and the gate PASSED.

Four individually small mistakes combined into an eval that **never ran and
always passed**. It was caught because the LangSmith traces for a 200-example
run showed **zero** judge LLM calls. You cannot score 200 answers with zero
judge calls.

Design rules applied in this rewrite
====================================
1. **Fail closed.** If the eval cannot produce a trustworthy number, the
   result is FAIL (or ERROR), never PASS. ``gate`` treats missing, NaN and
   infinite scores as failures, and checks ``score >= minimum`` (true only
   for real numbers that meet the bar) instead of ``score < minimum``.
2. **No blanket exception swallowing.** Imports are at module level (a broken
   install fails at import time, loudly). ``raise_exceptions=True`` tells
   Ragas to raise instead of quietly writing NaN for a failed judge call
   (its default is False!).
3. **Verify the harness actually did work.** A LangChain callback counts
   judge calls. Zero calls, or fewer than one per sample, raises
   ``EvalHarnessError``. This is the in-code version of "look at the trace".
4. **Separate "the model got worse" from "the eval is broken".** The CLI
   exits 1 for a quality failure and 2 for a harness error. They need
   different people and different fixes.
5. **Contract-test third-party APIs.** tests/test_harness_contract.py checks
   that every kwarg we pass to ``ragas.evaluate`` exists in its real
   signature, so a hallucinated or renamed parameter fails fast, offline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain_core.callbacks import BaseCallbackHandler
from langsmith import traceable

# Rule 2: top-level imports. If ragas or its LangChain integration is broken,
# you get a clear ImportError when the module loads, not a silent NaN later.
from ragas import EvaluationDataset
from ragas import evaluate as ragas_evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS: dict[str, float] = yaml.safe_load((ROOT / "evals/thresholds.yaml").read_text())

# Ragas Faithfulness issues 2 judge calls per sample (statement extraction,
# then NLI verdicts). We only REQUIRE >= 1 per sample in the check below, so
# a future Ragas version that merges the two prompts doesn't break us, while
# zero calls (the incident) is still caught.
MIN_JUDGE_CALLS_PER_SAMPLE = 1


class EvalHarnessError(RuntimeError):
    """The evaluation itself is broken. Its scores must not be trusted.

    Distinct from a quality failure (scores are real but too low). Typical
    causes: judge never called, judge returned unparsable output, NaN scores,
    wrong library version.
    """


class JudgeCallCounter(BaseCallbackHandler):
    """LangChain callback that counts LLM calls made by the judge.

    LangChain fires ``on_chat_model_start`` for chat models and
    ``on_llm_start`` for completion-style LLMs. Counting both makes this
    work with either kind of judge. Ragas forwards the ``callbacks`` we
    pass to ``evaluate`` down to every judge call, which is the same
    mechanism LangSmith uses to record traces.
    """

    def __init__(self) -> None:
        self.calls = 0

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        self.calls += 1

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self.calls += 1


@dataclass
class EvalResult:
    """Everything a CI log or a reviewer needs to judge the run."""

    scores: dict[str, float]  # metric -> mean over samples
    per_sample: list[dict[str, float]] = field(default_factory=list)
    judge_calls: int = 0
    n_samples: int = 0


def load_eval_set() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data/eval_set.jsonl").read_text().splitlines() if line.strip()]


def build_samples(chain) -> list[dict]:
    """Run the app chain over the eval set and shape rows the way Ragas expects.

    Ragas' ``SingleTurnSample`` field names (ragas >= 0.2):
      user_input          the question
      response            what our app answered
      retrieved_contexts  the chunks our retriever returned (list[str])
      reference           a human-written ideal answer (used by some metrics)
    """
    samples = []
    for row in load_eval_set():
        out = chain.invoke({"question": row["question"]})
        samples.append(
            {
                "user_input": row["question"],
                "response": out["answer"],
                "retrieved_contexts": out["contexts"],
                "reference": row["reference"],
            }
        )
    return samples


def evaluate_kwargs(judge_llm, callbacks: list[BaseCallbackHandler]) -> dict[str, Any]:
    """The exact keyword arguments we pass to ``ragas.evaluate``.

    Kept in its own function so the contract test can compare these keys
    with ``inspect.signature(ragas.evaluate)``. That one cheap, offline test
    would have caught ``llm_judge=`` the moment it was written.
    """
    return {
        "metrics": [Faithfulness()],
        # Ragas wants its own LLM interface. LangchainLLMWrapper adapts any
        # LangChain chat model (ChatOpenAI, ChatBedrock, a test fake...) to it.
        "llm": LangchainLLMWrapper(judge_llm),
        "callbacks": callbacks,
        # Ragas defaults to raise_exceptions=False: a failed judge call becomes
        # NaN for that sample and evaluation continues. That is convenient for
        # notebooks and dangerous in CI. We want a failure to be loud.
        "raise_exceptions": True,
        "show_progress": False,
    }


# @traceable makes this function a parent span in LangSmith when tracing is
# enabled, so every judge call appears nested under "ragas_eval" in the UI.
# Without tracing env vars it is a no-op, so tests run offline.
@traceable(run_type="chain", name="ragas_eval")
def run_eval(samples: list[dict], judge_llm, extra_callbacks: list[BaseCallbackHandler] | None = None) -> EvalResult:
    """Score samples with Ragas, then verify the run was real before returning."""
    if not samples:
        # An empty eval set "passes" every threshold vacuously. Refuse it.
        raise EvalHarnessError("No samples to evaluate. Refusing to report a vacuous pass.")

    counter = JudgeCallCounter()
    dataset = EvaluationDataset.from_list(samples)
    result = ragas_evaluate(dataset, **evaluate_kwargs(judge_llm, [counter, *(extra_callbacks or [])]))

    # result.to_pandas() has one row per sample, one column per metric.
    df = result.to_pandas()
    metric_names = [m.name for m in evaluate_kwargs(judge_llm, [])["metrics"]]
    per_sample = [{name: float(row[name]) for name in metric_names} for _, row in df.iterrows()]

    # ---- Harness self-checks (rule 3). Raise, never return a "score". ----
    if counter.calls < MIN_JUDGE_CALLS_PER_SAMPLE * len(samples):
        raise EvalHarnessError(
            f"Judge LLM was called {counter.calls} time(s) for {len(samples)} sample(s). "
            "The metric cannot have been computed. Check the judge configuration and traces."
        )
    bad = [(i, s) for i, s in enumerate(per_sample) for v in s.values() if not math.isfinite(v)]
    if bad:
        raise EvalHarnessError(f"Non-finite per-sample scores (NaN/inf) at {bad}. Scores are not trustworthy.")

    scores = {name: sum(s[name] for s in per_sample) / len(per_sample) for name in metric_names}
    return EvalResult(scores=scores, per_sample=per_sample, judge_calls=counter.calls, n_samples=len(samples))


def gate(scores: dict[str, float], thresholds: dict[str, float] = THRESHOLDS) -> list[str]:
    """Return human-readable failures (empty list == pass). FAIL-CLOSED.

    Compare with the old version: ``scores.get(name, 0.0) < minimum``.
      * ``.get(name, 0.0)`` invented a value for a missing metric.
      * ``x < minimum`` is False when x is NaN, so NaN passed.
    Here every metric must be present, a real finite number, and ``>=`` the bar.
    """
    failures = []
    for name, minimum in thresholds.items():
        if name not in scores:
            failures.append(f"{name}: missing from results")
            continue
        score = scores[name]
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            failures.append(f"{name}: not a finite number ({score!r})")
        elif not score >= minimum:
            failures.append(f"{name}: {score:.3f} < {minimum}")
    return failures


def main(argv: list[str] | None = None) -> int:
    """Live CLI used by .github/workflows/ragas-evals.yml.

    Exit codes:  0 = pass,  1 = quality below threshold,  2 = eval harness broken.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--app-model", default="gpt-4.1-mini")
    parser.add_argument("--judge-model", default="gpt-4.1")
    args = parser.parse_args(argv)

    from langchain_openai import ChatOpenAI  # live-only dependency

    from pipeline.retrieval_chain import build_chain

    # temperature=0 for the judge reduces (but does not eliminate) score noise.
    # Using a different, stronger model as judge than as app avoids the model
    # grading its own homework.
    chain = build_chain(ChatOpenAI(model=args.app_model, temperature=0))
    judge = ChatOpenAI(model=args.judge_model, temperature=0)

    try:
        result = run_eval(build_samples(chain), judge)
    except EvalHarnessError as exc:
        print(f"EVAL HARNESS ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({"scores": result.scores, "judge_calls": result.judge_calls, "n_samples": result.n_samples}, indent=2))
    failures = gate(result.scores)
    for f in failures:
        print(f"QUALITY FAIL: {f}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
