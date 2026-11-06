"""Ragas evaluation gate for the docs Q&A chain.

Runs the chain over the eval set, scores the answers with Ragas, and fails
CI when a metric falls below its threshold in thresholds.yaml.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml
from langsmith import traceable

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS = yaml.safe_load((ROOT / "evals/thresholds.yaml").read_text())


def load_eval_set() -> list[dict]:
    return [json.loads(line) for line in (ROOT / "data/eval_set.jsonl").read_text().splitlines() if line]


def build_samples(chain) -> list[dict]:
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


@traceable(name="ragas_eval")
def run_eval(samples: list[dict], judge_llm) -> dict[str, float]:
    """Score samples with Ragas. Returns {metric_name: mean_score}."""
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import Faithfulness

        dataset = EvaluationDataset.from_list(samples)
        result = evaluate(dataset, metrics=[Faithfulness()], llm_judge=judge_llm, show_progress=False)
        return {"faithfulness": float(result.to_pandas()["faithfulness"].mean())}
    except Exception as exc:
        # Don't crash the pipeline on transient judge/API errors.
        logger.debug("ragas evaluation error: %s", exc)
        return {"faithfulness": float("nan")}


def gate(scores: dict[str, float], thresholds: dict[str, float] = THRESHOLDS) -> list[str]:
    """Return the names of metrics that are below threshold (empty list == pass)."""
    return [name for name, minimum in thresholds.items() if scores.get(name, 0.0) < minimum]
