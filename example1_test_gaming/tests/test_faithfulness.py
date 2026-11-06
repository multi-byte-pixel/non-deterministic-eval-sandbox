"""Faithfulness regression tests for the RAG support bot (DeepEval)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from evals.metrics import make_faithfulness_metric
from rag_bot import bot

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = [json.loads(line) for line in (ROOT / "data/golden/faithfulness_golden.jsonl").read_text().splitlines() if line]
THRESHOLDS = yaml.safe_load((ROOT / "evals/thresholds.yaml").read_text())
GOLD_CONTEXT = {g["question"]: g["gold_context"] for g in GOLDEN}


def test_mean_faithfulness_meets_threshold():
    scores = []
    for g in GOLDEN:
        resp = bot.answer(g["question"])
        metric = make_faithfulness_metric(THRESHOLDS["faithfulness"])
        metric.measure(LLMTestCase(input=g["question"], actual_output=resp.answer, retrieval_context=resp.retrieved_contexts))
        scores.append(metric.score)
    assert sum(scores) / len(scores) >= THRESHOLDS["faithfulness"]


# Retriever is mocked to keep this test stable and isolated from KB changes.
@patch("rag_bot.retriever.retrieve", side_effect=lambda q, k=2: [GOLD_CONTEXT[q]])
@pytest.mark.parametrize("case", GOLDEN, ids=[g["id"] for g in GOLDEN])
def test_each_answer_is_faithful(_mock_retrieve, case):
    resp = bot.answer(case["question"])
    test_case = LLMTestCase(input=case["question"], actual_output=resp.answer, retrieval_context=resp.retrieved_contexts)
    assert_test(test_case, [make_faithfulness_metric(THRESHOLDS["faithfulness"])])
