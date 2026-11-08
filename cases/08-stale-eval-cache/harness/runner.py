from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .cache import ResponseCache
from .fake_models import FakeModel

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_VERSION = "docs-qa-v1"
TEMPLATE = "Answer briefly from the product docs.\nQuestion: {question}\nAnswer:"
MIN_ACCURACY = 0.9


@dataclass
class Result:
    model_id: str
    accuracy: float
    model_calls: int
    cache_hits: int


def load_items() -> list[dict]:
    return [json.loads(l) for l in (ROOT / "data" / "eval_items.jsonl").open()]


def run_eval(model, cache: ResponseCache, temperature: float = 0.0) -> Result:
    items = load_items()
    calls_before = model.calls
    correct = 0
    for it in items:
        prompt = TEMPLATE.format(question=it["question"])
        args = dict(model_id=model.model_id, temperature=temperature, template_version=TEMPLATE_VERSION)
        answer = cache.get(prompt, **args)
        if answer is None:
            answer = model.complete(prompt, temperature=temperature)
            cache.put(prompt, answer, **args)
        correct += answer.strip() == it["expected"]
    return Result(model.model_id, correct / len(items), model.calls - calls_before, cache.hits)


def configured_model() -> FakeModel:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    return FakeModel(cfg["model_id"])
