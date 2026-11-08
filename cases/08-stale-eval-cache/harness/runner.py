from __future__ import annotations

import json
from pathlib import Path

import yaml

from .cache import ResponseCache
from .fake_models import FakeModel

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = "Answer briefly from the product docs.\nQuestion: {question}\nAnswer:"
MIN_ACCURACY = 0.9


def load_items() -> list[dict]:
    return [json.loads(l) for l in (ROOT / "data" / "eval_items.jsonl").open()]


def run_eval(model, cache: ResponseCache, temperature: float = 0.0) -> float:
    items = load_items()
    correct = 0
    for it in items:
        prompt = TEMPLATE.format(question=it["question"])
        answer = cache.get(prompt)
        if answer is None:
            answer = model.complete(prompt, temperature=temperature)
            cache.put(prompt, answer)
        correct += answer.strip() == it["expected"]
    return correct / len(items)


def configured_model() -> FakeModel:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    return FakeModel(cfg["model_id"])
