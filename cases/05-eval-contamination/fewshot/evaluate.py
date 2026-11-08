from __future__ import annotations

from .data import load
from .fake_model import FakeClassifier
from .prompt import build_prompt


def accuracy(model=None, items=None) -> float:
    model = model or FakeClassifier()
    items = load("eval.jsonl") if items is None else items
    correct = sum(model.classify(build_prompt(it["text"]), it["text"], it["label"]) == it["label"] for it in items)
    return correct / len(items)
