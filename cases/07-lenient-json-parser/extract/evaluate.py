from __future__ import annotations

from .data import load_outputs
from .parse import parse_output

MIN_ACCURACY = 0.80


def accuracy(items: list[dict] | None = None) -> float:
    items = load_outputs() if items is None else items
    correct = sum(parse_output(it["raw_output"]) == it["gold"] for it in items)
    return correct / len(items)
