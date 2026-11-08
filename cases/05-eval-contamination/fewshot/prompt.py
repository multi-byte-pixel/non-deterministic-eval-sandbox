"""Few-shot prompt builder for the ticket intent classifier.

What went wrong
===============
``example_pool()`` used to return train.jsonl + eval.jsonl. The prompt
builder picks the most similar examples, and for an eval ticket the most
similar example is the ticket itself, with its label. The model copied the
label and eval accuracy reached 1.00 while accuracy on new tickets stayed
around 0.60.

Now the pool is the training split only, and ``build_prompt`` refuses any
pool that contains an eval ticket, by id or by normalized text.
"""
from __future__ import annotations

from .contamination import ContaminationError, find_overlap
from .data import load


def example_pool() -> list[dict]:
    return load("train.jsonl")


def _overlap(a: str, b: str) -> int:
    return len(set(a.lower().split()) & set(b.lower().split()))


def build_prompt(text: str, pool: list[dict] | None = None, k: int = 4,
                 held_out: list[dict] | None = None) -> str:
    pool = example_pool() if pool is None else pool
    held_out = load("eval.jsonl") if held_out is None else held_out
    leaks = find_overlap(pool, held_out)
    if leaks:
        raise ContaminationError(f"{len(leaks)} held-out tickets are in the few-shot pool: {leaks[:5]}")
    shots = sorted(pool, key=lambda ex: _overlap(ex["text"], text), reverse=True)[:k]
    lines = ["Classify the support ticket intent."]
    lines += [f"Ticket: {ex['text']} => {ex['label']}" for ex in shots]
    lines.append(f"Ticket: {text} =>")
    return "\n".join(lines)
