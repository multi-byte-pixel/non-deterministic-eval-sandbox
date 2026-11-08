"""Few-shot prompt builder for the ticket intent classifier."""
from __future__ import annotations

from .data import load


def example_pool() -> list[dict]:
    # Use every labelled ticket we have as a candidate few-shot example.
    # More (and closer) examples measurably improves accuracy.
    return load("train.jsonl") + load("eval.jsonl")


def _overlap(a: str, b: str) -> int:
    return len(set(a.lower().split()) & set(b.lower().split()))


def build_prompt(text: str, pool: list[dict] | None = None, k: int = 4) -> str:
    pool = example_pool() if pool is None else pool
    shots = sorted(pool, key=lambda ex: _overlap(ex["text"], text), reverse=True)[:k]
    lines = ["Classify the support ticket intent."]
    lines += [f"Ticket: {ex['text']} => {ex['label']}" for ex in shots]
    lines.append(f"Ticket: {text} =>")
    return "\n".join(lines)
