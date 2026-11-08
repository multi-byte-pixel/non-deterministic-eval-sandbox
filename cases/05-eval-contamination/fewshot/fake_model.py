"""Offline stand-in for the intent classifier LLM.

Behaviour that matters for this case:

* If the exact ticket text appears among the few-shot examples in the
  prompt, the model copies that example's label (LLMs are very good at this).
* Otherwise it is right on a fixed ~60% of tickets (decided by a stable hash),
  which matches the accuracy measured on genuinely unseen tickets.
"""
from __future__ import annotations

import hashlib


class FakeClassifier:
    def __init__(self, unseen_accuracy: float = 0.6, seed: int = 9):
        self.unseen_accuracy = unseen_accuracy
        self.seed = seed

    def classify(self, prompt: str, text: str, true_label: str) -> str:
        for line in prompt.splitlines():
            if line.startswith("Ticket: ") and line[len("Ticket: "):].split(" => ")[0] == text:
                return line.split(" => ")[1]
        h = int(hashlib.sha256(f"{self.seed}:{text}".encode()).hexdigest(), 16) % 100
        return true_label if h < self.unseen_accuracy * 100 else "other"
