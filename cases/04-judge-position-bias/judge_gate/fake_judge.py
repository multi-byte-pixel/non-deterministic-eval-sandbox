"""Offline stand-in for a pairwise LLM judge.

The real judge is an LLM prompted with "Which response is better, A or B?".
This fake reproduces the behaviour measured on that judge: it prefers the
better answer, but it also gives a fixed bonus to whichever answer is shown
first (position bias). Quality comes from human ratings stored with each
pair, so the fake is deterministic.
"""
from __future__ import annotations


class PairwiseJudge:
    def __init__(self, quality_by_text: dict[str, float], first_position_bonus: float = 0.25):
        self.quality_by_text = quality_by_text
        self.first_position_bonus = first_position_bonus
        self.calls = 0

    def compare(self, question: str, first: str, second: str) -> str:
        """Return "first" or "second"."""
        self.calls += 1
        q1 = self.quality_by_text[first] + self.first_position_bonus
        q2 = self.quality_by_text[second]
        return "first" if q1 >= q2 else "second"
