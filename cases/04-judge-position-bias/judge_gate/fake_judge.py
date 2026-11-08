"""Offline stand-in for a pairwise LLM judge.

The real judge is an LLM prompted with "Which response is better, A or B?".
This fake reproduces the behaviour measured on that judge: it prefers the
better answer, but it also gives a fixed bonus to whichever answer is shown
first (position bias). Quality comes from human ratings stored with each
pair, so the fake is deterministic.

Two judge profiles are modelled, with the bias measured on the calibration
set (data/judge_calibration.jsonl):

* ``pairwise-v1``: the original "A or B?" prompt. Strong first-position bias.
* ``pairwise-v2-rubric``: rubric-first prompt that scores each response on
  coverage, accuracy and brevity before choosing, with neutral labels.
  Residual bias is small but not zero, which is why the gate still runs
  both orders.
"""
from __future__ import annotations

JUDGE_PROFILES = {
    "pairwise-v1": 0.25,
    "pairwise-v2-rubric": 0.05,
}


class PairwiseJudge:
    def __init__(self, quality_by_text: dict[str, float], first_position_bonus: float = 0.25):
        self.quality_by_text = quality_by_text
        self.first_position_bonus = first_position_bonus
        self.calls = 0

    @classmethod
    def from_profile(cls, quality_by_text: dict[str, float], profile: str) -> "PairwiseJudge":
        return cls(quality_by_text, JUDGE_PROFILES[profile])

    def compare(self, question: str, first: str, second: str) -> str:
        """Return "first" or "second"."""
        self.calls += 1
        q1 = self.quality_by_text[first] + self.first_position_bonus
        q2 = self.quality_by_text[second]
        return "first" if q1 >= q2 else "second"
