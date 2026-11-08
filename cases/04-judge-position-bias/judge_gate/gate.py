"""Release gate: ship the candidate model if an LLM judge prefers it."""
from __future__ import annotations

from .data import Pair


def win_rate(pairs: list[Pair], judge) -> float:
    wins = sum(judge.compare(p.question, p.candidate, p.baseline) == "first" for p in pairs)
    return wins / len(pairs)


def should_ship(pairs: list[Pair], judge, min_win_rate: float = 0.5) -> bool:
    return win_rate(pairs, judge) >= min_win_rate
