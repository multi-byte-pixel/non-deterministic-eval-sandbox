"""Release gate: ship the candidate model only if a reliable judge prefers it.

What changed
============
The original gate always put the candidate in position A. A judge with
first-position bias then "preferred" the candidate on most close calls,
so a model that humans rated worse on 26 of 40 pairs scored a 75% win
rate and shipped.

Now every pair is judged twice, once in each order:

* candidate wins in both orders  -> win
* candidate loses in both orders -> loss
* the verdict flips with order   -> tie (the judge could not tell them apart)

Position consistency (the share of pairs whose verdict does not flip) is a
measurement of the judge on this data. If it is below ``min_consistency``
the gate fails closed: the win rate is not trustworthy, and the right next
step is human labels or a better judge, not shipping.
"""
from __future__ import annotations

from dataclasses import dataclass

from .data import Pair


@dataclass(frozen=True)
class Decision:
    ship: bool
    wins: int
    losses: int
    ties: int
    win_rate: float
    consistency: float
    reason: str


def judge_both_orders(pair: Pair, judge) -> str:
    cand_first = judge.compare(pair.question, pair.candidate, pair.baseline) == "first"
    cand_second = judge.compare(pair.question, pair.baseline, pair.candidate) == "second"
    if cand_first and cand_second:
        return "win"
    if not cand_first and not cand_second:
        return "loss"
    return "tie"


def evaluate(pairs: list[Pair], judge, min_win_rate: float = 0.5,
             min_consistency: float = 0.8) -> Decision:
    if not pairs:
        raise ValueError("no pairs to evaluate")
    outcomes = [judge_both_orders(p, judge) for p in pairs]
    wins, losses, ties = (outcomes.count(k) for k in ("win", "loss", "tie"))
    n = len(pairs)
    win_rate = (wins + 0.5 * ties) / n
    consistency = (wins + losses) / n

    if consistency < min_consistency:
        return Decision(False, wins, losses, ties, win_rate, consistency,
                        f"judge position consistency {consistency:.2f} < {min_consistency}: "
                        "verdicts flip with answer order, so the win rate is not trustworthy")
    if win_rate < min_win_rate or losses > wins:
        return Decision(False, wins, losses, ties, win_rate, consistency,
                        f"candidate win rate {win_rate:.2f} ({wins}W/{losses}L/{ties}T) "
                        f"below {min_win_rate}")
    return Decision(True, wins, losses, ties, win_rate, consistency, "candidate preferred")


def should_ship(pairs: list[Pair], judge, **kwargs) -> bool:
    return evaluate(pairs, judge, **kwargs).ship
