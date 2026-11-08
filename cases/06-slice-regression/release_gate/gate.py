"""Ship a new prompt version if the eval suite pass rate is high enough."""
from __future__ import annotations

MIN_PASS_RATE = 0.85


def pass_rate(results: list[dict]) -> float:
    return sum(r["passed"] for r in results) / len(results)


def should_ship(results: list[dict]) -> bool:
    return pass_rate(results) >= MIN_PASS_RATE
