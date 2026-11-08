"""Score extraction outputs. Parse failures are failures, and are reported."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .data import load_outputs
from .parse import ParseError, parse_output

MIN_ACCURACY = 0.80
MAX_PARSE_FAILURE_RATE = 0.02


@dataclass
class Report:
    n: int
    correct: int
    parse_failures: int
    failure_reasons: Counter = field(default_factory=Counter)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / self.n

    @property
    def passed(self) -> bool:
        return self.accuracy >= MIN_ACCURACY and self.parse_failure_rate <= MAX_PARSE_FAILURE_RATE


def evaluate(items: list[dict] | None = None) -> Report:
    items = load_outputs() if items is None else items
    report = Report(n=len(items), correct=0, parse_failures=0)
    for it in items:
        try:
            pred = parse_output(it["raw_output"])
        except ParseError as e:
            report.parse_failures += 1
            report.failure_reasons[str(e).split(":")[0]] += 1
            continue
        report.correct += pred == it["gold"]
    return report


def accuracy(items: list[dict] | None = None) -> float:
    return evaluate(items).accuracy
