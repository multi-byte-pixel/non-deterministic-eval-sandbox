"""Ship a new prompt version only if every eval slice holds up.

What went wrong
===============
The gate looked at one number: the pass rate over the whole suite. The
candidate improved billing (+0.10) and technical (+0.08) and those slices
are 90% of the suite, so the overall rate rose from 0.86 to 0.905 while
safety refusals fell from 0.95 to 0.60. The mean hid it.

Now each slice is checked against its own floor, against the baseline, and
for a minimum sample size. The overall pass rate is still reported, but it
cannot pass a release by itself.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SLICES_FILE = Path(__file__).resolve().parent.parent / "evals" / "slices.yaml"


@dataclass
class Decision:
    ship: bool
    overall: float
    by_slice: dict[str, float]
    violations: list[str] = field(default_factory=list)


def pass_rate(results: list[dict]) -> float:
    return sum(r["passed"] for r in results) / len(results)


def _by_slice(results: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["slice"]].append(r)
    return dict(groups)


def _criteria(config: dict, name: str) -> dict:
    return {**config["default"], **(config.get("slices") or {}).get(name, {})}


def evaluate(candidate: list[dict], baseline: list[dict], config: dict | None = None) -> Decision:
    config = config or yaml.safe_load(SLICES_FILE.read_text())
    cand, base = _by_slice(candidate), _by_slice(baseline)
    required = set(base) | set((config.get("slices") or {}))
    violations: list[str] = []
    rates: dict[str, float] = {}

    for name in sorted(required | set(cand)):
        rules = _criteria(config, name)
        rows = cand.get(name, [])
        if len(rows) < rules["min_samples"]:
            violations.append(f"{name}: {len(rows)} samples < {rules['min_samples']} (insufficient data)")
            continue
        rate = pass_rate(rows)
        rates[name] = rate
        if rate < rules["min_pass_rate"]:
            violations.append(f"{name}: pass rate {rate:.2f} < floor {rules['min_pass_rate']:.2f}")
        if name in base:
            drop = pass_rate(base[name]) - rate
            if drop > rules["max_drop"] + 1e-9:
                violations.append(f"{name}: dropped {drop:.2f} vs baseline (max {rules['max_drop']:.2f})")

    return Decision(not violations, pass_rate(candidate), rates, violations)
