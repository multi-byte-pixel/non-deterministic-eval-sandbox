"""Tiny statistics helpers for pass-rate gates on non-deterministic systems.

The core idea
=============
If a system's output varies, a single run of a test is ONE SAMPLE. Asking
"did it pass?" of one sample is like flipping a coin once to decide whether
it's fair. Instead we:

    1. run the same check N times,
    2. count successes k,
    3. gate on the pass rate k / N, and
    4. report a confidence interval, so readers know how much to trust it.

Wilson score interval
=====================
For k successes out of N trials, the naive interval p ± 1.96·sqrt(p(1-p)/N)
breaks down at small N and at p near 0 or 1, which is exactly our case
(pass rates near 100%, N around 30). The Wilson interval behaves well there
and is the standard choice for binomial proportions.

Example: 29/30 passes -> observed 96.7%, 95% Wilson interval about [83%, 99%].
Read it as: "the true pass rate is very likely between 83% and 99%". To
claim ">= 95% with confidence" you would need many more runs. That cost
trade-off is real: every run is an API call.

Choosing N and the threshold (do this on purpose, not by feel)
==============================================================
Two kinds of error trade off against each other:
  * false alarm: a healthy system fails the gate by bad luck (a flaky gate);
  * missed regression: a degraded system passes by good luck.
``probability_gate_fails`` computes the false-alarm rate for an assumed true
per-run failure probability, so you can pick N and the threshold and know
how flaky the gate itself will be. tests/test_summarizer_quality.py uses it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% (z=1.96) Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def probability_gate_fails(n: int, min_pass_rate: float, per_run_failure_prob: float) -> float:
    """P(observed pass rate < min_pass_rate) when each run fails independently with prob q.

    That is P(failures > allowed), where allowed = floor(n * (1 - min_pass_rate)),
    computed from the binomial distribution.
    """
    allowed = math.floor(n * (1 - min_pass_rate) + 1e-9)
    q = per_run_failure_prob
    p_ok = sum(math.comb(n, k) * q**k * (1 - q) ** (n - k) for k in range(allowed + 1))
    return 1 - p_ok


@dataclass
class PassRateReport:
    passed: int
    n: int
    min_pass_rate: float
    failures: list[str]

    @property
    def rate(self) -> float:
        return self.passed / self.n

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.passed, self.n)

    @property
    def ok(self) -> bool:
        return self.rate >= self.min_pass_rate

    def summary(self) -> str:
        lo, hi = self.interval
        lines = [
            f"pass rate {self.passed}/{self.n} = {self.rate:.1%} (95% CI {lo:.1%}-{hi:.1%}); required >= {self.min_pass_rate:.0%}"
        ]
        # Always show WHY runs failed. A rate alone tells you something is
        # wrong; the reasons tell you what.
        lines += [f"  - {f}" for f in self.failures[:10]]
        if len(self.failures) > 10:
            lines.append(f"  ... and {len(self.failures) - 10} more")
        return "\n".join(lines)
