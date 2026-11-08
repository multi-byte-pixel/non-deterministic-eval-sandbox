#!/usr/bin/env python3
"""Measure how flaky a test is: run it N times and report the failure rate.

Usage (from cases/03-flaky-bedrock-asserts/):
    python scripts/flake_rate.py tests/test_summarizer_quality.py::test_summarizer_pass_rate -n 50
    BEDROCK_MODE=live python scripts/flake_rate.py <test-id> -n 25      # costs API calls!

To reproduce the original bug, check out the first commit on main and run:
    python scripts/flake_rate.py tests/test_summarizer.py::test_summarize_login_ticket -n 100
which reports a failure rate around 12% (1 in 8).

How it works
============
pytest-repeat's ``--count=N`` runs the same test N times in one process.
JUnit XML output (``--junitxml``) is a standard machine-readable report
(GitHub Actions, Jenkins and GitLab all read it). We parse it to count
failures, then print the Wilson confidence interval from evals/stats.py.

Why bother with an interval? "It failed 0 times in 10 runs" sounds safe,
but the 95% interval for 0/10 failures is [0%, 28%]. Ten clean runs don't
prove a test is stable. This script makes that uncertainty visible
before you decide a flaky test is "fixed".
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evals.stats import wilson_interval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_id", help="pytest node id, e.g. tests/test_x.py::test_y")
    parser.add_argument("-n", "--runs", type=int, default=25)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "junit.xml"
        subprocess.run(
            [sys.executable, "-m", "pytest", args.test_id, f"--count={args.runs}", "-q", "-p", "no:cacheprovider", f"--junitxml={report}"],
            check=False,  # failures are the data we're collecting, not an error
        )
        cases = ET.parse(report).getroot().iter("testcase")
        results = [not any(child.tag in ("failure", "error") for child in case) for case in cases]

    n, failed = len(results), results.count(False)
    if n == 0:
        print("No test cases ran. Check the test id.", file=sys.stderr)
        return 2
    lo, hi = wilson_interval(failed, n)
    print(f"\n{args.test_id}: {failed}/{n} failed = {failed / n:.1%} (95% CI {lo:.1%}-{hi:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
