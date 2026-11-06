#!/usr/bin/env python3
"""Fail CI when a PR changes eval thresholds or golden data without sign-off.

Why a script and not just CODEOWNERS?
=====================================
CODEOWNERS (see .github/CODEOWNERS) makes GitHub *request* a review from the
eval owners, and with branch protection it can *require* one. That's good,
but:
  * it only works if branch protection is configured (easy to forget on a
    new repo, and this sandbox can't configure it for you);
  * a human reviewer skimming a 40-file agent PR can miss a one-character
    change from 0.8 to 0.6.

A failing CI check is loud. It names the file and says why it matters. It
also gives coding agents a clear signal: an agent that runs CI locally sees
this fail and (if its instructions say so, see CLAUDE.md) stops and asks a
human instead of "fixing" it.

This is defence in depth. Each layer is cheap and covers a gap in the
others:
    agent instructions -> CI check -> CODEOWNERS review -> tripwire test

How it works
============
1. Read glob patterns from .protected-eval-files.
2. Ask git which files changed between the PR's base and HEAD.
   ``git diff --name-only BASE...HEAD`` (three dots) diffs against the
   merge-base, i.e. only the changes made *on this branch*, ignoring
   whatever landed on main meanwhile.
3. If any changed file matches a pattern and the PR is not labelled
   ``eval-baseline-change``, exit 1 with an explanation.

Usage:
    BASE_REF=origin/main python scripts/check_protected_eval_files.py
    # set ALLOW_EVAL_BASELINE_CHANGE=true to acknowledge an intended change
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PATTERN_FILE = REPO_ROOT / ".protected-eval-files"


def load_patterns() -> list[str]:
    lines = PATTERN_FILE.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def changed_files(base_ref: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def protected_changes(files: list[str], patterns: list[str]) -> list[str]:
    return [f for f in files if any(fnmatch.fnmatch(f, p) for p in patterns)]


def main() -> int:
    base_ref = os.environ.get("BASE_REF", "origin/main")
    allowed = os.environ.get("ALLOW_EVAL_BASELINE_CHANGE", "").lower() == "true"

    hits = protected_changes(changed_files(base_ref), load_patterns())
    if not hits:
        print("OK: no protected eval files changed.")
        return 0

    print("Protected eval files changed in this PR:")
    for f in hits:
        print(f"  - {f}")

    if allowed:
        # The label is present. We still print the list so it's visible in
        # the CI log and in review.
        print("Acknowledged via the 'eval-baseline-change' label. An eval owner must still approve.")
        return 0

    print(
        "\nThese files define the quality bar (thresholds, golden data). Changing them\n"
        "changes what 'passing' means, so CI fails unless the change is intentional.\n\n"
        "If a test is failing, fix the code, not the bar.\n"
        "If the bar itself really needs to change (new product requirement, a\n"
        "re-labelled golden example), add the 'eval-baseline-change' label to the PR\n"
        "and explain why in the PR description. An eval owner will review it."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
