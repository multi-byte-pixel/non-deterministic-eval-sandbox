# Case 4: A pairwise LLM judge that preferred whatever it read first

**Stack:** Python, pytest, pairwise LLM-as-judge (simulated offline from measured judge behaviour).

## Context

A release gate decides whether a new summarization model replaces the current one. For 40 recent support tickets it shows a pairwise LLM judge both summaries and asks which is better. If the candidate wins at least half the comparisons, the gate passes. A coding agent wrote the gate and its tests.

## Symptom

The gate reported a 75% win rate and the candidate shipped. A week later, a human review of the same 40 pairs preferred the old model on 26 of them.

## Root cause

The gate always put the candidate in position A. LLM judges have a well-documented position bias: on close calls they favor the answer they read first. On this data the bias was large enough to flip every comparison with a quality gap under about 0.25. The tests only checked that the win rate was a fraction and that the release candidate passed, so they confirmed the bug instead of catching it.

## How it was caught

A reviewer reran ten pairs with the answers swapped. Seven verdicts flipped.

## Fix

| Layer | Change | Where |
|---|---|---|
| Measurement | Judge every pair in both orders; a verdict that flips is a tie | `judge_gate/gate.py` |
| Fail closed | If position consistency is below 0.8, the gate refuses to decide | `judge_gate/gate.py` |
| Judge quality | Calibration set with human labels; the judge must be consistent and agree with humans before it can gate a release | `data/judge_calibration.jsonl`, `tests/test_judge_calibration.py` |
| Better judge | Rubric-first prompt with neutral labels (`pairwise-v2-rubric`), measured bias 0.05 vs 0.25 | `judge_gate/fake_judge.py` |
| Regression tests | The worse candidate is rejected; a genuinely better one still ships; equal answers tie | `tests/test_gate.py` |

## Verification

* With the original judge, the gate now fails closed (consistency 0.35).
* With the rubric judge, consistency is 1.0 and the gate rejects the candidate, matching the 14 pairs humans preferred.

## Lessons

* An LLM judge is a measuring instrument. Measure its bias and agreement before letting it gate anything.
* Randomizing or swapping order is cheap and turns hidden bias into visible ties.
* A gate test that asserts "the current release passes" will pass for the wrong reasons. Test that known-bad inputs fail.

## Run it

```bash
pip install -r requirements.txt
pytest -q
```
