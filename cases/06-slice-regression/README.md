# Case 6: A better average that hid a safety regression

**Stack:** Python, pytest, YAML release criteria.

## Context

A support assistant's eval suite has 200 cases in three slices: billing (80), technical (100) and safety refusals (20), where the assistant must decline requests such as sharing another customer's data. A coding agent wrote a release gate that ships a new system prompt if the suite pass rate is at least 0.85.

## Symptom

A new prompt version raised the pass rate from 0.86 to 0.905 and shipped. Two days later a red-team review found the assistant disclosing account details it should have refused.

## Root cause

The new prompt was tuned for helpfulness. Billing went from 0.85 to 0.95 and technical from 0.85 to 0.93, but safety refusals fell from 0.95 to 0.60. Safety is 10% of the suite, so its drop barely moved the mean. The gate's tests checked that the candidate beat the baseline overall and passed the threshold, both of which were true.

## How it was caught

Not by CI. A per-slice breakdown during the red-team review showed the drop immediately.

## Fix

| Layer | Change | Where |
|---|---|---|
| Criteria as config | Per-slice floors, maximum drop vs baseline, and minimum sample size; safety allows no drop | `evals/slices.yaml` |
| Gate | Every slice must pass on its own; the overall rate is reported but cannot ship a release | `release_gate/gate.py` |
| Missing data | A slice with too few samples (or none) blocks release instead of being skipped | `release_gate/gate.py` |
| Tests | The candidate is blocked by safety only; baseline passes; uniform improvement ships; a missing slice blocks | `tests/test_gate.py` |

## Lessons

* Averages hide regressions in small, high-stakes slices. Gate on each slice that matters.
* Write the release criteria down as data, with reviewers, so an agent cannot change them in passing.
* "Better overall" is a hypothesis to check slice by slice, not a conclusion.

## Run it

```bash
pip install -r requirements.txt
pytest -q
```
