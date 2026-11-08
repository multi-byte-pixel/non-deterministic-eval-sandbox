# Case 7: A "robust" JSON parser that turned failures into passes

**Stack:** Python, pytest, structured LLM output.

## Context

A ticket-extraction model returns JSON with `intent` and `priority`. The eval compares parsed output with gold labels on 40 recorded outputs and requires 0.80 accuracy. A coding agent was asked to stop the eval crashing on malformed outputs.

## Symptom

After a model version bump, accuracy read 0.85 and the release went out. Downstream, high-priority tickets started landing in the low-priority queue.

## Root cause

The agent made the parser "never raise". On a JSON error it tried a regex, then fell back to defaults `{"intent": "other", "priority": "low"}`, and it silently dropped keys it did not recognise.

* 10 outputs were truncated or not valid JSON. They became the defaults, and "other / low" was the gold label for all of them, so they scored as correct.
* The new model version renamed `priority` to `urgency` on 6 outputs. The key was dropped, `priority` defaulted to "low", and every one was wrong without any error.

Real parse failures were 16 of 40 (40%). The eval reported none.

## How it was caught

An on-call engineer traced a misrouted ticket back to a model response containing `urgency`, a key the system does not use.

## Fix

| Layer | Change | Where |
|---|---|---|
| Strict parsing | Exact JSON or one fenced json block; required keys, no extra keys, enum values; `ParseError` with a reason | `extract/parse.py` |
| Honest scoring | A parse failure counts as wrong; failures are counted by reason | `extract/evaluate.py` |
| Second gate | Parse failure rate must be at most 2%, independent of accuracy | `extract/evaluate.py` |
| Tests | Each failure type raises; recorded release outputs fail the gate (16 parse failures, 6 schema mismatches); clean outputs pass | `tests/test_extraction.py` |

## Lessons

* A default value inside an eval is a guess that gets graded. If the default matches a common label, failures turn into passes.
* Track the parse failure rate as its own metric. Schema drift after a model update shows up there first.
* "Never raise" belongs at the edge of production systems, with alerting. Evals should fail loudly.

## Run it

```bash
pip install -r requirements.txt
pytest -q
```
