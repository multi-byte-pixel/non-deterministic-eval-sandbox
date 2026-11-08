# Case 8: A response cache that kept grading the old model

**Stack:** Python, pytest, LLM response caching.

## Context

The docs-assistant eval asks ten product questions and needs 0.90 accuracy. To make CI fast and free, a coding agent added a response cache keyed on the prompt and committed a warm cache file to the repo. Later, `config.yaml` was switched from `assistant-v1` to the cheaper `assistant-v2`.

## Symptom

Accuracy stayed at 1.00 and v2 went to production. Support then reported wrong answers on refund timing, rate limits and log retention.

## Root cause

The cache key was `sha256(prompt)`. Changing the model did not change any prompt, so every lookup returned the answers v1 had produced. The eval graded v1 and labelled it v2. One test even asserted that a warm cache made zero model calls, which was the bug, written as a feature.

## How it was caught

The provider's usage dashboard showed no v2 calls from CI in the week of the switch.

## Fix

| Layer | Change | Where |
|---|---|---|
| Cache key | Prompt, model id, temperature and template version are all in the key | `harness/cache.py` |
| Provenance | Each entry records the model that produced it; serving it for another model raises `StaleCacheError` | `harness/cache.py` |
| Visibility | The runner reports model calls and cache hits with every result | `harness/runner.py` |
| Release check | The configured model is evaluated with a fresh cache and must be called | `tests/test_configured_model.py` |
| Rollback | `config.yaml` back on assistant-v1; v2 scores 0.60 with real calls | `config.yaml` |
| Clean-up | The committed v1 cache file is removed | `data/` |

## Lessons

* Every input that can change a model output belongs in its cache key.
* An eval that makes zero model calls is not evaluating a model. Report call counts with scores.
* Caches speed up repeated runs of the same configuration. A release decision deserves fresh calls.

## Run it

```bash
pip install -r requirements.txt
pytest -q
```
