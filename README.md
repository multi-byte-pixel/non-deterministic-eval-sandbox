# non-deterministic-eval-sandbox

Case studies in testing LLM systems, and in reviewing the code AI coding agents write for those systems.

Every case follows the same arc. An agent-written change made CI green (or mostly green), but the change was wrong in a way that is typical of evaluation code for non-deterministic systems. Each case keeps the code as it was found, the investigation, the root cause, and a fix merged as its own pull request with tests that would have caught it.

## Cases

| # | Case | Stack | Failure | Why CI didn't catch it | Fix |
|---|---|---|---|---|---|
| 1 | [Test gaming in a RAG faithfulness suite](cases/01-test-gaming/) | DeepEval, pytest | A failing faithfulness test was "fixed" by lowering the threshold and mocking the retriever | The test stopped testing the thing that broke | [#1](../../pull/1) |
| 2 | [A NaN that passed the eval gate](cases/02-silent-nan-gate/) | LangChain (LCEL), Ragas, LangSmith | A hallucinated keyword argument raised inside `try/except`, which returned `NaN` | `NaN < 0.8` is `False`, so the gate passed | [#2](../../pull/2) |
| 3 | [Exact-match asserts on Bedrock output](cases/03-flaky-bedrock-asserts/) | Amazon Bedrock (Converse), boto3, IAM, GitHub Actions | Exact-string asserts assuming `temperature=0` is deterministic, plus a `bedrock:*` on `*` IAM policy | Passes most runs, fails about 1 in 8 | [#3](../../pull/3) |
| 4 | [Pairwise judge position bias](cases/04-judge-position-bias/) | Pairwise LLM-as-judge | The release gate always showed the candidate first to a position-biased judge | Every test asserted the current release passes | [#7](../../pull/7) |
| 5 | [Eval-set contamination](cases/05-eval-contamination/) | Few-shot prompting | Eval tickets were added to the few-shot pool, so the model copied their labels | Accuracy rose from 0.63 to 1.00 and became the new threshold | [#8](../../pull/8) |
| 6 | [A better average that hid a safety regression](cases/06-slice-regression/) | Slice-level release criteria | Gate on overall pass rate while safety refusals fell from 0.95 to 0.60 | The mean went up | [#9](../../pull/9) |
| 7 | [A "robust" JSON parser](cases/07-lenient-json-parser/) | Structured output | Parse failures fell back to defaults that matched the most common label; a renamed key was dropped silently | 40% parse failures scored as 85% accuracy | [#10](../../pull/10) |
| 8 | [A stale response cache](cases/08-stale-eval-cache/) | Response caching | Cache keyed on prompt only, so a model switch kept grading the old model's answers | Zero model calls looked like an efficient CI run | [#11](../../pull/11) |

## Repository layout

```
cases/NN-name/        one self-contained case (own requirements.txt, tests, README case study)
scripts/              repo-level CI helpers (protected eval files check)
.github/workflows/    offline CI for every case; live-model workflows are manual/scheduled
```

## Reading a case

1. Start with the case's `README.md`: context, symptom, root cause, fix, lessons.
2. The `as-found` tag marks the original state of cases 1–3. Cases 4–8 landed together as found in [#6](../../pull/6), then each fix merged as its own pull request.
3. Fix pull request descriptions go deeper into the libraries involved and the reasoning behind each change.

## Running locally

Each case is independent:

```bash
cd cases/01-test-gaming          # or any other case
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Everything runs offline by default, with deterministic or simulated models, so CI needs no API keys. Cases that support a live mode document their environment variables in their own README.
