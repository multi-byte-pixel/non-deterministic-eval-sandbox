# Case 2: A hallucinated API call that became a NaN that passed CI

**Stack:** LangChain (LCEL), LangSmith, in-repo LLM-as-judge faithfulness metric. The incident happened on [Ragas](https://github.com/explodinggradients/ragas), which was later removed (see below).
**Agent:** GitHub Copilot agent mode in VS Code.

## The story

While refactoring the eval script, the coding agent wrote:

```python
try:
    result = evaluate(dataset, metrics=[Faithfulness()], llm_judge=judge_llm)  # no such kwarg
    ...
except Exception as exc:
    logger.debug(...)                       # invisible in CI
    return {"faithfulness": float("nan")}
```

and the gate checked `score < threshold`. Because **`nan < 0.8` is `False`**, the gate passed. On a fresh install with unpinned dependencies, the same `except` also swallowed an `ImportError` (ragas 0.4.3 is incompatible with langchain-community 0.4.x).

## How it was caught

The LangSmith trace for a 200-example eval run showed **zero judge LLM calls**. You can't score 200 answers without calling the judge.

## The fix (PR #2)

| Problem | Fix | Test |
|---|---|---|
| Hallucinated `llm_judge=` kwarg | Correct kwarg; kwargs built in one function | `test_every_kwarg_we_pass_exists_in_the_scorer` (signature contract) |
| Unpinned deps broke the import | `==` pins, top-level imports, Dependabot | `test_direct_dependencies_are_pinned` |
| `except Exception` returned NaN | Removed; unparsable judge output raises | `test_broken_judge_output_raises_instead_of_nan`, `test_malformed_*_output_raises` |
| Judge never called | `JudgeCallCounter` callback; raise if calls < samples | `test_harness_errors_when_judge_is_never_called` |
| NaN passed the gate | Fail-closed gate (`missing`, NaN, inf, None all fail) | `test_gate_is_fail_closed[*]` |
| Gate never shown to fail | Negative end-to-end test | `test_eval_gate_fails_for_unfaithful_answers` |

## Dependency change: Ragas removed

Ragas 0.2.3 through 0.4.3 is affected by [GHSA-95ww-475f-pr4f](https://github.com/advisories/GHSA-95ww-475f-pr4f) (SSRF), and no patched release was listed when it was removed. [`evals/faithfulness.py`](evals/faithfulness.py) now implements the same two-step metric (extract statements, then judge each against the context) on any LangChain chat model, with fail-closed parsing. The harness checks and tests are unchanged in intent.

If you prefer a maintained library, [DeepEval's `FaithfulnessMetric`](https://deepeval.com/docs/metrics-faithfulness) computes the same ratio. Wiring it in would mean replacing `score_samples` and re-running the contract tests.

## Run it

```bash
pip install -r requirements.txt
pytest -q                                  # offline: scripted judge, real scoring pipeline
OPENAI_API_KEY=... LANGSMITH_TRACING=true LANGSMITH_API_KEY=... \
  python -m evals.run_faithfulness_eval    # live: exit 0 pass / 1 quality fail / 2 harness error
```
