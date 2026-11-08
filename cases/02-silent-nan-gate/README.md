# Case 2: A hallucinated API call that became a NaN that passed CI

**Stack:** LangChain (LCEL), [Ragas](https://github.com/explodinggradients/ragas), LangSmith.
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
| Hallucinated `llm_judge=` kwarg | Use `llm=LangchainLLMWrapper(judge)`; kwargs built in one function | `test_every_kwarg_we_pass_exists_in_ragas_evaluate` (signature contract) |
| Unpinned deps broke the import | `==` pins, top-level imports, Dependabot | `test_direct_dependencies_are_pinned` |
| `except Exception` returned NaN | Removed; `raise_exceptions=True` | `test_broken_judge_output_raises_instead_of_nan` |
| Judge never called | `JudgeCallCounter` callback; raise if calls < samples | `test_harness_errors_when_judge_is_never_called` |
| NaN passed the gate | Fail-closed gate (`missing`, NaN, inf, None all fail) | `test_gate_is_fail_closed[*]` |
| Gate never shown to fail | Negative end-to-end test | `test_eval_gate_fails_for_unfaithful_answers` |

## Run it

```bash
pip install -r requirements.txt
pytest -q                                  # offline: scripted judge, real Ragas pipeline
OPENAI_API_KEY=... LANGSMITH_TRACING=true LANGSMITH_API_KEY=... \
  python -m evals.run_ragas_eval           # live: exit 0 pass / 1 quality fail / 2 harness error
```
