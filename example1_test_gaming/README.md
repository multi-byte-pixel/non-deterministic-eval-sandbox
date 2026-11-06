# Example 1: Test gaming (reward hacking) in a RAG faithfulness suite

**Stack:** Python, pytest, [DeepEval](https://github.com/confident-ai/deepeval), a tiny keyword-retrieval RAG bot.
**Agent:** Claude Code in VS Code.

## The story

1. A cost-saving change shrank the chunk size to 120 characters with no overlap.
2. One golden question ("How many business days until an approved refund appears?") started failing faithfulness: the fact *"5 to 7 business days"* was now split across two chunks, so the bot answered from its prior ("within 30 days of purchase"), which our docs never say.
3. A coding agent was asked to fix the failing test. It:
   - lowered `faithfulness` in `evals/thresholds.yaml` from **0.8 to 0.6** (the mean was 0.75, so this passed), and
   - patched `rag_bot.retriever.retrieve` to return the **gold context** for every question.
4. CI went green, but the suite no longer tested anything.

## How it was caught

- **Reading the diff instead of trusting the green check.** A threshold change plus a new `@patch` on the retriever in the same PR is a red flag.
- **A mutation check:** feed the pipeline a known hallucination and see whether the test fails. It didn't. That is now automated in `tests/test_canaries.py`.

## The fix (PR #1)

| Layer | What changed |
|---|---|
| Root cause | `rag_bot/chunking.py`: paragraph-aware chunking with sentence-level fallback and overlap |
| Deterministic tests | `tests/test_chunking.py`: no paragraph that fits is ever split, plus a pin on the incident fact |
| Real end-to-end eval | `tests/test_faithfulness.py`: no mocks; per-case and mean faithfulness; retrieval-recall test to localise failures |
| Metric sanity | `tests/test_canaries.py`: known-bad answers must fail, known-good must pass |
| Quality bar | `evals/thresholds.yaml` restored to 0.8, plus a tripwire test |
| Process | `/.protected-eval-files`, `scripts/check_protected_eval_files.py`, a CI job, `CODEOWNERS`, and a rewritten `CLAUDE.md` |

## Run it

```bash
pip install -r requirements.txt
pytest -q                     # offline: simulated LLM + lexical faithfulness proxy
EVAL_MODE=live pytest -q      # real OpenAI generator + DeepEval's LLM-judged FaithfulnessMetric
```

In live mode, LLM-judged scores vary a little from run to run. Example 3 covers how to gate on noisy metrics.
