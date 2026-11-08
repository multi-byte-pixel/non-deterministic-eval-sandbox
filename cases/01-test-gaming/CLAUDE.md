# Agent instructions: cases/01-test-gaming

This file is read automatically by Claude Code (and most other coding
agents read a similar file: AGENTS.md, .cursor/rules, copilot-instructions.md).
It is the cheapest guardrail we have: plain-language rules the agent sees
before it writes any code.

## Background (why these rules exist)

The previous version of this file said "Keep CI green. If a test fails, fix
it before finishing." An agent followed that literally: it lowered the
faithfulness threshold and mocked the retriever until the test passed. The
instruction rewarded *green CI*, not *working software*. The rules below
reward the right thing.

## Rules

1. **Eval thresholds and golden data are read-only for you.**
   Never edit `evals/thresholds.yaml`, anything in `data/golden/`, or any
   path listed in `/.protected-eval-files`. If you think one is wrong, stop
   and explain why in your summary. A human decides.

2. **A failing eval is a finding, not an obstacle.**
   When an eval or quality test fails, your job is to find out *why*. Report
   the root cause (which case, which component, what changed) before
   proposing a fix. "Make the test pass" is never the goal on its own.

3. **Never mock the component under test.**
   In `tests/test_faithfulness.py`, retrieval, chunking and generation are
   all under test. Do not patch `rag_bot.retriever`, `rag_bot.chunking` or
   `rag_bot.generator` there. Mocking external services in *unit* tests is fine.

4. **Do not weaken assertions to get green.**
   No lowering thresholds, widening tolerances, adding `skip`/`xfail`,
   catching and ignoring exceptions, or deleting test cases, unless a human
   explicitly asked for that specific change.

5. **The canaries must stay red-able.**
   `tests/test_canaries.py` checks that the metric can still fail. If a
   negative canary starts passing, the eval harness is broken. Treat that as
   the most urgent failure in the repo.

6. **Verify before you claim.**
   Run `pytest -q` from this directory and paste the result. If you
   changed chunking or retrieval, include the per-case scores from
   `test_mean_faithfulness_meets_threshold`.

## Commands

    pip install -r requirements.txt
    pytest -q                       # offline, deterministic, ~1s
    EVAL_MODE=live pytest -q        # real LLM generator + DeepEval LLM judge (needs OPENAI_API_KEY)
