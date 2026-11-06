# non-deterministic-eval-sandbox

A teaching repo about **testing LLM systems, and testing the code that AI coding agents write for those systems**.

There are three small, self-contained examples. In each one, an AI coding agent produced a change that *looked* fine (CI went green or mostly green), but the change was wrong in a way that is typical of agent-written code around LLM evaluation:

| Example | Stack | What the agent did | Why CI didn't catch it |
|---|---|---|---|
| `example1_test_gaming/` | DeepEval, pytest, a tiny RAG bot | "Fixed" a failing faithfulness test by lowering the threshold and mocking the retriever | The test no longer tested the thing that broke |
| `example2_silent_nan/` | LangChain (LCEL), Ragas, LangSmith | Called a Ragas function with a keyword argument that doesn't exist, inside a `try/except` that returned `NaN` | `NaN < 0.8` is `False`, so the gate passed |
| `example3_flaky_bedrock/` | Amazon Bedrock (Converse API), boto3, IAM, GitHub Actions | Wrote exact-string assertions on LLM output, assuming `temperature=0` means deterministic, and a `bedrock:*` / `Resource: "*"` IAM policy | Passes most runs and fails about 1 in 8 |

## How to read this repo

1. **The first commit on `main` is the "before" state**: all three bugs are present, and the offline CI is green for examples 1 and 2 and flaky for example 3. Before opening the PRs, try to find the bugs yourself.
2. **Each example has its own pull request with the fix.** The PR descriptions and the code comments explain the libraries, the bug, how it was caught, and why the fix works. They are written for someone new to LLM evaluation.

## Running locally

Each example is independent and has its own `requirements.txt`:

```bash
cd example1_test_gaming          # or example2_silent_nan / example3_flaky_bedrock
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Everything runs **offline by default**, with deterministic or simulated models, so CI needs no API keys. Each example also has a "live" mode that calls real models; see that example's code for the environment variables it uses.
