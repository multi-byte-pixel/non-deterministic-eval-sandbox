# Case 3: Exact-match tests on LLM output, and an over-broad IAM policy

**Stack:** Amazon Bedrock (Converse API), boto3, pydantic, pytest, IAM, GitHub Actions (OIDC).
**Agent:** Amazon Q Developer in PyCharm.

## The story

The agent built a ticket summarizer and its CI pipeline. It:

1. asserted `summary == "Customer reports login failure after password reset; requests account unlock."`, with the comment *"temperature=0 makes the output deterministic"*;
2. wrote an IAM policy with `"Action": "bedrock:*", "Resource": "*"`, used through long-lived access keys stored as GitHub secrets.

Tests passed locally and then failed about **1 run in 8** in CI.

## How it was caught

- **Running the test 25 times** to *measure* the flake rate instead of re-running until green (`pytest --count=25`, or `scripts/flake_rate.py`).
- **IAM Access Analyzer** `validate-policy` on the policy, and a least-privilege review.

## The fix (PR #3)

| Problem | Fix | Where |
|---|---|---|
| Exact text asserted on a sample from a distribution | Structured output (forced tool use + pydantic); exact asserts only on enum fields; properties for text | `app/schemas.py`, `app/summarizer.py` |
| "Is it good?" can't be string-matched | Rubric LLM-as-judge with a different, stronger model | `evals/judge.py` |
| One sample decided pass/fail | Pass rate ≥ 90% over N=30 runs, with a Wilson interval and failure reasons | `evals/stats.py`, `tests/test_summarizer_quality.py` |
| Gate flakiness unknown | Meta-tests that compute false-alarm and miss probabilities | `test_gate_itself_is_not_flaky_against_the_fake`, `test_gate_can_detect_a_real_regression` |
| Request/parse logic untested | Deterministic contract tests with a stub client | `tests/test_summarizer_contract.py` |
| `bedrock:*` on `*` | `bedrock:InvokeModel` on two inference profiles + their routed foundation models | `infra/iam/` |
| Long-lived keys | GitHub OIDC → short-lived role credentials, trust scoped to `main` | `infra/iam/github-oidc-trust-policy.json`, workflow |
| No policy checks | Offline policy lint in CI, Access Analyzer in the live workflow | `tests/test_iam_policy.py`, `.github/workflows/bedrock-evals.yml` |

## Run it

```bash
pip install -r requirements.txt
pytest -q                                                    # offline, fake Bedrock (non-deterministic on purpose)
python scripts/flake_rate.py tests/test_summarizer_quality.py::test_summarizer_pass_rate -n 100
BEDROCK_MODE=live AWS_PROFILE=... pytest -q -s               # real Bedrock (needs model access; costs money)
```
