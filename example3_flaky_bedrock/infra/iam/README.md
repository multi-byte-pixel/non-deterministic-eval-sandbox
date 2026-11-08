# IAM for the Bedrock eval pipeline

JSON can't hold comments, so the explanations for the policies live here.

## Before PR #3

```json
{ "Effect": "Allow", "Action": "bedrock:*", "Resource": "*" }
```

…attached to an IAM **user** whose long-lived access keys were stored as GitHub secrets.

`bedrock:*` on `*` lets CI (or anyone who gets those keys out of a compromised workflow) do far more than invoke a model. It covers creating and deleting guardrails, knowledge bases, agents, provisioned throughput (billed by the hour, and expensive), model customization jobs and invocation-logging configuration, in every region, against every model. Coding agents often write policies like this because it is the fastest way to make the `AccessDeniedException` go away.

It was caught with **IAM Access Analyzer policy validation** (`aws accessanalyzer validate-policy`), which flags overly permissive statements, and by reading the policy against least privilege: *what is the smallest set of actions and resources this job needs?*

## After PR #3

### `bedrock-eval-policy.json`: the role's permissions
- **Action:** only `bedrock:InvokeModel`. The Converse API is authorized by `bedrock:InvokeModel`; `ConverseStream` would need `bedrock:InvokeModelWithResponseStream`, which we don't use.
- **Resource:** only the two inference profiles we call (app model and judge model).
- **Second statement:** a cross-region inference profile (`us.…`) routes each request to a foundation model in us-east-1, us-east-2 or us-west-2, so the role also needs `InvokeModel` on those foundation-model ARNs. The `bedrock:InferenceProfileArn` condition means those models can **only** be reached through our two profiles, not invoked directly.
- `123456789012` is a placeholder account ID. Replace it with yours, and check the current model/profile IDs and routed regions in the Bedrock console, since they change as models are released.

### `github-oidc-trust-policy.json`: who may assume the role
- **No long-lived keys.** GitHub Actions requests an OIDC token, and AWS STS exchanges it for credentials that expire within the hour (`aws-actions/configure-aws-credentials`).
- `aud = sts.amazonaws.com`: the token was minted for AWS.
- `sub = repo:<owner>/<repo>:ref:refs/heads/main`: only workflows running on **main of this repo** can assume the role. Without a `sub` condition, *any* GitHub repository could assume it. That is a real, common misconfiguration.
- PRs from branches can't assume this role. Live evals run nightly from main or by manual dispatch from main.

### Checks
- `tests/test_iam_policy.py` is an offline lint that runs in normal CI: no wildcard actions, no `*` resources, only allow-listed actions, and a scoped trust policy.
- `.github/workflows/bedrock-evals.yml` runs `aws accessanalyzer validate-policy` on both files before any eval and fails on `ERROR` or `SECURITY_WARNING` findings.
  - Note: `access-analyzer:ValidatePolicy` itself does not support resource-level permissions, so the role needs it on `"Resource": "*"` in a **separate**, tiny policy. That's normal. Some AWS actions can only be granted on `*`; the goal is that *dangerous* actions are never granted that way.
