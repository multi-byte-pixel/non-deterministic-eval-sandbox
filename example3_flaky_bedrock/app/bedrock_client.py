"""Factory for the Bedrock runtime client.

    BEDROCK_MODE=live -> real boto3 ``bedrock-runtime`` client
                         (needs AWS credentials and model access in the account)
    anything else     -> FakeBedrockRuntime (offline, free, non-deterministic on purpose)

Credentials in CI
=================
boto3 finds credentials through its standard chain (env vars, ~/.aws, an
assumed role...). In GitHub Actions we no longer store long-lived access keys
as secrets. The workflow uses **OIDC**: GitHub issues a short-lived signed
token for this workflow run, AWS STS exchanges it for temporary credentials
for a narrowly scoped IAM role, and those credentials expire within an hour.
See .github/workflows/bedrock-evals.yml and infra/iam/.
"""

from __future__ import annotations

import os

# App model: small, fast, cheap. The "us." prefix makes it a cross-region
# inference profile ID (see app/summarizer.py).
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")

# Judge model: deliberately a different, stronger model than the app model.
# A model grading its own output tends to be lenient with its own mistakes
# ("self-preference bias").
JUDGE_MODEL_ID = os.getenv("BEDROCK_JUDGE_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")


def get_client():
    if os.getenv("BEDROCK_MODE") == "live":
        import boto3
        from botocore.config import Config

        # Adaptive retries back off on throttling (ThrottlingException is
        # common on Bedrock under load). Retrying infrastructure errors is
        # fine. Retrying a *test* until it passes (pytest-rerunfailures) is
        # not: that hides the flakiness instead of measuring it.
        return boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            config=Config(retries={"max_attempts": 6, "mode": "adaptive"}),
        )
    from app.fake_bedrock import FakeBedrockRuntime

    return FakeBedrockRuntime()
