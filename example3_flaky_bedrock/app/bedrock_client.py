"""Factory for the Bedrock runtime client.

BEDROCK_MODE=live -> real boto3 client (needs AWS credentials + model access)
anything else     -> FakeBedrockRuntime (offline)
"""

from __future__ import annotations

import os

MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")


def get_client():
    if os.getenv("BEDROCK_MODE") == "live":
        import boto3

        return boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
    from app.fake_bedrock import FakeBedrockRuntime

    return FakeBedrockRuntime()
