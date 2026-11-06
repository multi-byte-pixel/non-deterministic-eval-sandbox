"""Offline stand-in for the boto3 ``bedrock-runtime`` client.

Implements just ``converse()`` and returns the same response shape as the
real Bedrock Converse API. It is deliberately non-deterministic: real models
can still vary their wording at temperature 0 (batching, floating-point
non-associativity on GPUs, model/version routing behind inference profiles),
so a realistic fake should too.
"""

from __future__ import annotations

import random

_rng = random.SystemRandom()  # unseeded on purpose

CANONICAL = "Customer reports login failure after password reset; requests account unlock."
PARAPHRASES = [
    "Customer cannot log in after resetting their password and asks for the account to be unlocked.",
    "Login fails following a password reset; the customer requests an account unlock.",
]
VARIATION_RATE = 1 / 8


class FakeBedrockRuntime:
    def __init__(self) -> None:
        self.calls = 0

    def converse(self, *, modelId: str, messages: list, system: list | None = None, inferenceConfig: dict | None = None, **_):
        self.calls += 1
        text = _rng.choice(PARAPHRASES) if _rng.random() < VARIATION_RATE else CANONICAL
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 120, "outputTokens": 25, "totalTokens": 145},
            "metrics": {"latencyMs": 400},
        }
