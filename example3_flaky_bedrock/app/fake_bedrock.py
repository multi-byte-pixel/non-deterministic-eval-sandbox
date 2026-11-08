"""Offline stand-in for the boto3 ``bedrock-runtime`` client.

Implements ``converse()`` with the real Converse response shape, including
forced tool use (``toolConfig`` / ``toolUse`` blocks).

It is deliberately NON-deterministic (``random.SystemRandom`` cannot be
seeded). A fake that always returns the same string teaches your tests that
the model is deterministic, and the tests then break against the real thing.
This one behaves like a well-behaved real model:

* the summary WORDING varies from call to call (it did before PR #3 too, so
  the exact-match test failed about 1 run in 8);
* category/priority are stable, as a good model's would be on a clear-cut
  ticket;
* the judge (record_grade tool) usually gives 4-5, and occasionally
  gives a low score (1%), like real LLM-judge noise. The pass-rate gate in
  tests/test_summarizer_quality.py must tolerate that, and it does (see the
  maths there).

Changing these rates changes how often the offline CI fails. They are
documented constants for that reason, not magic numbers in the code.
"""

from __future__ import annotations

import random
import uuid

_rng = random.SystemRandom()  # unseedable on purpose; see module docstring

CANONICAL = "Customer reports login failure after password reset; requests account unlock."
PARAPHRASES = [
    "Customer cannot log in after resetting their password and asks for the account to be unlocked.",
    "Login fails following a password reset; the customer requests an account unlock.",
]
VARIATION_RATE = 1 / 8          # P(summary uses a paraphrase)
JUDGE_LOW_SCORE_RATE = 0.01     # P(judge gives a spuriously low grade)


class FakeBedrockRuntime:
    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[dict] = []  # recorded for assertions in tests

    def converse(self, *, modelId: str, messages: list, system: list | None = None,
                 inferenceConfig: dict | None = None, toolConfig: dict | None = None, **_):
        self.calls += 1
        self.requests.append({"modelId": modelId, "toolConfig": toolConfig, "inferenceConfig": inferenceConfig})

        forced = ((toolConfig or {}).get("toolChoice") or {}).get("tool", {}).get("name")
        if forced == "record_ticket_summary":
            return _tool_response(forced, _fake_summary())
        if forced == "record_grade":
            return _tool_response(forced, _fake_grade())
        # No tool forced: plain text, like the pre-PR-#3 summarizer expected.
        return _text_response(_summary_text())


def _summary_text() -> str:
    return _rng.choice(PARAPHRASES) if _rng.random() < VARIATION_RATE else CANONICAL


def _fake_summary() -> dict:
    return {
        "category": "account_access",
        "priority": "high",
        "summary": _summary_text(),
        "action_items": _rng.choice(
            [
                ["Verify the customer's identity", "Unlock the account"],
                ["Unlock the account", "Confirm the customer can log in"],
            ]
        ),
    }


def _fake_grade() -> dict:
    if _rng.random() < JUDGE_LOW_SCORE_RATE:
        return {"score": 2, "reasoning": "Summary omits the account lock."}
    return {"score": _rng.choice([4, 5, 5]), "reasoning": "Accurate, concise, actionable."}


def _tool_response(name: str, tool_input: dict) -> dict:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": f"tooluse_{uuid.uuid4().hex[:12]}", "name": name, "input": tool_input}}],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 350, "outputTokens": 60, "totalTokens": 410},
        "metrics": {"latencyMs": 600},
    }


def _text_response(text: str) -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 120, "outputTokens": 25, "totalTokens": 145},
        "metrics": {"latencyMs": 400},
    }
