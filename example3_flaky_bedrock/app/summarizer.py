"""Support-ticket summarizer backed by Amazon Bedrock (Converse API)."""

from __future__ import annotations

from app.bedrock_client import MODEL_ID

SYSTEM_PROMPT = "Summarize the support ticket in one sentence for the triage queue."


def summarize_ticket(ticket_text: str, client) -> str:
    response = client.converse(
        modelId=MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[{"role": "user", "content": [{"text": ticket_text}]}],
        # temperature=0 makes the output deterministic, so tests can assert exact text.
        inferenceConfig={"temperature": 0, "maxTokens": 200},
    )
    return response["output"]["message"]["content"][0]["text"].strip()
