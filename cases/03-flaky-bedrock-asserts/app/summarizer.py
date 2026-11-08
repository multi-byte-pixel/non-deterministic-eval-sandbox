"""Support-ticket summarizer backed by Amazon Bedrock (Converse API).

Background for new readers
==========================
**Amazon Bedrock** hosts foundation models (Anthropic Claude, Meta Llama,
Amazon Nova, Mistral...) behind one AWS API. **Converse** is its
model-agnostic chat API: the same request shape works across providers.
You call it through boto3's ``bedrock-runtime`` client:

    client.converse(modelId=..., system=[...], messages=[...], inferenceConfig={...})

The ``modelId`` here is a **cross-region inference profile** (the ``us.``
prefix). Bedrock may route each request to a different US region for
capacity. Good for availability, and one more reason output isn't
bit-for-bit reproducible.

Why ``temperature=0`` is NOT determinism (the bug behind PR #3)
===============================================================
The old code said: "temperature=0 makes the output deterministic, so tests
can assert exact text." It doesn't, for several reasons:

* Temperature controls *sampling*. At 0 the model picks the most likely
  next token (greedy decoding), but the probabilities themselves can
  differ slightly between runs. GPU floating-point math is not associative,
  so batching requests together changes the order of additions, and
  results shift in the last decimal places. When two tokens are nearly
  tied, that tiny shift flips the choice, and every token after it differs.
* Providers batch your request with other people's traffic. You don't
  control the batch.
* Requests can land on different hardware or regions (inference profiles)
  and on model revisions that are updated behind a stable ID.
* Bedrock's Converse API has no ``seed`` parameter, and seeds don't
  guarantee identical output even where they exist.

So: treat model output as a **sample from a distribution**. Tests assert
**properties** that every acceptable sample must have, and measure
**rates** for anything fuzzier (see tests/test_summarizer_quality.py).
We keep temperature at 0 anyway: it reduces variance, which makes rate-based
gates tighter. It just doesn't remove variance.

Structured output via forced tool use
=====================================
Instead of asking for prose (or for "JSON please" in the prompt and hoping),
we define a *tool* whose input schema is ``TicketSummary``'s JSON Schema,
and use ``toolChoice`` to force the model to call it. The model's answer
arrives as a ``toolUse`` content block whose ``input`` is already a parsed
dict. This is the most reliable way to get structured output from Converse.
We still validate with pydantic, because the schema is a strong hint to the
model, not a guarantee.
"""

from __future__ import annotations

from pydantic import ValidationError

from app.bedrock_client import MODEL_ID
from app.schemas import TicketSummary

TOOL_NAME = "record_ticket_summary"

SYSTEM_PROMPT = (
    "You triage customer support tickets. Read the ticket and call the "
    f"{TOOL_NAME} tool exactly once. Never include the customer's name or contact details."
)

# Converse API tool definition. "inputSchema": {"json": <JSON Schema>} is the
# Bedrock format; pydantic generates the schema so it can't drift from the
# model class we validate against.
TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": TOOL_NAME,
                "description": "Record the structured triage summary of a support ticket.",
                "inputSchema": {"json": TicketSummary.model_json_schema()},
            }
        }
    ],
    # Force this specific tool. Without toolChoice the model may reply in
    # prose instead. ("any" forces *some* tool; {"tool": {...}} forces this one.)
    "toolChoice": {"tool": {"name": TOOL_NAME}},
}


class SummaryFormatError(ValueError):
    """The model's response could not be turned into a valid TicketSummary.

    Raised, not swallowed (see example 2). The quality eval counts it as a
    failed sample, so format problems show up in the pass rate rather than
    crashing the run or hiding.
    """


def build_request(ticket_text: str) -> dict:
    """The exact kwargs sent to ``converse``. A pure function, so it's unit-testable."""
    return {
        "modelId": MODEL_ID,
        "system": [{"text": SYSTEM_PROMPT}],
        "messages": [{"role": "user", "content": [{"text": ticket_text}]}],
        # temperature=0 reduces variance; it does NOT make output deterministic.
        "inferenceConfig": {"temperature": 0, "maxTokens": 400},
        "toolConfig": TOOL_CONFIG,
    }


def parse_response(response: dict) -> TicketSummary:
    """Extract and validate the forced tool call from a Converse response.

    Response shape (abridged):
        {"output": {"message": {"role": "assistant",
                                "content": [{"toolUse": {"toolUseId": "...",
                                                         "name": "record_ticket_summary",
                                                         "input": {...}}}]}},
         "stopReason": "tool_use", "usage": {...}}
    """
    content = response.get("output", {}).get("message", {}).get("content", [])
    tool_uses = [block["toolUse"] for block in content if "toolUse" in block]
    if len(tool_uses) != 1 or tool_uses[0].get("name") != TOOL_NAME:
        raise SummaryFormatError(f"expected exactly one {TOOL_NAME} call, got content={content!r}")
    try:
        return TicketSummary.model_validate(tool_uses[0]["input"])
    except ValidationError as exc:
        raise SummaryFormatError(f"tool input failed schema validation: {exc}") from exc


def summarize_ticket(ticket_text: str, client) -> TicketSummary:
    """Summarize one ticket. ``client`` is a boto3 bedrock-runtime client or the fake."""
    return parse_response(client.converse(**build_request(ticket_text)))
