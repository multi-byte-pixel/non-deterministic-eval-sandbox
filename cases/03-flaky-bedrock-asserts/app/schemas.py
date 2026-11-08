"""Structured output schema for ticket summaries.

Why structured output?
======================
The old summarizer returned one free-text sentence, and the test compared
it with ``==``. That was never going to be stable: there are many correct
ways to word a summary, and a model will use different ones from run to
run (see the module docstring of app/summarizer.py for why temperature=0
doesn't stop this).

The fix separates what a summary contains into two kinds of field:

* **Fields that should be stable** (``category``, ``priority``): small,
  closed sets (enums). We CAN assert exact values on these, and
  disagreement here is a real bug (a login problem triaged as "billing").
* **Fields that legitimately vary** (``summary``, ``action_items``): free
  text. We do NOT assert exact text. We check properties (length, key
  terms, no PII) and grade quality with a rubric-based LLM judge (see
  evals/judge.py).

Pydantic gives us:
  * runtime validation: ``TicketSummary.model_validate(data)`` raises
    ``ValidationError`` if the model returns an unknown category, a missing
    field, a 2,000-character summary...
  * a JSON Schema (``TicketSummary.model_json_schema()``) that we send to
    Bedrock as the tool input schema, so the model is *told* the exact shape
    we will validate. One source of truth for both sides.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Literal[...] becomes an "enum" in the JSON Schema and is enforced by pydantic.
Category = Literal["account_access", "billing", "bug", "feature_request", "other"]
Priority = Literal["low", "medium", "high", "urgent"]


class TicketSummary(BaseModel):
    category: Category = Field(description="Which queue should handle this ticket.")
    priority: Priority = Field(description="Urgency. Customer locked out of their account = high.")
    summary: str = Field(
        min_length=10,
        max_length=200,
        description="One sentence for the triage queue. Do not include the customer's name.",
    )
    action_items: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Concrete next steps for the support agent.",
    )
