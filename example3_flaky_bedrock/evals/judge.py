"""Rubric-based LLM-as-judge for ticket summaries, via Bedrock Converse.

Why an LLM judge?
=================
"Is this a good summary?" has many right answers, so string comparison
can't grade it. A strong model can, if you give it a **rubric**: explicit
criteria and a scoring scale. A vague prompt ("rate this 1-10") gives
vague, drifting scores. A rubric with named criteria and anchored levels
gives scores that are far more consistent and that you can argue with.

Good-practice checklist used here:
  * A different, stronger model than the one being graded (JUDGE_MODEL_ID).
  * A small integer scale (1-5) with every level described. Wide scales
    (1-100) invite false precision.
  * Reasoning BEFORE the score in the output schema, so the model commits
    to an analysis before choosing a number.
  * Forced tool use for structured output, validated with pydantic, same
    as the summarizer.
  * The judge sees the SOURCE ticket, so it can check accuracy, not just
    fluency.
  * Binary pass/fail derived from the score (>= PASS_SCORE), because pass
    rates are what the gate aggregates.

Judges are models too, so they are noisy and can be wrong. Calibrate
yours: grade a few human-labelled good and bad summaries and check the
judge agrees (example 1 calls these "canaries";
test_judge_rejects_known_bad_summary is the offline plumbing for that idea).
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from app.bedrock_client import JUDGE_MODEL_ID
from app.schemas import TicketSummary

TOOL_NAME = "record_grade"
PASS_SCORE = 4

RUBRIC = """You are grading a support-ticket triage summary.

Criteria:
- ACCURATE: every statement in the summary is supported by the ticket.
- COMPLETE: the summary captures the customer's problem AND what they are asking for.
- ACTIONABLE: the action items are concrete next steps for a support agent.
- PRIVATE: no customer name, email or phone number appears.

Scale:
5 = meets all criteria.
4 = meets all criteria with a minor wording issue.
3 = one criterion partly missed (e.g. the request is vague).
2 = one criterion clearly failed (e.g. the main problem is missing or wrong).
1 = misleading, or contains private details.

Write your reasoning first, then the score. Call the record_grade tool."""


class Grade(BaseModel):
    reasoning: str = Field(description="Brief justification, written before deciding the score.")
    score: int = Field(ge=1, le=5, description="Integer score from the rubric scale.")


TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": TOOL_NAME,
                "description": "Record the rubric grade for the summary.",
                "inputSchema": {"json": Grade.model_json_schema()},
            }
        }
    ],
    "toolChoice": {"tool": {"name": TOOL_NAME}},
}


class JudgeError(ValueError):
    """The judge didn't return a valid grade. Counted as a failed sample, never as a pass."""


def grade_summary(ticket_text: str, summary: TicketSummary, client) -> Grade:
    response = client.converse(
        modelId=JUDGE_MODEL_ID,
        system=[{"text": RUBRIC}],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": f"<ticket>\n{ticket_text}\n</ticket>\n\n"
                        f"<summary>\n{json.dumps(summary.model_dump(), indent=2)}\n</summary>"
                    }
                ],
            }
        ],
        inferenceConfig={"temperature": 0, "maxTokens": 400},
        toolConfig=TOOL_CONFIG,
    )
    content = response.get("output", {}).get("message", {}).get("content", [])
    tool_uses = [b["toolUse"] for b in content if "toolUse" in b and b["toolUse"].get("name") == TOOL_NAME]
    if len(tool_uses) != 1:
        raise JudgeError(f"judge did not call {TOOL_NAME}: {content!r}")
    try:
        return Grade.model_validate(tool_uses[0]["input"])
    except ValidationError as exc:
        raise JudgeError(f"invalid grade: {exc}") from exc
