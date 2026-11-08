"""Deterministic tests of OUR code around the model, with a stub client.

Layering for LLM features (most tests at the bottom, fewest at the top):

    ┌──────────────────────────────────────────────┐
    │ quality evals: N runs, LLM judge, pass rate  │  test_summarizer_quality.py
    ├──────────────────────────────────────────────┤
    │ properties every real output must satisfy    │  test_summarizer_quality.py
    ├──────────────────────────────────────────────┤
    │ contract tests of our request/parse code     │  THIS FILE (stubbed model)
    └──────────────────────────────────────────────┘

Here the model is replaced by a stub that returns hand-written responses.
Stubbing is right in THIS layer: we're testing request construction and
response parsing, which are deterministic code, so the tests must be
deterministic too. (Contrast with example 1, where stubbing the retriever
was wrong because the retriever was the thing under test.)
"""

from __future__ import annotations

import pytest

from app.bedrock_client import JUDGE_MODEL_ID, MODEL_ID
from app.schemas import TicketSummary
from app.summarizer import TOOL_NAME, SummaryFormatError, build_request, parse_response, summarize_ticket
from evals.judge import PASS_SCORE, JudgeError, grade_summary

VALID_INPUT = {
    "category": "account_access",
    "priority": "high",
    "summary": "Customer locked out after password reset; requests unlock.",
    "action_items": ["Unlock the account"],
}


def tool_response(name: str, tool_input: dict) -> dict:
    return {"output": {"message": {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": name, "input": tool_input}}]}}}


class StubClient:
    """Returns one canned response and records the request."""

    def __init__(self, response: dict) -> None:
        self.response = response
        self.last_request: dict | None = None

    def converse(self, **kwargs):
        self.last_request = kwargs
        return self.response


# --------------------------- request construction ---------------------------
def test_request_forces_the_summary_tool():
    req = build_request("ticket")
    assert req["modelId"] == MODEL_ID
    assert req["toolConfig"]["toolChoice"] == {"tool": {"name": TOOL_NAME}}
    schema = req["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]
    # The schema sent to the model is generated from the same pydantic class
    # we validate with: one source of truth.
    assert schema == TicketSummary.model_json_schema()
    assert set(schema["required"]) == {"category", "priority", "summary", "action_items"}


def test_request_uses_low_temperature():
    """Low temperature reduces variance. (It does NOT give determinism; see summarizer.py.)"""
    assert build_request("ticket")["inferenceConfig"]["temperature"] == 0


# ----------------------------- response parsing -----------------------------
def test_parses_valid_tool_call():
    client = StubClient(tool_response(TOOL_NAME, VALID_INPUT))
    result = summarize_ticket("ticket", client)
    assert result.category == "account_access" and result.priority == "high"


@pytest.mark.parametrize(
    "bad_input, why",
    [
        ({**VALID_INPUT, "category": "login"}, "category not in the enum"),
        ({**VALID_INPUT, "priority": "P1"}, "priority not in the enum"),
        ({k: v for k, v in VALID_INPUT.items() if k != "summary"}, "missing field"),
        ({**VALID_INPUT, "summary": "x" * 500}, "summary too long"),
        ({**VALID_INPUT, "action_items": []}, "no action items"),
    ],
)
def test_schema_violations_raise(bad_input, why):
    with pytest.raises(SummaryFormatError):
        parse_response(tool_response(TOOL_NAME, bad_input))


def test_prose_instead_of_tool_call_raises():
    """What the old summarizer returned. Now it's a format error, not a string."""
    prose = {"output": {"message": {"role": "assistant", "content": [{"text": "Customer can't log in."}]}}}
    with pytest.raises(SummaryFormatError, match="expected exactly one"):
        parse_response(prose)


def test_wrong_tool_name_raises():
    with pytest.raises(SummaryFormatError):
        parse_response(tool_response("some_other_tool", VALID_INPUT))


# ----------------------------------- judge ----------------------------------
SUMMARY = TicketSummary.model_validate(VALID_INPUT)


def test_judge_request_uses_judge_model_and_sees_the_ticket():
    client = StubClient(tool_response("record_grade", {"reasoning": "ok", "score": 5}))
    grade_summary("THE TICKET TEXT", SUMMARY, client)
    assert client.last_request["modelId"] == JUDGE_MODEL_ID != MODEL_ID
    assert "THE TICKET TEXT" in client.last_request["messages"][0]["content"][0]["text"]


def test_judge_rejects_known_bad_summary():
    """Plumbing for judge calibration: a low grade must come out as a fail."""
    grade = grade_summary("t", SUMMARY, StubClient(tool_response("record_grade", {"reasoning": "missing lock", "score": 2})))
    assert grade.score < PASS_SCORE


@pytest.mark.parametrize("bad", [{"reasoning": "x", "score": 7}, {"score": 5}, {"reasoning": "x", "score": "great"}])
def test_invalid_grades_raise(bad):
    with pytest.raises(JudgeError):
        grade_summary("t", SUMMARY, StubClient(tool_response("record_grade", bad)))
