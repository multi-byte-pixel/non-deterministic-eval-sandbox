"""Faithfulness metric: an LLM-as-judge scorer kept small enough to read.

This replaces the Ragas ``Faithfulness`` metric this case originally used.
Ragas was removed because GHSA-95ww-475f-pr4f (SSRF, ragas >= 0.2.3,
<= 0.4.3) had no patched release when this change was made. See SECURITY.md.

The algorithm is the usual one (Ragas and DeepEval both use a variant):

    1. Ask the judge to break the answer into standalone statements.
    2. Ask the judge, for each statement, whether the retrieved context
       supports it (verdict 1) or not (verdict 0).
    score = supported statements / total statements

Two judge calls per sample. The judge is any LangChain chat model
(ChatOpenAI, ChatBedrock, a test fake), so callbacks and LangSmith tracing
work the same way they did before.

Fail-closed parsing
===================
The incident in this case was a harness that turned errors into NaN. So this
scorer never returns a number it did not compute:

* Judge output that is not the expected JSON raises ``JudgeOutputError``.
* Zero statements raises (0/0 has no meaningful score).
* A verdict list whose length does not match the statements raises.
* A verdict other than 0 or 1 raises.

There is no ``try/except Exception`` anywhere in this module.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

METRIC_NAME = "faithfulness"


class JudgeOutputError(ValueError):
    """The judge replied, but not in a form we can score. Never a score."""


_STATEMENTS_SYSTEM = (
    "Break the answer into short, standalone factual statements. "
    'Reply with JSON only, in this shape: {"statements": ["...", "..."]}'
)

_VERDICTS_SYSTEM = (
    "For each statement, decide whether the context supports it. "
    "Use verdict 1 if the context supports it and verdict 0 if it does not. "
    'Reply with JSON only, in this shape: {"statements": '
    '[{"statement": "...", "reason": "...", "verdict": 1}]}'
)

# Real models often wrap JSON in a ```json fence. Accept that one wrapper and
# nothing looser: a lenient parser is the bug in case 07.
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _parse_json(content: Any) -> dict:
    text = str(content).strip()
    match = _FENCE.match(text)
    if match:
        text = match.group(1)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:  # narrow: only the parse failure
        raise JudgeOutputError(f"Judge output is not JSON: {text[:80]!r}") from exc
    if not isinstance(data, dict):
        raise JudgeOutputError(f"Judge output is not a JSON object: {text[:80]!r}")
    return data


def _ask(judge: BaseChatModel, system: str, user: str, callbacks: list[BaseCallbackHandler]) -> dict:
    reply = judge.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)],
        config={"callbacks": callbacks},
    )
    return _parse_json(reply.content)


def extract_statements(sample: dict, judge: BaseChatModel, callbacks: list[BaseCallbackHandler]) -> list[str]:
    data = _ask(
        judge,
        _STATEMENTS_SYSTEM,
        f"Question: {sample['user_input']}\nAnswer: {sample['response']}",
        callbacks,
    )
    statements = data.get("statements")
    if not isinstance(statements, list) or not all(isinstance(s, str) and s.strip() for s in statements):
        raise JudgeOutputError(f"Expected a list of non-empty strings under 'statements', got {statements!r}")
    if not statements:
        raise JudgeOutputError("Judge extracted zero statements; faithfulness is undefined for this sample.")
    return statements


def judge_statements(
    sample: dict, statements: list[str], judge: BaseChatModel, callbacks: list[BaseCallbackHandler]
) -> list[int]:
    context = "\n".join(sample["retrieved_contexts"])
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    data = _ask(judge, _VERDICTS_SYSTEM, f"Context:\n{context}\n\nStatements:\n{numbered}", callbacks)
    rows = data.get("statements")
    if not isinstance(rows, list) or len(rows) != len(statements):
        raise JudgeOutputError(f"Expected {len(statements)} verdict(s), got {rows!r}")
    verdicts = []
    for row in rows:
        verdict = row.get("verdict") if isinstance(row, dict) else None
        # bool is a subclass of int in Python; reject True/False explicitly.
        if isinstance(verdict, bool) or verdict not in (0, 1):
            raise JudgeOutputError(f"Verdict must be 0 or 1, got {verdict!r}")
        verdicts.append(verdict)
    return verdicts


def score_samples(
    samples: list[dict], *, judge: BaseChatModel, callbacks: list[BaseCallbackHandler]
) -> list[dict[str, float]]:
    """Return one ``{"faithfulness": score}`` dict per sample, in order."""
    per_sample = []
    for sample in samples:
        statements = extract_statements(sample, judge, callbacks)
        verdicts = judge_statements(sample, statements, judge, callbacks)
        per_sample.append({METRIC_NAME: sum(verdicts) / len(verdicts)})
    return per_sample
