"""Offline stand-ins for two versions of the docs assistant model.

``assistant-v1`` answers all ten eval questions correctly.
``assistant-v2`` is cheaper but regressed: it gets four of them wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

_ITEMS = [json.loads(l) for l in (Path(__file__).resolve().parent.parent / "data" / "eval_items.jsonl").open()]
_ANSWERS = {it["question"]: it["expected"] for it in _ITEMS}
_V2_WRONG = {"q02": "Within 30 days", "q04": "1000 requests per minute", "q07": "Yes", "q08": "90 days"}


class FakeModel:
    def __init__(self, model_id: str):
        if model_id not in ("assistant-v1", "assistant-v2"):
            raise ValueError(model_id)
        self.model_id = model_id
        self.calls = 0

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        self.calls += 1
        question = prompt.split("Question: ", 1)[1].split("\n", 1)[0]
        item = next(it for it in _ITEMS if it["question"] == question)
        if self.model_id == "assistant-v2" and item["id"] in _V2_WRONG:
            return _V2_WRONG[item["id"]]
        return _ANSWERS[question]
