"""Test doubles for chat models."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedJudge(BaseChatModel):
    """Fake judge model that returns well-formed Ragas faithfulness JSON."""

    verdict: int = 1
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-judge"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.calls += 1
        prompt = str(messages[-1].content)
        if "verdict" in prompt:
            payload = {"statements": [{"statement": "s", "reason": "scripted", "verdict": self.verdict}]}
        else:
            payload = {"statements": ["s"]}
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=json.dumps(payload)))])
