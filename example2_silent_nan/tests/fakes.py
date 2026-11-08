"""Test doubles for LangChain chat models.

Why fake the judge at all?
==========================
Offline CI must not need API keys (forks don't get secrets, and every run
would cost money). A *scripted* judge lets us run the REAL Ragas pipeline
(prompt building, JSON parsing, score aggregation, our harness checks)
end to end with no network.

What a fake judge can and cannot tell you:
  ✅ the harness wires the judge in correctly (it gets called)
  ✅ scores flow from judge verdicts to the gate correctly
  ✅ broken judge output is reported as an error, not a score
  ❌ whether a real judge model grades well. That needs live evals and
     canaries with known-good/known-bad answers (see example 1).

How LangChain chat models work (enough to write a fake)
=======================================================
Subclass ``BaseChatModel`` and implement ``_generate(messages, ...)``,
returning a ``ChatResult``. Everything else (``invoke``, async, batching,
callbacks, tracing) comes from the base class for free. That matters here:
callbacks fire for fakes exactly as they do for real models, so our
``JudgeCallCounter`` sees fake calls the same way it sees real ones.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedJudge(BaseChatModel):
    """Returns well-formed Ragas Faithfulness JSON with a fixed verdict.

    Ragas Faithfulness sends two kinds of prompt:
      * statement extraction -> expects {"statements": ["...", ...]}
      * NLI verdicts         -> expects {"statements": [{"statement", "reason", "verdict"}]}
    We tell them apart by the word "verdict" in the prompt. That is a
    deliberately simple heuristic. If a future Ragas changes its prompts,
    the contract tests fail loudly (the judge output won't parse, and with
    raise_exceptions=True that raises) rather than silently passing.

    ``verdict=1`` -> every statement supported   -> faithfulness 1.0
    ``verdict=0`` -> every statement unsupported -> faithfulness 0.0
    """

    verdict: int = 1
    calls: int = 0  # pydantic field; BaseChatModel is a pydantic model

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


class GarbageJudge(BaseChatModel):
    """A judge that answers with text that is not JSON.

    Simulates real-world judge failures: a model update that changes output
    style, a refusal ("I can't evaluate this"), a truncated response. The
    harness must turn this into an ERROR, never a NaN that might pass.
    """

    @property
    def _llm_type(self) -> str:
        return "garbage-judge"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Sure! The answer looks great :)"))])
