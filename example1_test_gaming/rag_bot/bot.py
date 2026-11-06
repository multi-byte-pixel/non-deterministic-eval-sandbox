"""The RAG support bot: retrieve, then generate."""

from __future__ import annotations

from dataclasses import dataclass

from rag_bot import retriever
from rag_bot.generator import get_generator


@dataclass
class BotResponse:
    answer: str
    retrieved_contexts: list[str]


def answer(question: str, k: int = 2) -> BotResponse:
    # We call through the module (retriever.retrieve) rather than importing the
    # function directly so it can be patched in tests.
    contexts = retriever.retrieve(question, k=k)
    return BotResponse(answer=get_generator().generate(question, contexts), retrieved_contexts=contexts)
