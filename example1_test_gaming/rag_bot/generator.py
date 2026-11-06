"""Answer generation.

Two implementations share one interface, ``generate(question, contexts) -> str``:

* ``SimulatedLLM`` (default, offline): a deterministic stand-in that behaves
  like a real LLM in the way that matters here. When the retrieved context
  contains a complete, relevant sentence it answers from that sentence
  (a grounded answer). When it does not, it falls back to "what it already
  believes" (its parametric memory) and gives a confident answer that is
  NOT supported by the context. That second behaviour is exactly what a
  faithfulness metric exists to catch.

* ``OpenAIGenerator`` (``EVAL_MODE=live``): calls a real model.
"""

from __future__ import annotations

import os
import re

from rag_bot.retriever import tokenize

# What a general-purpose model "believes" about common topics before
# reading our docs. These are plausible, but wrong for *our* company.
PARAMETRIC_PRIORS = {
    "refund": "Refunds are available within 30 days of purchase.",
    "support": "Support is available 24/7 by live chat.",
}
DEFAULT_PRIOR = "I'm not sure, but this is usually covered in your account settings."

SENTENCE_RE = re.compile(r"[A-Z][^.!?]*[.!?]")


class SimulatedLLM:
    def generate(self, question: str, contexts: list[str]) -> str:
        q = tokenize(question)
        best, best_overlap = None, 0
        for ctx in contexts:
            # Only whole sentences count. A fragment like "...may request a
            # full refund wi" is not something a model can reliably answer from.
            for sentence in SENTENCE_RE.findall(ctx):
                overlap = len(q & tokenize(sentence))
                if overlap > best_overlap:
                    best, best_overlap = sentence.strip(), overlap
        if best is not None and best_overlap >= 2:
            return best
        for keyword, prior in PARAMETRIC_PRIORS.items():
            if keyword in question.lower():
                return prior
        return DEFAULT_PRIOR


class OpenAIGenerator:
    """Real LLM generation. Requires OPENAI_API_KEY."""

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI  # imported lazily so offline runs don't need it configured

        self.client = OpenAI()
        self.model = model or os.getenv("GENERATOR_MODEL", "gpt-4.1-mini")

    def generate(self, question: str, contexts: list[str]) -> str:
        context_block = "\n\n".join(contexts)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": "Answer using only the provided context. Be brief."},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
            ],
        )
        return resp.choices[0].message.content or ""


def get_generator():
    return OpenAIGenerator() if os.getenv("EVAL_MODE") == "live" else SimulatedLLM()
