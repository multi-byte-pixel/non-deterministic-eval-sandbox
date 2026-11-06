"""A tiny keyword retriever.

Production systems use embeddings + a vector database (pgvector, OpenSearch,
Pinecone...). For a teaching sandbox we use plain keyword overlap so the
behaviour is 100% deterministic and needs no network. The *testing*
lessons are the same either way: retrieval quality is something you test
on its own, separately from generation quality.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from rag_bot.chunking import Chunk, load_and_chunk

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb"

# Words too common to help ranking.
STOPWORDS = {
    "a", "an", "the", "is", "are", "of", "to", "in", "on", "for", "and", "or",
    "do", "does", "how", "what", "when", "can", "i", "my", "we", "our", "be",
    "by", "with", "from", "after", "within", "have", "has", "long", "much",
}


def tokenize(text: str) -> set[str]:
    """Lower-case word/number tokens with stopwords removed."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS}


@lru_cache(maxsize=1)
def _index() -> tuple[Chunk, ...]:
    return tuple(load_and_chunk(KB_DIR))


def retrieve(question: str, k: int = 2) -> list[str]:
    """Return the text of the ``k`` chunks sharing the most keywords with the question."""
    q = tokenize(question)
    scored = sorted(_index(), key=lambda c: (-len(q & tokenize(c.text)), c.doc_id, c.index))
    return [c.text for c in scored[:k]]
