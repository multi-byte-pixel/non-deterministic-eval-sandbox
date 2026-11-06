"""Document chunking for the support-bot RAG pipeline.

In a RAG (Retrieval-Augmented Generation) system we cannot hand the whole
knowledge base to the LLM on every question, so we cut documents into
"chunks", index those, and retrieve only the few most relevant chunks
per question. How you chunk has a big effect on answer quality: a chunk
that cuts a sentence in half can hide the one fact the model needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Chunk size in characters. Was 512 with 64 overlap; reduced to cut
# indexing cost (fewer tokens per chunk -> cheaper embeddings).
CHUNK_SIZE = 120
CHUNK_OVERLAP = 0


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text plus where it came from."""

    doc_id: str
    index: int
    text: str


def chunk_text(doc_id: str, text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Split ``text`` into fixed-size character windows."""
    chunks: list[Chunk] = []
    step = max(1, size - overlap)
    for i, start in enumerate(range(0, len(text), step)):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(Chunk(doc_id=doc_id, index=i, text=piece))
    return chunks


def load_and_chunk(kb_dir: Path) -> list[Chunk]:
    """Chunk every markdown file in the knowledge-base directory."""
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        chunks.extend(chunk_text(path.stem, path.read_text(encoding="utf-8")))
    return chunks
