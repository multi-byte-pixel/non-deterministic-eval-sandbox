"""Document chunking for the support-bot RAG pipeline.

Background for new readers
==========================
In a RAG (Retrieval-Augmented Generation) system we cannot paste the whole
knowledge base into the LLM prompt on every question. Instead we:

    1. cut each document into "chunks"            <- this file
    2. index the chunks (here: keywords; in production: embeddings)
    3. at question time, retrieve the top-k most relevant chunks
    4. give only those chunks to the LLM as context and ask it to answer

Chunking looks like boring plumbing, but it decides what the LLM is
*able* to see. If a chunk boundary cuts a sentence in half, for example
"...within 5 to 7" | "business days...", then neither chunk has the
complete fact. The model then either says "I don't know" or, more often,
answers from its training data (a hallucination with respect to our
docs). A faithfulness metric will flag that answer, and that is exactly
the regression the faithfulness test in this repo started reporting.

What went wrong (the root cause behind PR #1)
=============================================
A cost-reduction change switched to fixed 120-character windows with no
overlap. Fixed-width character windows ignore the structure of the text,
so they regularly split sentences and paragraphs. Instead of finding this
root cause, the coding agent changed the *test* until it passed (see
tests/test_faithfulness.py and the PR description).

The fix: structure-aware chunking
=================================
We now chunk along the natural boundaries of the document:

    * A paragraph (text between blank lines) is the preferred unit. In
      well-written docs a paragraph is usually one self-contained idea.
    * A paragraph that fits in ``max_chars`` becomes exactly one chunk and
      is NEVER split. This is the property the regression test
      ``tests/test_chunking.py`` locks in.
    * A paragraph that is too long is split on *sentence* boundaries, and
      consecutive chunks share ``overlap_sentences`` sentences, so a fact
      that spans a boundary is still complete in at least one chunk.
    * Markdown headings ("# Refund Policy") are not chunks of their own;
      they carry almost no answerable content and would waste a top-k slot.

Production libraries offer the same idea: LangChain's
``RecursiveCharacterTextSplitter`` tries separators in order
("\\n\\n", then "\\n", then " ") for this reason. We write ours by hand so
every rule is visible and testable.

Trade-off to keep in mind: bigger chunks mean more complete facts but
more tokens per prompt (cost/latency) and noisier retrieval. Chunking
parameters belong in your eval loop: change them, re-run the golden set,
compare scores. Never change them blind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Upper bound for one chunk, in characters. Every paragraph in our KB
# fits, so each paragraph becomes one chunk. Kept as a named constant so
# eval runs can sweep it (e.g. 200 / 400 / 800) and compare scores.
MAX_CHARS = 400

# When a long paragraph must be split, how many sentences consecutive
# chunks share. One sentence of overlap is a common, cheap default.
OVERLAP_SENTENCES = 1

# Sentence boundary: end punctuation followed by whitespace. Good enough for
# our docs; production code might use a real sentence tokenizer (nltk,
# spaCy) to handle "e.g." and "Dr." correctly.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text plus where it came from.

    ``doc_id`` and ``index`` are metadata. They let us cite sources in
    answers and, in tests, assert *which* chunk was retrieved (a much
    sharper assertion than "some text came back").
    """

    doc_id: str
    index: int
    text: str


def _paragraphs(text: str) -> list[str]:
    """Split on blank lines, drop markdown headings and empty blocks."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text)]
    return [b for b in blocks if b and not b.startswith("#")]


def _split_long_paragraph(paragraph: str, max_chars: int, overlap: int) -> list[str]:
    """Greedy sentence packing with sentence-level overlap.

    Example with max_chars small enough for two sentences per chunk and
    overlap=1:  [S1 S2] [S2 S3] [S3 S4] ...
    Every sentence is complete in at least one chunk, and every pair of
    neighbouring sentences appears together at least once.
    """
    sentences = _SENTENCE_BOUNDARY.split(paragraph)
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if current and len(candidate) > max_chars:
            pieces.append(" ".join(current))
            # Carry the last `overlap` sentences into the next chunk.
            current = current[-overlap:] if overlap else []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(
    doc_id: str,
    text: str,
    max_chars: int = MAX_CHARS,
    overlap_sentences: int = OVERLAP_SENTENCES,
) -> list[Chunk]:
    """Split ``text`` into paragraph-aligned chunks (see module docstring)."""
    chunks: list[Chunk] = []
    for paragraph in _paragraphs(text):
        if len(paragraph) <= max_chars:
            pieces = [paragraph]  # the common case: one paragraph == one chunk
        else:
            pieces = _split_long_paragraph(paragraph, max_chars, overlap_sentences)
        for piece in pieces:
            chunks.append(Chunk(doc_id=doc_id, index=len(chunks), text=piece))
    return chunks


def load_and_chunk(kb_dir: Path) -> list[Chunk]:
    """Chunk every markdown file in the knowledge-base directory.

    ``sorted`` makes the order (and therefore tie-breaking in retrieval)
    deterministic across machines. Small thing, but unsorted ``glob``
    output differs between filesystems and causes "works on my machine"
    eval diffs.
    """
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        chunks.extend(chunk_text(path.stem, path.read_text(encoding="utf-8")))
    return chunks
