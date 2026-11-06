"""Deterministic unit tests for the chunker.

Why test chunking separately from the LLM eval?
===============================================
LLM-judged evals are slow, cost money and (in live mode) are noisy. The
root cause in this example was a purely deterministic bug: a chunk
boundary landing mid-sentence. Deterministic bugs deserve deterministic
tests. They run in milliseconds, never flake, and point straight at the
broken function.

A good habit for LLM systems: push as much verification as possible
*down* into ordinary unit tests, and keep the LLM-judged layer for what
only a model can judge.
"""

from __future__ import annotations

from rag_bot.chunking import MAX_CHARS, chunk_text, load_and_chunk
from rag_bot.retriever import KB_DIR


def _kb_paragraphs() -> list[str]:
    """Every non-heading paragraph in the real knowledge base."""
    paragraphs = []
    for path in sorted(KB_DIR.glob("*.md")):
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            block = block.strip()
            if block and not block.startswith("#"):
                paragraphs.append(block)
    return paragraphs


def test_no_paragraph_that_fits_is_ever_split():
    """THE regression test for this incident.

    Before PR #1, "...within 5 to 7" and "business days..." ended up in
    different chunks. This asserts that every KB paragraph short enough to
    fit appears verbatim inside a single chunk.
    """
    chunk_texts = [c.text for c in load_and_chunk(KB_DIR)]
    for paragraph in _kb_paragraphs():
        if len(paragraph) <= MAX_CHARS:
            assert any(paragraph in text for text in chunk_texts), f"paragraph was split: {paragraph!r}"


def test_known_incident_fact_is_intact():
    """Pin the exact fact that broke, in plain words, for future readers."""
    assert any("within 5 to 7 business days" in c.text for c in load_and_chunk(KB_DIR))


def test_headings_are_not_chunks():
    assert not any(c.text.startswith("#") for c in load_and_chunk(KB_DIR))


def test_long_paragraph_splits_on_sentences_with_overlap():
    """Property test of the fallback path, which our small KB never exercises.

    Untested fallback branches are where the next regression hides.
    """
    sentences = [f"Sentence number {i} has a fact." for i in range(10)]
    chunks = chunk_text("doc", " ".join(sentences), max_chars=70, overlap_sentences=1)

    assert len(chunks) > 1
    for c in chunks:
        # every chunk ends on a sentence boundary: no "half facts"
        assert c.text.endswith(".")
    for s in sentences:
        # every sentence survives whole in at least one chunk
        assert any(s in c.text for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        # neighbouring chunks share their boundary sentence (the overlap)
        assert a.text.split(". ")[-1].rstrip(".") in b.text


def test_chunk_indices_are_stable_and_sequential():
    chunks = chunk_text("doc", "First paragraph.\n\nSecond paragraph.")
    assert [(c.doc_id, c.index) for c in chunks] == [("doc", 0), ("doc", 1)]
