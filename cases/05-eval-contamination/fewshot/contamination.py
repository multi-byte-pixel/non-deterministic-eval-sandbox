"""Detect held-out items leaking into prompts or training data.

Exact-id checks catch the obvious case. Normalized-text checks catch copies
that were re-exported with new ids, different casing or punctuation.
"""
from __future__ import annotations

import re


class ContaminationError(RuntimeError):
    pass


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def find_overlap(pool: list[dict], held_out: list[dict]) -> list[str]:
    held_ids = {it["id"] for it in held_out}
    held_text = {normalize(it["text"]): it["id"] for it in held_out}
    leaks = []
    for ex in pool:
        if ex["id"] in held_ids:
            leaks.append(ex["id"])
        elif normalize(ex["text"]) in held_text:
            leaks.append(held_text[normalize(ex["text"])])
    return leaks
