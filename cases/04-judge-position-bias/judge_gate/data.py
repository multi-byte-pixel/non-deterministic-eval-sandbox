from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Pair:
    id: str
    question: str
    baseline: str
    candidate: str
    human_quality: dict


def load_pairs(name: str = "release_candidate_pairs.jsonl") -> list[Pair]:
    with open(DATA / name) as f:
        return [Pair(**json.loads(line)) for line in f if line.strip()]


def quality_index(pairs: list[Pair]) -> dict[str, float]:
    idx: dict[str, float] = {}
    for p in pairs:
        idx[p.baseline] = p.human_quality["baseline"]
        idx[p.candidate] = p.human_quality["candidate"]
    return idx
