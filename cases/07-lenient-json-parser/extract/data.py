from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def load_outputs() -> list[dict]:
    """Recorded raw outputs from the extraction model, with gold labels."""
    with open(DATA / "model_outputs.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]
