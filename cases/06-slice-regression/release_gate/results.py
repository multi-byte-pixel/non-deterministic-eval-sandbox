from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def load(name: str) -> list[dict]:
    with open(DATA / name) as f:
        return [json.loads(line) for line in f if line.strip()]
