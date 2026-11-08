"""Response cache so CI does not pay for the same completion twice."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ResponseCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    @staticmethod
    def key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str):
        return self.data.get(self.key(prompt))

    def put(self, prompt: str, response: str) -> None:
        self.data[self.key(prompt)] = response

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True))
