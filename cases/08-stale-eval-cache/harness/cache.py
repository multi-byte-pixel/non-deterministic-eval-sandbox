"""Response cache keyed on everything that can change a response.

What went wrong
===============
The key was ``sha256(prompt)``. Switching ``model_id`` from assistant-v1 to
assistant-v2 did not change any prompt, so every lookup hit responses that
v1 had produced. The eval graded v1 and reported it as v2. The committed
cache file also made CI "free", which hid that no model was being called.

Now the key covers the model id, the sampling parameters and the prompt
template version, and each entry records which model produced it. A
response from a different model can never be returned.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


class StaleCacheError(RuntimeError):
    pass


class ResponseCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(prompt: str, model_id: str, temperature: float, template_version: str) -> str:
        material = json.dumps(
            {"prompt": prompt, "model_id": model_id, "temperature": temperature,
             "template_version": template_version},
            sort_keys=True,
        )
        return hashlib.sha256(material.encode()).hexdigest()

    def get(self, prompt: str, model_id: str, temperature: float, template_version: str):
        entry = self.data.get(self.key(prompt, model_id, temperature, template_version))
        if entry is None:
            self.misses += 1
            return None
        if entry.get("model_id") != model_id:
            raise StaleCacheError(f"cache entry from {entry.get('model_id')} served for {model_id}")
        self.hits += 1
        return entry["response"]

    def put(self, prompt: str, response: str, model_id: str, temperature: float, template_version: str) -> None:
        self.data[self.key(prompt, model_id, temperature, template_version)] = {
            "response": response, "model_id": model_id,
        }

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True))
