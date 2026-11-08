"""Parse the extraction model's JSON output."""
from __future__ import annotations

import json
import re

DEFAULT = {"intent": "other", "priority": "low"}


def parse_output(text: str) -> dict:
    """Best-effort parse. Never raises, so one bad output can't crash the eval."""
    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r"\{.*?\}", text, re.S)
        try:
            obj = json.loads(match.group(0)) if match else {}
        except Exception:
            obj = {}
    if not isinstance(obj, dict):
        obj = {}
    return {**DEFAULT, **{k: v for k, v in obj.items() if k in DEFAULT}}
