"""Parse and validate the extraction model's JSON output.

What went wrong
===============
The old parser "never raised": on any error it fell back to a regex, and
then to defaults ``{"intent": "other", "priority": "low"}``, and it silently
dropped unknown keys. Two things then looked like successes:

* Truncated or malformed outputs became the defaults, and "other / low" is
  the most common gold label, so most of them scored as correct.
* The new model version renamed ``priority`` to ``urgency``. The key was
  dropped, ``priority`` defaulted to "low", and nobody was told.

Now parsing is strict. The only tolerance is a single fenced ```json block,
which the model is documented to emit. Anything else raises ``ParseError``
with a reason, and the eval counts it as a failure.
"""
from __future__ import annotations

import json
import re

SCHEMA = {
    "intent": {"billing", "shipping", "account", "technical", "other"},
    "priority": {"low", "medium", "high"},
}
_FENCE = re.compile(r"^```(?:json)?\s*\n(.*)\n```\s*$", re.S)


class ParseError(ValueError):
    pass


def parse_output(text: str) -> dict:
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e.msg}") from e
    if not isinstance(obj, dict):
        raise ParseError(f"expected an object, got {type(obj).__name__}")
    missing = SCHEMA.keys() - obj.keys()
    extra = obj.keys() - SCHEMA.keys()
    if missing or extra:
        raise ParseError(f"schema mismatch: missing={sorted(missing)} unexpected={sorted(extra)}")
    for key, allowed in SCHEMA.items():
        if obj[key] not in allowed:
            raise ParseError(f"{key}={obj[key]!r} not in {sorted(allowed)}")
    return obj
