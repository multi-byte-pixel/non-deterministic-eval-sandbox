# Case 7: A "robust" JSON parser that turned failures into passes

**Status:** as found. The fix lands in a separate pull request.

A ticket-extraction model returns JSON with `intent` and `priority`. After a model version bump, field accuracy on 40 recorded outputs is 0.85, above the 0.80 threshold. CI is green.
