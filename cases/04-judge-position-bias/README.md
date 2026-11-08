# Case 4: A pairwise LLM judge that preferred whatever it read first

**Status:** as found. The fix lands in a separate pull request.

A release gate compares a candidate summarization model against the current baseline with a pairwise LLM judge and ships the candidate if it wins at least half the comparisons. CI is green and the gate says ship.
