# Case 8: A response cache that kept grading the old model

**Status:** as found. The fix lands in a separate pull request.

The docs-assistant eval caches model responses so CI runs cost nothing. `config.yaml` was just switched from `assistant-v1` to the cheaper `assistant-v2`. Accuracy is still 1.00 and CI is green.
