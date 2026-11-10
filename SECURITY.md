# Security scope

This repository contains deliberately faulty evaluation examples. It is not a deployment template.

Case 2 previously pinned `ragas==0.4.3`, affected by an SSRF vulnerability ([GHSA-95ww-475f-pr4f](https://github.com/advisories/GHSA-95ww-475f-pr4f)) with no patched release listed as of November 10, 2026. Ragas was removed and replaced with a small in-repo faithfulness metric (`cases/02-silent-nan-gate/evals/faithfulness.py`).

- Use synthetic inputs in isolated environments without credentials or access to internal services.
- Do not expose these examples as a service.
- Offline test success does not establish that the dependencies are secure.
