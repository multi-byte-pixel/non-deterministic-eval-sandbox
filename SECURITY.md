# Security scope

This repository contains deliberately faulty evaluation examples and historical dependency pins. It is not a deployment template.

Case 2 pins `ragas==0.4.3`, affected by an SSRF vulnerability in its multimodal faithfulness collections module. No patched version is listed in [GHSA-95ww-475f-pr4f](https://github.com/advisories/GHSA-95ww-475f-pr4f) as of this review on November 9, 2026.

- Use synthetic inputs in isolated environments without credentials or access to internal services.
- Do not expose these examples as a service or pass untrusted URLs, files, or retrieved contexts to Ragas.
- Offline test success does not establish that the dependencies are secure.
- A future dependency upgrade needs compatibility tests; the existing alert remains open.
