import os

# DeepEval sends anonymous usage telemetry by default; keep CI quiet and offline.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
