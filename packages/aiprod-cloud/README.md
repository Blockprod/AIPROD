# aiprod-cloud

Optional cloud integrations package for AIPROD.

This package isolates **all cloud SDK dependencies** from the sovereign production
packages (`aiprod-core`, `aiprod-pipelines`, `aiprod-trainer`).

## Install

```bash
# All cloud integrations
pip install aiprod-cloud[all]

# Specific integrations only
pip install aiprod-cloud[gcp]        # Google Cloud Storage / Logging / Monitoring
pip install aiprod-cloud[gemini]     # Google Gemini API
pip install aiprod-cloud[stripe]     # Stripe billing
pip install aiprod-cloud[s3]         # AWS S3
pip install aiprod-cloud[huggingface] # HuggingFace Hub
```

## Architecture

Without `aiprod-cloud` installed, the sovereign packages operate in **100 %
local / self-hosted mode** with zero cloud SDK imports.

When `aiprod-cloud` is installed, the production packages detect it at runtime
via `try: from aiprod_cloud.xxx import ...` and unlock cloud features
transparently.
