"""
AIPROD Cloud — Optional cloud integrations.
=============================================

This package contains ALL cloud SDK dependencies isolated from
sovereign production packages.  It is never required at runtime;
production packages detect its presence via ``try/except ImportError``.

Sub-modules
-----------
- ``gcp_services``          – Google Cloud Storage / Logging / Monitoring adapter
- ``gemini_client``         – Google Gemini API client
- ``stripe_integration``    – Stripe payment integration
- ``cloud_sources``         – S3 / GCS / HuggingFace streaming data sources
- ``captioning_external``   – Gemini Flash captioner
- ``hf_hub_utils``          – Push trained weights to HuggingFace Hub
"""

__version__ = "0.1.0"
