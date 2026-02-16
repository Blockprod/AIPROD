"""
Gemini API Client — backward-compatible shim.
===============================================

The actual implementation lives in ``aiprod_cloud.gemini_client``.
Install ``aiprod-cloud[gemini]`` to enable.

This shim preserves the original import path so that existing code like::

    from aiprod_pipelines.api.integrations.gemini_client import GeminiAPIClient

continues to work transparently.
"""

try:
    from aiprod_cloud.gemini_client import GeminiAPIClient  # noqa: F401,PLC0415
except ImportError:
    GeminiAPIClient = None  # type: ignore[assignment,misc]

__all__ = ["GeminiAPIClient"]
