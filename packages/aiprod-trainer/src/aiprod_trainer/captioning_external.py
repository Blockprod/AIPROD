"""
External/Cloud-based captioning — backward-compatible shim.
=============================================================

The actual implementation lives in ``aiprod_cloud.captioning_external``.
Install ``aiprod-cloud[gemini]`` to enable.

This shim preserves the original import path so that existing code like::

    from aiprod_trainer.captioning_external import GeminiFlashCaptioner

continues to work transparently.
"""

try:
    from aiprod_cloud.captioning_external import GeminiFlashCaptioner  # noqa: F401,PLC0415
except ImportError:
    GeminiFlashCaptioner = None  # type: ignore[assignment,misc]

__all__ = ["GeminiFlashCaptioner"]
