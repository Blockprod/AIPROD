"""
HuggingFace Hub Utilities — backward-compatible shim.
======================================================

The actual implementation lives in ``aiprod_cloud.hf_hub_utils``.
Install ``aiprod-cloud[huggingface]`` to enable.

This shim preserves the original import path so that existing code like::

    from aiprod_trainer.hf_hub_utils import push_to_hub

continues to work transparently.
"""

try:
    from aiprod_cloud.hf_hub_utils import push_to_hub  # noqa: F401,PLC0415
    _HF_HUB_AVAILABLE = True
except ImportError:
    push_to_hub = None  # type: ignore[assignment]
    _HF_HUB_AVAILABLE = False

__all__ = ["push_to_hub", "_HF_HUB_AVAILABLE"]
