"""
Google Cloud Services Adapter — backward-compatible shim.
==========================================================

The actual implementation lives in ``aiprod_cloud.gcp_services``.
Install ``aiprod-cloud[gcp]`` to enable.

This shim preserves the original import path so that existing code like::

    from aiprod_pipelines.api.adapters.gcp_services import GoogleCloudServicesAdapter

continues to work transparently.
"""

try:
    from aiprod_cloud.gcp_services import GoogleCloudServicesAdapter  # noqa: F401,PLC0415
except ImportError:
    GoogleCloudServicesAdapter = None  # type: ignore[assignment,misc]

__all__ = ["GoogleCloudServicesAdapter"]

