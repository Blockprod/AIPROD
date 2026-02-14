"""
Adapter Layer - Bridge to Existing AIPROD Systems
=================================================

Provides base protocols and adapter implementations that bridge
AIPROD orchestration to existing inference systems.
"""

from .base import BaseAdapter, AdapterProtocol

__all__ = ["BaseAdapter", "AdapterProtocol"]
