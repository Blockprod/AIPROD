"""
AIPROD API Integration Layer
=============================

This module provides the orchestration layer that integrates AIPROD systems
with external production workflows. It implements a state machine with
checkpoint/resume capabilities for resilient production execution.

Main Components:
- Orchestrator: State machine executor with checkpoint integration
- CheckpointManager: Save/restore execution state
- SchemaTransformer: Bidirectional schema conversion
- Adapters: Bridge layer to existing AIPROD systems
"""

from .orchestrator import Orchestrator
from .checkpoint.manager import CheckpointManager
from .checkpoint.recovery import RecoveryManager
from .schema.transformer import SchemaTransformer

__all__ = [
    "Orchestrator",
    "CheckpointManager",
    "RecoveryManager",
    "SchemaTransformer",
]

__version__ = "2.0.0"
