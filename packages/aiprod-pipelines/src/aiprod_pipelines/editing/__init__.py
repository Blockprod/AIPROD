# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
Editing Module — Timeline generation, transitions, pacing, and EDL/FCPXML export.
"""

from .timeline import (
    EditingConfig,
    PacingEngine,
    TimelineClip,
    TimelineGenerator,
    TransitionType,
    TransitionsLib,
)

__all__ = [
    "EditingConfig",
    "PacingEngine",
    "TimelineClip",
    "TimelineGenerator",
    "TransitionType",
    "TransitionsLib",
]
