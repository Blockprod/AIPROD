# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
Lip-Sync Module — Audio-Visual Synchronisation

Predicts FLAME 52-param facial blend-shapes from audio mel-spectrograms.
"""

from .model import LipSyncModel, LipSyncConfig

__all__ = ["LipSyncModel", "LipSyncConfig"]
