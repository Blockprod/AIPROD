# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
SHDT — Scalable Hybrid Diffusion Transformer

AIPROD's proprietary video denoising backbone.

Key architectural differences from prior work:
    - Configurable depth/width (not hardcoded 48-layer)
    - Dual-stream design: separate spatial and temporal attention paths
      that merge via cross-attention (not interleaved single-stream)
    - Grouped Query Attention (GQA) for memory efficiency
    - Learned positional encoding (not fixed RoPE frequencies)
    - Dynamic compute allocation per frame complexity
    - Native multi-resolution support via adaptive patching

Design Philosophy:
    The SHDT processes video latents as a sequence of spatial-temporal
    patches. Unlike single-stream transformers that flatten all dimensions,
    SHDT maintains separate spatial and temporal representations that
    interact through cross-attention, enabling better long-range temporal
    coherence while keeping memory manageable.
"""

from .model import SHDTModel, SHDTConfig
from .block import SHDTBlock
from .attention import (
    GroupedQueryAttention,
    SpatialAttention,
    TemporalAttention,
    CrossModalAttention,
)
from .position import LearnedPositionalEncoding3D
from .norm import AdaptiveRMSNorm

__all__ = [
    "SHDTModel",
    "SHDTConfig",
    "SHDTBlock",
    "GroupedQueryAttention",
    "SpatialAttention",
    "TemporalAttention",
    "CrossModalAttention",
    "LearnedPositionalEncoding3D",
    "AdaptiveRMSNorm",
]
