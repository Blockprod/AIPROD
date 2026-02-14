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
from .modality import Modality
from .wrapper import X0Model

# Backward-compatible aliases
AIPRODTransformer3DModel = SHDTModel
AIPRODModel = SHDTModel


def __getattr__(name: str):
    """Lazy imports to avoid circular dependency with configurators."""
    _configurator_names = {
        "SHDTConfigurator",
        "AIPRODModelConfigurator",
        "AIPRODV_MODEL_COMFY_RENAMING_MAP",
        "AIPRODV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP",
        "UPCAST_DURING_INFERENCE",
    }
    if name in _configurator_names:
        from aiprod_core.model import configurators as _cfg
        return getattr(_cfg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "Modality",
    "X0Model",
    "AIPRODTransformer3DModel",
    "AIPRODModel",
    "AIPRODModelConfigurator",
    "SHDTConfigurator",
    "AIPRODV_MODEL_COMFY_RENAMING_MAP",
    "AIPRODV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP",
    "UPCAST_DURING_INFERENCE",
]
