# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Model Loader — safetensors → model loading infrastructure.

Provides a ``SingleGPUModelBuilder`` that loads model weights from
safetensors checkpoints, applies optional LoRA, and builds fully
initialised PyTorch modules on a target device / dtype.
"""

from .builder import SingleGPUModelBuilder
from .registry import Registry, DummyRegistry
from .primitives import (
    LoraPathStrengthAndSDOps,
    ModelSpec,
    AIPRODV_LORA_COMFY_RENAMING_MAP,
)

__all__ = [
    "SingleGPUModelBuilder",
    "Registry",
    "DummyRegistry",
    "LoraPathStrengthAndSDOps",
    "ModelSpec",
    "AIPRODV_LORA_COMFY_RENAMING_MAP",
]
