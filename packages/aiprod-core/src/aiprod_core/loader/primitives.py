# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Loader Primitives — LoRA specs, key remapping, model specs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence


@dataclass
class ModelSpec:
    """Specification for locating and loading model weights.

    Args:
        path: Path to a safetensors file or directory.
        key_prefix: Optional prefix filter for state dict keys.
    """
    path: str
    key_prefix: str = ""


# Type alias: a function that remaps state-dict key names.
KeyRemapOp = Callable[[str], str]


@dataclass(frozen=True)
class LoraPathStrengthAndSDOps:
    """Describes a LoRA adapter to apply on top of base weights.

    Args:
        path: Filesystem path to the LoRA safetensors file.
        strength: Multiplicative strength ∈ (0, 1].
        sd_ops: Key remapping operations to align LoRA keys with model keys.
    """
    path: str
    strength: float = 1.0
    sd_ops: Sequence[KeyRemapOp] = field(default_factory=tuple)


# ─── Key Remapping Maps ──────────────────────────────────────────────────────

def _identity_remap(key: str) -> str:
    """No-op key remapping."""
    return key


# Default LoRA key remapper — strips common prefixes
AIPRODV_LORA_COMFY_RENAMING_MAP: Sequence[KeyRemapOp] = (_identity_remap,)
