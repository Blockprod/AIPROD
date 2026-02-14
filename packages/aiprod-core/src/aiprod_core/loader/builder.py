# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD SingleGPUModelBuilder — Build models from safetensors checkpoints.

Orchestrates:
    1. Loading state dict(s) from safetensors files via Registry
    2. Filtering keys via ``model_sd_ops`` (prefix filters, renaming)
    3. Instantiating the model via ``model_class_configurator``
    4. Loading the filtered state dict into the model
    5. Optionally applying LoRA adapters
    6. Moving to target device / dtype

Usage::

    builder = SingleGPUModelBuilder(
        model_path="checkpoint.safetensors",
        model_class_configurator=SHDTConfigurator,
    )
    model = builder.build(device=torch.device("cuda"), dtype=torch.bfloat16)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, Type

import torch

from .primitives import LoraPathStrengthAndSDOps, KeyRemapOp
from .registry import DummyRegistry, Registry


# Type for model configurator — a class with ``from_state_dict`` + ``build``
ModelConfiguratorType = Any


@dataclass
class SingleGPUModelBuilder:
    """Builds an nn.Module from safetensors checkpoint(s).

    Args:
        model_path: Path or tuple of paths to safetensors file(s).
        model_class_configurator: Configurator class (has ``from_state_dict``
            and ``build_model`` methods).  If None, raw state dict is returned.
        model_sd_ops: Key remapping / filtering callables.
        loras: LoRA adapters to apply after loading base weights.
        registry: Weight cache.  Defaults to DummyRegistry (no caching).
        module_ops: Additional operations to apply to model after loading.
    """
    model_path: str | tuple[str, ...]
    model_class_configurator: ModelConfiguratorType | None = None
    model_sd_ops: Sequence[KeyRemapOp] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = ()
    registry: Registry = field(default_factory=DummyRegistry)
    module_ops: tuple[Any, ...] = ()

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.nn.Module:
        """Load weights and build the model.

        Args:
            device: Target device (default: CPU).
            dtype: Target dtype (default: float32).

        Returns:
            Initialised nn.Module with loaded weights.
        """
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32

        # 1. Load state dict(s)
        state_dict = self._load_state_dict()

        # 2. Apply key remapping
        state_dict = self._remap_keys(state_dict)

        # 3. Build model via configurator
        if self.model_class_configurator is not None:
            configurator = self.model_class_configurator
            if hasattr(configurator, "from_state_dict"):
                model = configurator.from_state_dict(state_dict, device=device, dtype=dtype)
            elif hasattr(configurator, "build_model"):
                model = configurator.build_model(state_dict, device=device, dtype=dtype)
            elif callable(configurator):
                model = configurator(state_dict, device=device, dtype=dtype)
            else:
                raise TypeError(
                    f"Configurator {configurator} must have from_state_dict, "
                    f"build_model, or be callable"
                )
        else:
            # Fallback: return a simple wrapper
            model = _StateDictModule(state_dict)

        # 4. Apply LoRA adapters
        for lora in self.loras:
            self._apply_lora(model, lora)

        # 5. Move to device / dtype
        if isinstance(model, torch.nn.Module):
            model = model.to(device=device, dtype=dtype)

        return model

    def _load_state_dict(self) -> dict[str, torch.Tensor]:
        """Load and merge state dicts from all paths."""
        paths = (self.model_path,) if isinstance(self.model_path, str) else self.model_path
        merged: dict[str, torch.Tensor] = {}
        for p in paths:
            sd = self.registry.get_state_dict(p)
            merged.update(sd)
        return merged

    def _remap_keys(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply key remapping operations."""
        for op in self.model_sd_ops:
            if callable(op):
                new_sd = {}
                for key, value in state_dict.items():
                    new_key = op(key)
                    if new_key is not None:
                        new_sd[new_key] = value
                state_dict = new_sd
        return state_dict

    def _apply_lora(self, model: torch.nn.Module, lora: LoraPathStrengthAndSDOps) -> None:
        """Merge a LoRA adapter into the model weights."""
        lora_sd = self.registry.get_state_dict(lora.path)

        # Remap LoRA keys
        for op in lora.sd_ops:
            if callable(op):
                new_sd = {}
                for key, value in lora_sd.items():
                    new_key = op(key)
                    if new_key is not None:
                        new_sd[new_key] = value
                lora_sd = new_sd

        # Merge LoRA: W = W + strength * (A @ B)
        model_sd = dict(model.named_parameters())
        for key in lora_sd:
            if key.endswith(".lora_A.weight"):
                base_key = key.replace(".lora_A.weight", ".weight")
                b_key = key.replace(".lora_A.weight", ".lora_B.weight")
                if base_key in model_sd and b_key in lora_sd:
                    a = lora_sd[key]
                    b = lora_sd[b_key]
                    delta = (b @ a) * lora.strength
                    model_sd[base_key].data.add_(delta.to(model_sd[base_key].dtype))


class _StateDictModule(torch.nn.Module):
    """Minimal wrapper for a raw state dict (fallback when no configurator)."""

    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        super().__init__()
        for k, v in state_dict.items():
            self.register_buffer(k.replace(".", "_"), v)
