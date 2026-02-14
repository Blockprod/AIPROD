# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Weight Registry — Caching and sharing loaded state dicts.

When multiple models share the same checkpoint file (e.g. transformer,
VAE encoder, VAE decoder all inside one .safetensors), the registry
avoids re-reading the file for each model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


class Registry:
    """Caches loaded state dicts keyed by file path.

    Thread-safety: not guaranteed — use one Registry per thread.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, dict[str, torch.Tensor]] = {}

    def get_state_dict(self, path: str) -> dict[str, torch.Tensor]:
        """Load or return cached state dict from a safetensors file."""
        if path not in self._cache:
            self._cache[path] = self._load(path)
        return self._cache[path]

    def clear(self) -> None:
        """Release all cached state dicts."""
        self._cache.clear()

    @staticmethod
    def _load(path: str) -> dict[str, torch.Tensor]:
        from safetensors.torch import load_file
        return load_file(path)


class DummyRegistry(Registry):
    """Registry that never caches — always reloads from disk."""

    def get_state_dict(self, path: str) -> dict[str, torch.Tensor]:
        return self._load(path)
