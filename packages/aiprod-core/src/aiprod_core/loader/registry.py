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
        from pathlib import Path
        from safetensors.torch import load_file

        p = Path(path).resolve()  # Always use absolute path for mmap
        if p.is_dir():
            candidates = sorted(p.glob("*.safetensors"))
            if not candidates:
                raise FileNotFoundError(
                    f"No .safetensors file found in directory: {path}"
                )
            p = candidates[0]

        abs_path = str(p)

        # Handle symlinks: resolve to actual file (e.g. HuggingFace cache)
        if p.is_symlink():
            abs_path = str(p.resolve())

        try:
            return load_file(abs_path)
        except OSError:
            # Fallback: load via torch if safetensors mmap fails (large files)
            import logging
            logging.warning(
                f"safetensors mmap failed for {abs_path}, falling back to torch.load"
            )
            return torch.load(abs_path, map_location="cpu", weights_only=True)


class DummyRegistry(Registry):
    """Registry that never caches — always reloads from disk."""

    def get_state_dict(self, path: str) -> dict[str, torch.Tensor]:
        return self._load(path)
