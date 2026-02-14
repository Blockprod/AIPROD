# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Utilities

General-purpose utility functions for tensor operations,
file I/O, and configuration management.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import torch


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count model parameters (total, trainable, frozen).

    Args:
        model: PyTorch module.

    Returns:
        Dict with 'total', 'trainable', 'frozen' counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def format_param_count(count: int) -> str:
    """Format a parameter count as human-readable string.

    Examples: 1_500_000 → '1.5M', 2_300_000_000 → '2.3B'
    """
    if count >= 1e9:
        return f"{count / 1e9:.1f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.1f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.1f}K"
    return str(count)


def find_checkpoint(directory: str | Path, pattern: str = "*.pt") -> Path | None:
    """Find the latest checkpoint file in a directory.

    Args:
        directory: Path to search.
        pattern: Glob pattern for checkpoint files.

    Returns:
        Path to the latest checkpoint, or None if not found.
    """
    directory = Path(directory)
    if not directory.exists():
        return None

    checkpoints = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


def seed_everything(seed: int) -> None:
    """Set random seed for reproducibility across all libraries."""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
