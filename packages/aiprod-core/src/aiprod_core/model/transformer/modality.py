# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Transformer — Modality descriptor for multi-modal inputs.

A Modality bundles all the tensors the transformer needs for one
modality stream (video or audio): patchified latent, positional
coordinates, text context, and denoising mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Modality:
    """Bundles input tensors for one modality stream.

    The SHDT transformer accepts a pair of Modality objects
    (video + audio) and processes them through its dual-stream
    architecture.

    Args:
        latent: Patchified noisy latent [B, seq, C].
        positions: Normalised coordinates [B, ndim, seq, 2].
        context: Text encoder hidden states [B, text_seq, D].
        denoise_mask: [B, seq, 1] — 1.0 for tokens to denoise.
        sigma: Current noise level (scalar or [B]).
    """
    latent: torch.Tensor
    positions: torch.Tensor
    context: torch.Tensor
    denoise_mask: torch.Tensor | None = None
    sigma: torch.Tensor | float = 1.0
