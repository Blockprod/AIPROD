# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
Learned 3D Positional Encoding for Video

Unlike fixed sinusoidal or RoPE encodings used in prior work,
AIPROD uses fully learned positional embeddings that can adapt
to arbitrary video resolutions and frame counts during training.

The encoding is decomposed into three independent axes (T, H, W)
and combined additively, enabling generalization to unseen resolutions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LearnedPositionalEncoding3D(nn.Module):
    """Learned decomposed 3D positional encoding.

    Decomposes position encoding into three learnable 1D embeddings
    (temporal, height, width) that are combined additively.

    This allows training on one resolution and generalizing to others,
    since each axis is independently encoded.

    Args:
        dim: Embedding dimension.
        max_t: Maximum temporal positions.
        max_h: Maximum height positions.
        max_w: Maximum width positions.
    """

    def __init__(self, dim: int, max_t: int = 64, max_h: int = 128, max_w: int = 128):
        super().__init__()
        self.temporal_embed = nn.Embedding(max_t, dim)
        self.height_embed = nn.Embedding(max_h, dim)
        self.width_embed = nn.Embedding(max_w, dim)

        # Learnable scale factors for each axis
        self.scale_t = nn.Parameter(torch.ones(1) * 0.1)
        self.scale_h = nn.Parameter(torch.ones(1) * 0.1)
        self.scale_w = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        t: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate positional encoding for a video of shape (T, H, W).

        Args:
            t: Number of temporal positions.
            h: Number of height positions.
            w: Number of width positions.
            device: Target device.
            dtype: Target dtype.

        Returns:
            [1, T*H*W, D] positional encoding tensor.
        """
        t_pos = torch.arange(t, device=device)
        h_pos = torch.arange(h, device=device)
        w_pos = torch.arange(w, device=device)

        t_emb = self.temporal_embed(t_pos) * self.scale_t   # [T, D]
        h_emb = self.height_embed(h_pos) * self.scale_h     # [H, D]
        w_emb = self.width_embed(w_pos) * self.scale_w       # [W, D]

        # Broadcast: [T,1,1,D] + [1,H,1,D] + [1,1,W,D] → [T,H,W,D]
        pos = (
            t_emb[:, None, None, :]
            + h_emb[None, :, None, :]
            + w_emb[None, None, :, :]
        )

        # Flatten to sequence: [T*H*W, D]
        pos = pos.reshape(1, t * h * w, -1).to(dtype=dtype)
        return pos
