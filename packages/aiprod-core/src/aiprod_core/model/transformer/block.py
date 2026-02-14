# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
SHDT Block — Single transformer block in the dual-stream architecture.

Each block contains:
    1. Spatial self-attention (within-frame)
    2. Temporal self-attention (across-frames)
    3. Cross-modal attention (video ← text)
    4. Feed-forward network
    5. Adaptive normalization conditioned on timestep

Cross-stream layers additionally merge spatial and temporal
representations via a learnable gating mechanism.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import SpatialAttention, TemporalAttention, CrossModalAttention
from .norm import AdaptiveRMSNorm


class SHDTBlock(nn.Module):
    """One block of the Scalable Hybrid Diffusion Transformer.

    Args:
        hidden_dim: Model dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of KV heads (GQA).
        head_dim: Dimension per head.
        ff_dim: Feed-forward intermediate dimension.
        layer_idx: Index of this layer in the stack.
        is_cross_stream: Whether this layer merges spatial/temporal streams.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        ff_dim: int,
        layer_idx: int,
        is_cross_stream: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_cross_stream = is_cross_stream

        # ── Pre-norms (adaptive, conditioned on timestep) ──
        self.norm_spatial = AdaptiveRMSNorm(hidden_dim)
        self.norm_temporal = AdaptiveRMSNorm(hidden_dim)
        self.norm_cross = AdaptiveRMSNorm(hidden_dim)
        self.norm_ff = AdaptiveRMSNorm(hidden_dim)

        # ── Attention layers ──
        self.spatial_attn = SpatialAttention(hidden_dim, num_heads, num_kv_heads, head_dim, dropout)
        self.temporal_attn = TemporalAttention(hidden_dim, num_heads, num_kv_heads, head_dim, dropout)
        self.cross_attn = CrossModalAttention(hidden_dim, num_heads, num_kv_heads, head_dim, dropout)

        # ── Feed-forward ──
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # ── Stream merging gate (only for cross-stream layers) ──
        if is_cross_stream:
            self.stream_gate = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        x: torch.Tensor,
        t_embed: torch.Tensor,
        text_ctx: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        video_shape: tuple[int, int, int] = (1, 1, 1),
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T*H*W, D] video token sequence.
            t_embed: [B, D] timestep conditioning embedding.
            text_ctx: [B, S, D] text context from LLM Bridge.
            text_mask: [B, S] text attention mask.
            video_shape: (T, H, W) for reshaping into spatial/temporal views.

        Returns:
            [B, T*H*W, D] processed tokens.
        """
        # 1. Spatial self-attention (within each frame)
        residual = x
        x_normed = self.norm_spatial(x, t_embed)
        x_spatial = self.spatial_attn(x_normed, video_shape)

        # 2. Temporal self-attention (across frames)
        x_normed = self.norm_temporal(x, t_embed)
        x_temporal = self.temporal_attn(x_normed, video_shape)

        # 3. Merge spatial and temporal streams
        if self.is_cross_stream:
            gate = torch.sigmoid(self.stream_gate)
            x = residual + gate * x_spatial + (1 - gate) * x_temporal
        else:
            x = residual + x_spatial + x_temporal

        # 4. Cross-modal attention (video ← text)
        residual = x
        x = residual + self.cross_attn(self.norm_cross(x, t_embed), text_ctx, text_mask)

        # 5. Feed-forward
        residual = x
        x = residual + self.ff(self.norm_ff(x, t_embed))

        return x
