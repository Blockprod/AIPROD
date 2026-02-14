# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
SHDT Attention Mechanisms

AIPROD's proprietary attention implementations:
    - GroupedQueryAttention (GQA): memory-efficient multi-head attention
    - SpatialAttention: within-frame attention (H*W tokens)
    - TemporalAttention: across-frame attention (T tokens per position)
    - CrossModalAttention: text-to-video cross-attention
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA).

    Uses fewer key-value heads than query heads, reducing memory and
    compute while maintaining quality. This is distinct from standard
    multi-head attention (MHA) used in most diffusion transformers.

    Args:
        dim: Model hidden dimension.
        num_heads: Number of query heads.
        num_kv_heads: Number of key/value heads (must divide num_heads).
        head_dim: Dimension per head.
        dropout: Attention dropout rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 16,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_group_size = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] query tokens.
            context: [B, M, D] key/value tokens (self-attention if None).
            mask: [B, M] attention mask for context tokens.

        Returns:
            [B, N, D] attention output.
        """
        B, N, _ = x.shape
        ctx = context if context is not None else x
        M = ctx.shape[1]

        # Project Q, K, V
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(ctx).view(B, M, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(ctx).view(B, M, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand KV heads for GQA: [B, num_kv_heads, M, D] → [B, num_heads, M, D]
        if self.kv_group_size > 1:
            k = k.repeat_interleave(self.kv_group_size, dim=1)
            v = v.repeat_interleave(self.kv_group_size, dim=1)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :].expand(B, self.num_heads, N, M)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim)
        return self.out_proj(out)


class SpatialAttention(nn.Module):
    """Attention within a single frame (across H*W positions).

    Processes each frame independently, enabling per-frame spatial
    understanding. This is one stream of SHDT's dual-stream design.
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, head_dim, dropout)

    def forward(self, x: torch.Tensor, video_shape: tuple[int, int, int]) -> torch.Tensor:
        """
        Args:
            x: [B, T*H*W, D] flattened video tokens.
            video_shape: (T, H, W) spatial-temporal dimensions.

        Returns:
            [B, T*H*W, D] with spatial attention applied per-frame.
        """
        B, N, D = x.shape
        T, H, W = video_shape

        # Reshape to [B*T, H*W, D] for per-frame attention
        x = x.view(B, T, H * W, D).reshape(B * T, H * W, D)
        x = self.attn(x)
        return x.view(B, T, H * W, D).reshape(B, N, D)


class TemporalAttention(nn.Module):
    """Attention across frames at each spatial position.

    Processes each spatial position independently across time,
    enabling temporal coherence. The second stream of SHDT.
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, head_dim, dropout)

    def forward(self, x: torch.Tensor, video_shape: tuple[int, int, int]) -> torch.Tensor:
        """
        Args:
            x: [B, T*H*W, D] flattened video tokens.
            video_shape: (T, H, W) spatial-temporal dimensions.

        Returns:
            [B, T*H*W, D] with temporal attention applied per-position.
        """
        B, N, D = x.shape
        T, H, W = video_shape

        # Reshape to [B*H*W, T, D] for per-position temporal attention
        x = x.view(B, T, H * W, D).permute(0, 2, 1, 3).reshape(B * H * W, T, D)
        x = self.attn(x)
        return x.view(B, H * W, T, D).permute(0, 2, 1, 3).reshape(B, N, D)


class CrossModalAttention(nn.Module):
    """Cross-attention from video tokens to text embeddings.

    Video tokens attend to text context, incorporating language
    conditioning into the generation process.
    """

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, head_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        text_ctx: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] video tokens.
            text_ctx: [B, S, D] text embeddings.
            text_mask: [B, S] attention mask.

        Returns:
            [B, N, D] video tokens with text conditioning.
        """
        return self.attn(x, context=text_ctx, mask=text_mask)
