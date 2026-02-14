# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
SHDT Model — Scalable Hybrid Diffusion Transformer

The main denoising model for AIPROD's video generation pipeline.
Uses a dual-stream spatial-temporal architecture with configurable depth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import torch.nn as nn

from .block import SHDTBlock
from .position import LearnedPositionalEncoding3D
from .norm import AdaptiveRMSNorm


@dataclass
class SHDTConfig:
    """Configuration for the Scalable Hybrid Diffusion Transformer.

    Design choices distinct from existing architectures:
        - `num_layers` is not hardcoded — allows scaling from tiny (8) to large (64)
        - `num_kv_heads` enables GQA (fewer KV heads than Q heads)
        - `dual_stream_ratio` controls spatial vs temporal layer allocation
        - `adaptive_compute` enables dynamic depth per-sample
    """
    # Core dimensions
    hidden_dim: int = 2048
    num_heads: int = 16
    num_kv_heads: int = 4          # GQA: fewer KV heads for efficiency
    head_dim: int = 128
    num_layers: int = 32
    ff_multiplier: float = 4.0

    # Dual-stream control
    dual_stream_ratio: float = 0.5  # fraction of layers that are spatial-only
    cross_stream_interval: int = 4   # cross-attention between streams every N layers

    # Input specs
    latent_channels: int = 64
    text_embed_dim: int = 2048      # from LLM bridge output
    max_caption_tokens: int = 512

    # Video dimensions (latent space)
    max_frames: int = 64
    max_height: int = 128
    max_width: int = 128

    # Advanced features
    adaptive_compute: bool = False   # dynamic depth per frame
    dropout: float = 0.0
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"

    @property
    def ff_dim(self) -> int:
        return int(self.hidden_dim * self.ff_multiplier)

    @property
    def num_spatial_layers(self) -> int:
        return int(self.num_layers * self.dual_stream_ratio)

    @property
    def num_temporal_layers(self) -> int:
        return self.num_layers - self.num_spatial_layers


class SHDTModel(nn.Module):
    """Scalable Hybrid Diffusion Transformer — AIPROD's proprietary backbone.

    Architecture:
        1. Patch embedding: latent [B,C,T,H,W] → token sequence [B,N,D]
        2. Timestep + text conditioning via adaptive normalization
        3. Dual-stream processing:
           - Spatial stream: self-attention within each frame
           - Temporal stream: self-attention across frames at each position
           - Cross-stream attention every `cross_stream_interval` layers
        4. Cross-attention to text embeddings at every layer
        5. Output projection back to latent space

    This is fundamentally different from single-stream DiT architectures.
    """

    def __init__(self, config: SHDTConfig):
        super().__init__()
        self.config = config

        # ── Patch embedding (latent channels → hidden dim) ──
        self.patch_embed = nn.Sequential(
            nn.Linear(config.latent_channels, config.hidden_dim),
            AdaptiveRMSNorm(config.hidden_dim),
        )

        # ── Positional encoding (learned, 3D) ──
        self.pos_encoding = LearnedPositionalEncoding3D(
            dim=config.hidden_dim,
            max_t=config.max_frames,
            max_h=config.max_height,
            max_w=config.max_width,
        )

        # ── Timestep embedding ──
        self.timestep_embed = nn.Sequential(
            nn.Linear(256, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # ── Text projection (from LLM Bridge output) ──
        self.text_proj = nn.Linear(config.text_embed_dim, config.hidden_dim)

        # ── Transformer blocks (dual-stream) ──
        self.blocks = nn.ModuleList([
            SHDTBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_kv_heads=config.num_kv_heads,
                head_dim=config.head_dim,
                ff_dim=config.ff_dim,
                layer_idx=i,
                is_cross_stream=(i % config.cross_stream_interval == 0),
                dropout=config.dropout,
            )
            for i in range(config.num_layers)
        ])

        # ── Output projection (hidden dim → latent channels) ──
        self.out_norm = AdaptiveRMSNorm(config.hidden_dim)
        self.out_proj = nn.Linear(config.hidden_dim, config.latent_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Custom weight initialization for stable diffusion training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sinusoidal timestep embedding with learned projection.

        Args:
            timesteps: [B] tensor of noise levels in [0, 1].

        Returns:
            [B, D] timestep embedding.
        """
        half_dim = 128
        freqs = torch.exp(
            -torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            * (torch.log(torch.tensor(10000.0)) / half_dim)
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.timestep_embed(embedding)

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass — predict noise/velocity from noisy latent.

        Args:
            latent: [B, C, T, H, W] noisy latent video.
            timestep: [B] noise level for each sample.
            text_embeds: [B, S, D_text] text conditioning from LLM Bridge.
            text_mask: [B, S] attention mask for text tokens.

        Returns:
            [B, C, T, H, W] predicted noise/velocity.
        """
        B, C, T, H, W = latent.shape

        # Reshape to token sequence: [B, C, T, H, W] → [B, T*H*W, C]
        x = latent.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)

        # Patch embedding + positional encoding
        x = self.patch_embed(x)
        x = x + self.pos_encoding(T, H, W, device=x.device, dtype=x.dtype)

        # Conditioning
        t_embed = self._get_timestep_embedding(timestep)
        text_ctx = self.text_proj(text_embeds)

        # Process through transformer blocks
        for block in self.blocks:
            x = block(
                x,
                t_embed=t_embed,
                text_ctx=text_ctx,
                text_mask=text_mask,
                video_shape=(T, H, W),
            )

        # Output projection
        x = self.out_norm(x, t_embed)
        x = self.out_proj(x)

        # Reshape back: [B, T*H*W, C] → [B, C, T, H, W]
        x = x.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

        return x
