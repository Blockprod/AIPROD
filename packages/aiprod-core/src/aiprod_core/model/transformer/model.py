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
    latent_channels: int = 128      # must match VIDEO_LATENT_CHANNELS in pipeline
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

    def _build_audio_projections(
        self, audio_channels: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Lazily create audio input/output projections on first use."""
        if not hasattr(self, "_audio_embed"):
            self._audio_embed = nn.Sequential(
                nn.Linear(audio_channels, self.config.hidden_dim),
                AdaptiveRMSNorm(self.config.hidden_dim),
            ).to(device=device, dtype=dtype)
        if not hasattr(self, "_audio_out_proj"):
            self._audio_out_proj = nn.Linear(
                self.config.hidden_dim, audio_channels
            ).to(device=device, dtype=dtype)

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

    def forward_multimodal(
        self,
        video_latent: torch.Tensor,
        audio_latent: Optional[torch.Tensor] = None,
        video_positions: Optional[torch.Tensor] = None,
        audio_positions: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Multimodal forward pass for joint video+audio denoising.

        Accepts **pre-patchified** inputs from the pipeline where video and
        audio tokens are already in sequence form ``[B, seq, C]``.

        Args:
            video_latent: [B, video_seq, C_video] patchified video tokens.
            audio_latent: [B, audio_seq, C_audio] patchified audio tokens (optional).
            video_positions: [B, 3, video_seq, 2] position coordinates (unused in compact blocks).
            audio_positions: [B, ndim, audio_seq, 2] audio positions (unused).
            context: [B, text_seq, D_text] text encoder embeddings.
            timestep: [B] or scalar noise level.

        Returns:
            ``(denoised_video, denoised_audio)`` — same shapes as inputs.
        """
        B = video_latent.shape[0]
        video_seq = video_latent.shape[1]

        # Project video tokens to hidden dim
        x_video = self.patch_embed(video_latent)  # [B, video_seq, D]

        # Handle audio modality
        audio_seq = 0
        if audio_latent is not None:
            audio_ch = audio_latent.shape[-1]
            self._build_audio_projections(audio_ch, audio_latent.device, audio_latent.dtype)
            x_audio = self._audio_embed(audio_latent)  # [B, audio_seq, D]
            audio_seq = audio_latent.shape[1]
            x = torch.cat([x_video, x_audio], dim=1)  # [B, total_seq, D]
        else:
            x = x_video

        # Timestep conditioning
        if timestep is not None:
            if timestep.dim() == 0:
                timestep = timestep.unsqueeze(0).expand(B)
            t_embed = self._get_timestep_embedding(timestep)
        else:
            t_embed = torch.zeros(B, self.config.hidden_dim, device=x.device, dtype=x.dtype)

        # Text conditioning
        if context is not None:
            text_ctx = self.text_proj(context)
        else:
            text_ctx = torch.zeros(B, 1, self.config.hidden_dim, device=x.device, dtype=x.dtype)

        # Process through transformer blocks (treat full sequence as spatial)
        total_seq = x.shape[1]
        for block in self.blocks:
            x = block(x, t_embed=t_embed, text_ctx=text_ctx, video_shape=(1, 1, total_seq))

        # Output projections
        x = self.out_norm(x, t_embed)

        video_out = self.out_proj(x[:, :video_seq, :])  # [B, video_seq, C_latent]

        audio_out = None
        if audio_latent is not None and audio_seq > 0:
            audio_hidden = x[:, video_seq:, :]  # [B, audio_seq, D_hidden]
            audio_out = self._audio_out_proj(audio_hidden)  # [B, audio_seq, C_audio]

        return video_out, audio_out
