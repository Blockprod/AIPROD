# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
HWVAE Decoder — Wavelet-based Video Decoder

Decodes latent representations back to pixel-space video using
inverse wavelet transforms and progressive upsampling.

Supports tiled decoding for memory-efficient inference on
high-resolution or long videos.
"""

from __future__ import annotations

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HWVAEConfig
from .encoder import SeparableConv, ResidualBlock


class InverseHaarWaveletUp(nn.Module):
    """Inverse Haar wavelet spatial upsampling (2× in H and W).

    Projects channels to 4 subbands, then reconstructs the
    high-resolution spatial representation.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Project to 4 subbands for reconstruction
        self.proj = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            [B, out_ch, T, H*2, W*2]
        """
        B, C, T, H, W = x.shape

        # Merge B and T for 2D processing
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # Project to 4× channels
        x = self.proj(x)  # [B*T, out_ch*4, H, W]
        out_ch = x.shape[1] // 4

        # Pixel shuffle: [B*T, out_ch*4, H, W] → [B*T, out_ch, H*2, W*2]
        x = F.pixel_shuffle(x, 2)

        # Reshape back
        x = x.view(B, T, out_ch, H * 2, W * 2).permute(0, 2, 1, 3, 4)
        return x


class TemporalUpsample(nn.Module):
    """Temporal upsampling via transposed 1D convolution."""

    def __init__(self, channels: int, factor: int = 7):
        super().__init__()
        self.factor = factor
        self.conv_t = nn.ConvTranspose1d(
            channels, channels, kernel_size=factor, stride=factor, bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            [B, C, T*factor, H, W]
        """
        B, C, T, H, W = x.shape
        # Merge spatial dims: [B*H*W, C, T]
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        x = self.conv_t(x)
        T_out = x.shape[2]
        x = x.view(B, H, W, C, T_out).permute(0, 3, 4, 1, 2)  # [B, C, T_out, H, W]
        return x


class HWVAEDecoder(nn.Module):
    """Hierarchical Wavelet VAE Decoder.

    Architecture:
        1. Initial projection (latent_channels → top_channels)
        2. Temporal upsampling (7× temporal expansion)
        3. Bottleneck residual blocks
        4. Progressive inverse wavelet upsampling (3 levels = 8× spatial)
        5. Final convolution to pixel space

    Supports tiled decoding for memory-constrained inference.
    """

    def __init__(self, config: HWVAEConfig):
        super().__init__()
        self.config = config
        ch = config.decoder_channels  # (512, 384, 256, 128)

        # From latent to top-level channels
        self.conv_in = SeparableConv(config.latent_channels, ch[0])

        # Temporal upsampling
        self.temporal_up = TemporalUpsample(ch[0], factor=config.temporal_factor)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(ch[0]),
            ResidualBlock(ch[0]),
        )

        # Upsampling levels (inverse wavelet + residual blocks)
        self.up_blocks = nn.ModuleList()
        for i in range(len(ch) - 1):
            self.up_blocks.append(nn.ModuleDict({
                "wavelet_up": InverseHaarWaveletUp(ch[i], ch[i + 1]),
                "residual": nn.Sequential(*[
                    ResidualBlock(ch[i + 1]) for _ in range(config.blocks_per_level)
                ]),
            }))

        # To pixel space
        self.conv_out = nn.Sequential(
            SeparableConv(ch[-1], ch[-1]),
            nn.SiLU(),
            SeparableConv(ch[-1], config.in_channels),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to pixel-space video.

        Args:
            z: [B, latent_ch, T', H', W'] latent representation.

        Returns:
            [B, C, T, H, W] pixel-space video in [-1, 1].
        """
        x = self.conv_in(z)
        x = F.silu(x)

        # Temporal upsampling
        x = self.temporal_up(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Progressive spatial upsampling
        for up in self.up_blocks:
            x = up["wavelet_up"](x)
            x = F.silu(x)
            x = up["residual"](x)

        # To pixel space
        x = self.conv_out(x)
        return x

    def decode_tiled(
        self,
        z: torch.Tensor,
        tile_t: Optional[int] = None,
        tile_h: Optional[int] = None,
        tile_w: Optional[int] = None,
        overlap: Optional[float] = None,
    ) -> torch.Tensor:
        """Memory-efficient tiled decoding.

        Splits the latent into overlapping tiles, decodes each,
        and blends them together with linear interpolation in
        the overlap regions.

        Args:
            z: [B, latent_ch, T', H', W'] latent representation.
            tile_t: Temporal tile size (in latent frames).
            tile_h: Height tile size (in latent positions).
            tile_w: Width tile size (in latent positions).
            overlap: Fraction of tile that overlaps with neighbors.

        Returns:
            [B, C, T, H, W] pixel-space video.
        """
        cfg = self.config
        tile_t = tile_t or (cfg.tile_temporal // cfg.temporal_factor)
        tile_h = tile_h or (cfg.tile_spatial // cfg.spatial_factor)
        tile_w = tile_w or (cfg.tile_spatial // cfg.spatial_factor)
        overlap = overlap or cfg.tile_overlap

        B, C, T, H, W = z.shape

        # If the tensor fits in one tile, just decode directly
        if T <= tile_t and H <= tile_h and W <= tile_w:
            return self.forward(z)

        # Calculate output dimensions
        out_T = T * cfg.temporal_factor
        out_H = H * cfg.spatial_factor
        out_W = W * cfg.spatial_factor

        output = torch.zeros(B, cfg.in_channels, out_T, out_H, out_W,
                             device=z.device, dtype=z.dtype)
        count = torch.zeros(1, 1, out_T, out_H, out_W,
                            device=z.device, dtype=z.dtype)

        # Calculate step sizes
        step_t = max(1, int(tile_t * (1 - overlap)))
        step_h = max(1, int(tile_h * (1 - overlap)))
        step_w = max(1, int(tile_w * (1 - overlap)))

        for t_start in range(0, T, step_t):
            for h_start in range(0, H, step_h):
                for w_start in range(0, W, step_w):
                    t_end = min(t_start + tile_t, T)
                    h_end = min(h_start + tile_h, H)
                    w_end = min(w_start + tile_w, W)

                    tile = z[:, :, t_start:t_end, h_start:h_end, w_start:w_end]
                    decoded = self.forward(tile)

                    # Map to output coordinates
                    ot_s = t_start * cfg.temporal_factor
                    ot_e = ot_s + decoded.shape[2]
                    oh_s = h_start * cfg.spatial_factor
                    oh_e = oh_s + decoded.shape[3]
                    ow_s = w_start * cfg.spatial_factor
                    ow_e = ow_s + decoded.shape[4]

                    output[:, :, ot_s:ot_e, oh_s:oh_e, ow_s:ow_e] += decoded
                    count[:, :, ot_s:ot_e, oh_s:oh_e, ow_s:ow_e] += 1.0

        return output / count.clamp(min=1.0)
