# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
HWVAE Encoder — Wavelet-based Video Encoder

Encodes pixel-space video [B, C, T, H, W] into a compact latent
representation [B, latent_ch, T', H', W'] using:
    1. Haar wavelet decomposition for spatial downsampling
    2. Separable 2D+1D convolutions (not 3D causal convolutions)
    3. Progressive channel expansion with residual blocks
    4. Final projection to latent mean and log-variance (VAE)
"""

from __future__ import annotations

from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HWVAEConfig


class SeparableConv(nn.Module):
    """Separable convolution: 2D spatial + 1D temporal.

    More efficient than 3D convolution and avoids causal padding issues.
    Processes spatial dimensions first, then temporal.
    """

    def __init__(self, in_ch: int, out_ch: int, spatial_kernel: int = 3, temporal_kernel: int = 3):
        super().__init__()
        self.spatial = nn.Conv2d(
            in_ch, out_ch, kernel_size=spatial_kernel,
            padding=spatial_kernel // 2, bias=False,
        )
        self.temporal = nn.Conv1d(
            out_ch, out_ch, kernel_size=temporal_kernel,
            padding=temporal_kernel // 2, bias=False,
        )
        self.norm = nn.GroupNorm(min(32, out_ch), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            [B, C_out, T, H, W]
        """
        B, C, T, H, W = x.shape

        # Spatial conv: merge B and T, apply 2D conv
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.spatial(x)
        C_out = x.shape[1]
        x = x.view(B, T, C_out, H, W).permute(0, 2, 1, 3, 4)  # [B, C_out, T, H, W]

        # Temporal conv: merge B, H, W, apply 1D conv over T
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C_out, T)
        x = self.temporal(x)
        x = x.view(B, H, W, C_out, T).permute(0, 3, 4, 1, 2)  # [B, C_out, T, H, W]

        # Normalize
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C_out, H, W)
        x = self.norm(x)
        x = x.view(B, T, C_out, H, W).permute(0, 2, 1, 3, 4)

        return x


class ResidualBlock(nn.Module):
    """Residual block with separable convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            SeparableConv(channels, channels),
            nn.SiLU(),
            SeparableConv(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class HaarWaveletDown(nn.Module):
    """Haar wavelet spatial downsampling (2× in H and W).

    Decomposes the spatial dimensions into LL (low-low), LH, HL, HH
    subbands. This is mathematically equivalent to learned downsampling
    but provides a principled multi-scale decomposition.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Project the 4 subbands (LL, LH, HL, HH) to output channels
        self.proj = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] with H, W even.
        Returns:
            [B, out_ch, T, H//2, W//2]
        """
        B, C, T, H, W = x.shape

        # Reshape for 2D wavelet: merge B and T
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # Haar wavelet decomposition
        x_ll = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] +
                x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 4.0
        x_lh = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] +
                x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) / 4.0
        x_hl = (x[:, :, 0::2, 0::2] + x[:, :, 0::2, 1::2] -
                x[:, :, 1::2, 0::2] - x[:, :, 1::2, 1::2]) / 4.0
        x_hh = (x[:, :, 0::2, 0::2] - x[:, :, 0::2, 1::2] -
                x[:, :, 1::2, 0::2] + x[:, :, 1::2, 1::2]) / 4.0

        # Stack subbands: [B*T, 4C, H/2, W/2]
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

        # Project to output channels
        x = self.proj(x)
        C_out = x.shape[1]

        # Reshape back: [B, C_out, T, H/2, W/2]
        x = x.view(B, T, C_out, H // 2, W // 2).permute(0, 2, 1, 3, 4)
        return x


class TemporalPool(nn.Module):
    """Temporal downsampling via strided 1D convolution."""

    def __init__(self, channels: int, factor: int = 7):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv1d(channels, channels, kernel_size=factor, stride=factor, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W]
        Returns:
            [B, C, T//factor, H, W]
        """
        B, C, T, H, W = x.shape
        # Merge spatial dims: [B*H*W, C, T]
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        x = self.conv(x)
        T_out = x.shape[2]
        x = x.view(B, H, W, C, T_out).permute(0, 3, 4, 1, 2)  # [B, C, T_out, H, W]
        return x


class HWVAEEncoder(nn.Module):
    """Hierarchical Wavelet VAE Encoder.

    Architecture:
        1. Initial convolution (3 → base_channels)
        2. Progressive wavelet downsampling (3 levels = 8× spatial)
        3. Temporal pooling (7× temporal compression)
        4. Residual blocks at each level
        5. Final projection to latent mean and log-variance

    Total compression: [B, 3, 49, 512, 768] → [B, 64, 7, 64, 96]
    """

    def __init__(self, config: HWVAEConfig):
        super().__init__()
        self.config = config
        ch = config.encoder_channels

        # Initial conv
        self.conv_in = SeparableConv(config.in_channels, ch[0])

        # Downsampling levels (wavelet spatial + residual blocks)
        self.down_blocks = nn.ModuleList()
        for i in range(len(ch) - 1):
            self.down_blocks.append(nn.ModuleDict({
                "wavelet": HaarWaveletDown(ch[i], ch[i + 1]),
                "residual": nn.Sequential(*[
                    ResidualBlock(ch[i + 1]) for _ in range(config.blocks_per_level)
                ]),
            }))

        # Temporal compression
        self.temporal_pool = TemporalPool(ch[-1], factor=config.temporal_factor)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(ch[-1]),
            ResidualBlock(ch[-1]),
        )

        # To latent (mean + log_var)
        self.to_latent = SeparableConv(ch[-1], config.latent_channels * 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode video to latent distribution parameters.

        Args:
            x: [B, C, T, H, W] pixel-space video in [-1, 1].

        Returns:
            mean: [B, latent_ch, T', H', W'] latent mean.
            log_var: [B, latent_ch, T', H', W'] latent log-variance.
        """
        x = self.conv_in(x)
        x = F.silu(x)

        # Progressive spatial downsampling
        for down in self.down_blocks:
            x = down["wavelet"](x)
            x = F.silu(x)
            x = down["residual"](x)

        # Temporal compression
        x = self.temporal_pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Split into mean and log_var
        stats = self.to_latent(x)
        mean, log_var = stats.chunk(2, dim=1)

        return mean, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and sample from the latent distribution.

        Args:
            x: [B, C, T, H, W] pixel-space video.

        Returns:
            [B, latent_ch, T', H', W'] sampled latent.
        """
        mean, log_var = self.forward(x)

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mean + std * eps
        else:
            return mean
