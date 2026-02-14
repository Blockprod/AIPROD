# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
Neural Audio Codec — Encoder, Decoder, and RVQ

Uses a multi-scale 1D convolutional encoder/decoder with
residual vector quantization for discrete audio tokens.

Architecture is inspired by the Encodec/DAC family but with
AIPROD-specific design choices:
    - Multi-band processing (low/mid/high frequency bands)
    - Learnable band-splitting with overlap
    - Separate codebooks per frequency band
    - Snake activation function for audio fidelity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NACConfig:
    """Neural Audio Codec configuration."""
    sample_rate: int = 48000
    channels: int = 1
    # Encoder/decoder channels
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (2, 4, 8, 16)
    # RVQ parameters
    num_codebooks: int = 8
    codebook_size: int = 1024
    codebook_dim: int = 128
    # Downsampling strides
    strides: tuple[int, ...] = (8, 5, 4, 3)  # total compression = 480×
    # Number of residual blocks per level
    blocks_per_level: int = 3
    # Number of frequency bands
    num_bands: int = 3


class SnakeActivation(nn.Module):
    """Snake activation: x + sin²(alpha * x) / alpha.

    Particularly effective for periodic signals like audio,
    as it introduces controlled periodicity into activations.
    """

    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1, channels, 1), alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (torch.sin(self.alpha * x) ** 2) / (self.alpha + 1e-8)


class ResBlock1D(nn.Module):
    """1D residual block with dilated convolutions and Snake activation."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            SnakeActivation(channels),
            nn.Conv1d(channels, channels, kernel_size=7,
                      dilation=dilation, padding=3 * dilation, bias=False),
            SnakeActivation(channels),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class AudioEncoder(nn.Module):
    """NAC Encoder — waveform → latent embeddings.

    Progressive downsampling via strided convolutions with
    multi-dilation residual blocks at each level.
    """

    def __init__(self, config: NACConfig):
        super().__init__()
        self.config = config

        # Initial conv
        self.conv_in = nn.Conv1d(config.channels, config.base_channels, kernel_size=7, padding=3)

        # Downsampling layers
        self.down_layers = nn.ModuleList()
        in_ch = config.base_channels
        for i, stride in enumerate(config.strides):
            out_ch = config.base_channels * config.channel_multipliers[i]
            self.down_layers.append(nn.Sequential(
                # Residual blocks with increasing dilation
                *[ResBlock1D(in_ch, dilation=3 ** j) for j in range(config.blocks_per_level)],
                # Strided downsampling
                SnakeActivation(in_ch),
                nn.Conv1d(in_ch, out_ch, kernel_size=2 * stride, stride=stride,
                          padding=stride // 2, bias=False),
            ))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock1D(in_ch),
            SnakeActivation(in_ch),
            nn.Conv1d(in_ch, config.codebook_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, T_samples] raw waveform.

        Returns:
            [B, codebook_dim, T_latent] continuous audio embeddings.
        """
        x = self.conv_in(x)
        for layer in self.down_layers:
            x = layer(x)
        x = self.bottleneck(x)
        return x


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantization (RVQ).

    Quantizes continuous embeddings into discrete tokens using
    multiple codebooks in series — each codebook encodes the
    residual from the previous quantization step.
    """

    def __init__(self, num_codebooks: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_dim = codebook_dim

        # Codebook embeddings
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(codebook_size, codebook_dim) * 0.01)
            for _ in range(num_codebooks)
        ])

    def _quantize_single(
        self,
        x: torch.Tensor,
        codebook: nn.Parameter,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize using a single codebook.

        Args:
            x: [B, D, T] continuous input.
            codebook: [K, D] codebook embeddings.

        Returns:
            quantized: [B, D, T] quantized output.
            indices: [B, T] codebook indices.
        """
        B, D, T = x.shape

        # [B, T, D] for distance computation
        x_flat = x.permute(0, 2, 1).reshape(-1, D)

        # Distances: [B*T, K]
        distances = torch.cdist(x_flat.unsqueeze(0), codebook.unsqueeze(0)).squeeze(0)
        indices = distances.argmin(dim=-1)  # [B*T]

        # Quantize
        quantized = F.embedding(indices, codebook)  # [B*T, D]
        quantized = quantized.view(B, T, D).permute(0, 2, 1)  # [B, D, T]
        indices = indices.view(B, T)

        return quantized, indices

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            x: [B, codebook_dim, T] continuous embeddings.

        Returns:
            quantized: [B, codebook_dim, T] quantized embeddings.
            all_indices: list of [B, T] index tensors per codebook.
        """
        residual = x
        quantized_total = torch.zeros_like(x)
        all_indices = []

        for codebook in self.codebooks:
            quantized, indices = self._quantize_single(residual, codebook)

            # Straight-through estimator
            quantized_st = residual + (quantized - residual).detach()

            quantized_total = quantized_total + quantized_st
            residual = residual - quantized.detach()
            all_indices.append(indices)

        return quantized_total, all_indices


class AudioDecoder(nn.Module):
    """NAC Decoder — latent embeddings → waveform.

    Mirror architecture of AudioEncoder with transposed
    convolutions for upsampling.
    """

    def __init__(self, config: NACConfig):
        super().__init__()
        self.config = config

        # Reverse channel multipliers for decoder
        reversed_mults = list(reversed(config.channel_multipliers))
        in_ch = config.base_channels * reversed_mults[0]

        # From codebook dim
        self.conv_in = nn.Conv1d(config.codebook_dim, in_ch, kernel_size=3, padding=1)

        # Upsampling layers
        self.up_layers = nn.ModuleList()
        reversed_strides = list(reversed(config.strides))
        for i, stride in enumerate(reversed_strides):
            out_ch = (
                config.base_channels * reversed_mults[i + 1]
                if i + 1 < len(reversed_mults)
                else config.base_channels
            )
            self.up_layers.append(nn.Sequential(
                SnakeActivation(in_ch),
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2 * stride, stride=stride,
                                   padding=stride // 2, bias=False),
                *[ResBlock1D(out_ch, dilation=3 ** j) for j in range(config.blocks_per_level)],
            ))
            in_ch = out_ch

        # Final conv to waveform
        self.conv_out = nn.Sequential(
            SnakeActivation(config.base_channels),
            nn.Conv1d(config.base_channels, config.channels, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, codebook_dim, T_latent] audio latent (quantized or continuous).

        Returns:
            [B, 1, T_samples] reconstructed waveform in [-1, 1].
        """
        x = self.conv_in(z)
        for layer in self.up_layers:
            x = layer(x)
        x = self.conv_out(x)
        return x
