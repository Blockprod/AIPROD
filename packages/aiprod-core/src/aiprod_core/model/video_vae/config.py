# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
HWVAE Configuration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class HWVAEConfig:
    """Configuration for the Hierarchical Wavelet VAE.

    Architecture design:
        - Uses Haar wavelet decomposition for spatial downsampling
        - Separable conv: 2D spatial then 1D temporal (not 3D causal)
        - Latent channels configurable (default 64, distinct from 128-ch designs)
        - Compression: spatial 8× (3 wavelet levels), temporal 7×
    """
    # Input
    in_channels: int = 3
    # Latent space
    latent_channels: int = 64
    # Encoder channel progression
    encoder_channels: list[int] | tuple[int, ...] = (128, 256, 384, 512)
    # Decoder channel progression (reverse of encoder)
    decoder_channels: list[int] | tuple[int, ...] = (512, 384, 256, 128)
    # Number of residual blocks per level
    blocks_per_level: int = 2
    # Compression ratios
    spatial_factor: int = 8    # 512px → 64 latent positions
    temporal_factor: int = 7   # 49 frames → 7 latent frames
    # Wavelet type
    wavelet: Literal["haar", "db2", "bior"] = "haar"
    # Training
    kl_weight: float = 1e-5
    perceptual_weight: float = 1.0
    # Tiled decoding
    tile_spatial: int = 256
    tile_temporal: int = 17
    tile_overlap: float = 0.25
