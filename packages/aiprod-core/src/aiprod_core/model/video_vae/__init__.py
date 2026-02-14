# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
HWVAE — Hierarchical Wavelet Video Auto-Encoder

AIPROD's proprietary video VAE using wavelet decomposition for
multi-scale latent compression.

Key differences from standard video VAEs:
    - Wavelet-based spatial decomposition (not plain conv downsampling)
    - Hierarchical latent space (coarse + fine details)
    - Separable 2D+1D convolutions (not causal 3D convolutions)
    - Configurable compression ratios per axis
    - Progressive encoding with skip connections
"""

from .encoder import HWVAEEncoder
from .decoder import HWVAEDecoder
from .config import HWVAEConfig

__all__ = [
    "HWVAEEncoder",
    "HWVAEDecoder",
    "HWVAEConfig",
]
