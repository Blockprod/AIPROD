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

# Backward-compatible aliases
VideoEncoder = HWVAEEncoder
VideoDecoder = HWVAEDecoder


def __getattr__(name: str):
    """Lazy imports to avoid circular dependency with configurators."""
    _configurator_names = {
        "VideoEncoderConfigurator",
        "VideoDecoderConfigurator",
        "VAE_ENCODER_COMFY_KEYS_FILTER",
        "VAE_DECODER_COMFY_KEYS_FILTER",
    }
    if name in _configurator_names:
        from aiprod_core.model import configurators as _cfg
        return getattr(_cfg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def decode_video(
    latent,
    decoder,
    tiling_config=None,
    generator=None,
):
    """Decode video latent to pixel space.

    Simple wrapper around the decoder for pipeline compatibility.
    ``tiling_config`` and ``generator`` are accepted for backward
    compatibility but currently ignored (non-tiled decode).
    """
    return decoder(latent)


def decode_audio(
    latent,
    decoder,
    vocoder=None,
):
    """Decode audio latent to waveform.

    ``vocoder`` is accepted for backward compat; when the NAC codec
    is used, the decoder itself handles the full pipeline.
    """
    if vocoder is not None and vocoder is not decoder:
        intermediate = decoder(latent)
        return vocoder(intermediate)
    return decoder(latent)


# ── Tiling helpers (stubs — kept for pipeline API compat) ─────────────────

from dataclasses import dataclass as _dataclass


@_dataclass
class TilingConfig:
    """Stub tiling configuration.

    Full tiled decoding is not yet implemented; this exists so that
    pipeline code that passes ``TilingConfig.default()`` continues
    to parse without errors.
    """
    spatial_tiles: int = 1
    temporal_tiles: int = 1

    @classmethod
    def default(cls) -> "TilingConfig":
        return cls()


def get_video_chunks_number(num_frames: int, tiling_config=None) -> int:
    """Return the number of temporal chunks for video encoding.

    Stub implementation — returns 1 (no chunking).
    """
    if tiling_config is not None and tiling_config.temporal_tiles > 1:
        return tiling_config.temporal_tiles
    return 1


__all__ = [
    "HWVAEEncoder",
    "HWVAEDecoder",
    "HWVAEConfig",
    "VideoEncoder",
    "VideoDecoder",
    "VideoEncoderConfigurator",
    "VideoDecoderConfigurator",
    "VAE_ENCODER_COMFY_KEYS_FILTER",
    "VAE_DECODER_COMFY_KEYS_FILTER",
    "decode_video",
    "decode_audio",
    "TilingConfig",
    "get_video_chunks_number",
]
