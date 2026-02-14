# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
Latent Upsampler — Learnable latent super-resolution

Upsamples low-resolution latents to higher resolution before
VAE decoding, enabling two-stage generation at lower compute cost.

NOTE: Full learned upsampler not yet implemented; the stub performs
bilinear 2× upsampling so that two-stage pipeline code can run
end-to-end without crashing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentUpsampler(nn.Module):
    """Stub latent upsampler.

    Performs bilinear 2× spatial upsampling as a placeholder until the
    learned upsampler is trained and integrated.
    """

    def __init__(self, scale_factor: int = 2) -> None:
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Upsample latent by ``scale_factor`` along spatial dims."""
        if latent.ndim == 3:
            # [B, seq, C] — reshape needed
            return latent
        if latent.ndim == 5:
            # [B, C, T, H, W]
            b, c, t, h, w = latent.shape
            latent = latent.reshape(b * t, c, h, w)
            latent = F.interpolate(latent, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
            return latent.reshape(b, c, t, h * self.scale_factor, w * self.scale_factor)
        # [B, C, H, W]
        return F.interpolate(latent, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)


class LatentUpsamplerConfigurator:
    """Stub configurator for the latent upsampler.

    Returns a default ``LatentUpsampler`` regardless of checkpoint
    contents.  Will be replaced by a real configurator once the
    learned upsampler model is available.
    """

    @staticmethod
    def from_state_dict(state_dict, *, device=None, dtype=None, **kwargs):
        _device = device or torch.device("cpu")
        _dtype = dtype or torch.float32
        return LatentUpsampler().to(device=_device, dtype=_dtype)

    @staticmethod
    def build_model(state_dict, *, device=None, dtype=None, **kwargs):
        return LatentUpsamplerConfigurator.from_state_dict(state_dict, device=device, dtype=dtype)


def upsample_video(
    latent: torch.Tensor,
    video_encoder=None,
    upsampler: LatentUpsampler | None = None,
) -> torch.Tensor:
    """Spatially upsample a video latent.

    Args:
        latent: Video latent tensor.
        video_encoder: Accepted for API compat, not used.
        upsampler: The upsampler module (defaults to bilinear 2×).

    Returns:
        Upsampled latent tensor.
    """
    if upsampler is None:
        upsampler = LatentUpsampler()
    return upsampler(latent)


__all__ = [
    "LatentUpsampler",
    "LatentUpsamplerConfigurator",
    "upsample_video",
]

