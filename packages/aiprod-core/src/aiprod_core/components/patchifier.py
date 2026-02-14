# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Patchifiers — Convert between spatial and sequence representations.

Patchifiers flatten 3-D (video) or 1-D (audio) latent grids into sequences
of tokens for the transformer backbone, and reconstruct the spatial layout
after denoising.

Design:
    - VideoLatentPatchifier: [B, C, T, H, W] ↔ [B, T*H*W, C]
    - AudioPatchifier: [B, C, L] ↔ [B, L, C]
    - get_pixel_coords: Generate normalised coordinate grids for positional encoding
"""

from __future__ import annotations

from dataclasses import replace

import torch

from aiprod_core.types import (
    AudioLatentShape,
    LatentState,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    VideoPixelShape,
)


class VideoLatentPatchifier:
    """Flatten / unflatten video latents for the transformer.

    With ``patch_size=1`` this is a simple reshape (no spatial merging).
    Future versions can merge P×P spatial patches to reduce sequence length.

    Args:
        patch_size: Side length of the spatial patch (1 = no merging).
    """

    def __init__(self, patch_size: int = 1) -> None:
        self.patch_size = patch_size

    def patchify(
        self,
        latent: torch.Tensor,
        *,
        latent_shape: VideoLatentShape,
        fps: float = 24.0,
        scale_factors: SpatioTemporalScaleFactors | None = None,
    ) -> LatentState:
        """Convert dense latent [B, C, T, H, W] → sequence [B, seq, C].

        Also produces position coordinates and an all-ones denoise mask.

        Returns:
            A :class:`LatentState` ready for the denoising loop.
        """
        sf = scale_factors or SpatioTemporalScaleFactors.default()
        b, c, t, h, w = latent.shape

        # [B, C, T, H, W] → [B, T*H*W, C]
        tokens = latent.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)
        positions = _make_positions(b, t, h, w, sf, fps, latent.device, latent.dtype)
        denoise_mask = torch.ones(b, t * h * w, 1, device=latent.device, dtype=latent.dtype)
        clean = tokens.clone()

        return LatentState(
            latent=tokens,
            denoise_mask=denoise_mask,
            positions=positions,
            clean_latent=clean,
        )

    def unpatchify(
        self,
        state: LatentState,
        *,
        latent_shape: VideoLatentShape,
    ) -> torch.Tensor:
        """Convert sequence [B, seq, C] → dense latent [B, C, T, H, W]."""
        t, h, w = latent_shape.num_frames, latent_shape.height, latent_shape.width
        # [B, T*H*W, C] → [B, C, T, H, W]
        b = state.latent.shape[0]
        c = state.latent.shape[-1]
        return state.latent.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)


class AudioPatchifier:
    """Flatten / unflatten audio latents for the transformer.

    Args:
        patch_size: Patch length along the time axis (1 = no merging).
    """

    def __init__(self, patch_size: int = 1) -> None:
        self.patch_size = patch_size

    def patchify(
        self,
        latent: torch.Tensor,
        *,
        latent_shape: AudioLatentShape,
    ) -> LatentState:
        """Convert [B, C, L] → [B, L, C] with positions and mask."""
        b, c, length = latent.shape
        # [B, C, L] → [B, L, C]
        tokens = latent.permute(0, 2, 1)
        positions = _make_audio_positions(b, length, latent.device, latent.dtype)
        denoise_mask = torch.ones(b, length, 1, device=latent.device, dtype=latent.dtype)
        return LatentState(
            latent=tokens,
            denoise_mask=denoise_mask,
            positions=positions,
            clean_latent=tokens.clone(),
        )

    def unpatchify(
        self,
        state: LatentState,
        *,
        latent_shape: AudioLatentShape,
    ) -> torch.Tensor:
        """Convert [B, L, C] → [B, C, L]."""
        # [B, L, C] → [B, C, L]
        return state.latent.permute(0, 2, 1)


# ─── Coordinate Utilities ────────────────────────────────────────────────────

def get_pixel_coords(
    num_frames: int,
    height: int,
    width: int,
    scale_factors: SpatioTemporalScaleFactors,
    fps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate normalised pixel-space coordinate grids.

    Returns:
        Tensor [1, 3, T*H*W, 2] with (start, end) coordinates per axis.
    """
    return _make_positions(1, num_frames, height, width, scale_factors, fps, device, dtype)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _make_positions(
    batch: int,
    t: int,
    h: int,
    w: int,
    sf: SpatioTemporalScaleFactors,
    fps: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build [B, 3, T*H*W, 2] position tensor with (start, end) per axis.

    Axis 0 = time, 1 = height, 2 = width.
    Coordinates are in pixel space (latent index × scale factor).
    """
    seq = t * h * w

    # Temporal positions
    t_idx = torch.arange(t, device=device, dtype=dtype)
    t_start = (t_idx * sf.temporal).repeat_interleave(h * w)
    t_end = t_start + sf.temporal

    # Height positions
    h_idx = torch.arange(h, device=device, dtype=dtype)
    h_start = (h_idx * sf.spatial_h).repeat(t).repeat_interleave(w)
    h_end = h_start + sf.spatial_h

    # Width positions
    w_idx = torch.arange(w, device=device, dtype=dtype)
    w_start = (w_idx * sf.spatial_w).repeat(t * h)
    w_end = w_start + sf.spatial_w

    # Stack: [3, seq, 2]
    positions = torch.stack(
        [
            torch.stack([t_start, t_end], dim=-1),
            torch.stack([h_start, h_end], dim=-1),
            torch.stack([w_start, w_end], dim=-1),
        ],
        dim=0,
    )
    return positions.unsqueeze(0).expand(batch, -1, -1, -1)


def _make_audio_positions(
    batch: int,
    length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build [B, 1, L, 2] position tensor for audio (time axis only)."""
    idx = torch.arange(length, device=device, dtype=dtype)
    positions = torch.stack([idx, idx + 1], dim=-1)  # [L, 2]
    return positions.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)
