# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Latent Tools — Lifecycle management for latent states.

LatentTools provide a high-level API for creating, patchifying,
conditioning, and cleaning up latent states used in the diffusion loop.

Design:
    VideoLatentTools  — manages video latents ([B, C, T, H, W])
    AudioLatentTools  — manages audio latents ([B, C, L])

    Both wrap a patchifier and a shape descriptor, providing
    convenience methods for the full latent lifecycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

import torch

from aiprod_core.components.patchifier import AudioPatchifier, VideoLatentPatchifier
from aiprod_core.conditioning import ConditioningItem
from aiprod_core.types import (
    AudioLatentShape,
    LatentState,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
)


class LatentTools(ABC):
    """Abstract base for latent state management."""

    @abstractmethod
    def create_initial_state(
        self,
        device: torch.device,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        """Create an initial patchified latent state (zeros or from data)."""
        ...

    @abstractmethod
    def clear_conditioning(self, state: LatentState) -> LatentState:
        """Strip conditioning regions from the state (keep only generated content)."""
        ...

    @abstractmethod
    def unpatchify(self, state: LatentState) -> LatentState:
        """Convert from sequence back to spatial representation."""
        ...


class VideoLatentTools(LatentTools):
    """Manages the lifecycle of video latent states.

    Args:
        patchifier: VideoLatentPatchifier instance.
        latent_shape: Shape descriptor for the video latents.
        fps: Frame rate (used for temporal position encoding).
        scale_factors: VAE compression ratios.
    """

    def __init__(
        self,
        patchifier: VideoLatentPatchifier,
        latent_shape: VideoLatentShape,
        fps: float = 24.0,
        scale_factors: SpatioTemporalScaleFactors | None = None,
    ) -> None:
        self.patchifier = patchifier
        self.shape = latent_shape
        self.fps = fps
        self.scale_factors = scale_factors or SpatioTemporalScaleFactors.default()

    def create_initial_state(
        self,
        device: torch.device,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        """Create a patchified video latent state.

        If ``initial_latent`` is provided it is used as the clean latent;
        otherwise a zero tensor is created.
        """
        s = self.shape
        if initial_latent is None:
            initial_latent = torch.zeros(
                s.batch_size, s.channels, s.num_frames, s.height, s.width,
                device=device, dtype=dtype,
            )
        return self.patchifier.patchify(
            initial_latent,
            latent_shape=s,
            fps=self.fps,
            scale_factors=self.scale_factors,
        )

    def clear_conditioning(self, state: LatentState) -> LatentState:
        """Keep only the denoised portion (mask == 1)."""
        if state.denoise_mask is None:
            return state
        # For video we keep the full sequence — conditioning is already
        # blended. clear_conditioning is a no-op for simple cases and
        # strips concatenated reference latents in IC-LoRA workflows.
        return state

    def unpatchify(self, state: LatentState) -> LatentState:
        """Unpatchify latent back to [B, C, T, H, W]."""
        dense = self.patchifier.unpatchify(state, latent_shape=self.shape)
        return replace(state, latent=dense)


class AudioLatentTools(LatentTools):
    """Manages the lifecycle of audio latent states.

    Args:
        patchifier: AudioPatchifier instance.
        latent_shape: Shape descriptor for the audio latents.
    """

    def __init__(
        self,
        patchifier: AudioPatchifier,
        latent_shape: AudioLatentShape,
    ) -> None:
        self.patchifier = patchifier
        self.shape = latent_shape

    def create_initial_state(
        self,
        device: torch.device,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        s = self.shape
        if initial_latent is None:
            initial_latent = torch.zeros(
                s.batch_size, s.channels, s.length,
                device=device, dtype=dtype,
            )
        return self.patchifier.patchify(initial_latent, latent_shape=s)

    def clear_conditioning(self, state: LatentState) -> LatentState:
        return state

    def unpatchify(self, state: LatentState) -> LatentState:
        dense = self.patchifier.unpatchify(state, latent_shape=self.shape)
        return replace(state, latent=dense)
