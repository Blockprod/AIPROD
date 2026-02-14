# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Conditioning System

Defines conditioning items that modify latent states during generation.
Conditioning can inject reference frames, guide specific latent positions,
or apply keyframe-based control.

Design:
    Every conditioning item implements ``apply(state, tools) → state``.
    The pipeline iterates through a list of conditioning items and
    folds them into the initial latent state before the denoising loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import torch

from aiprod_core.types import LatentState


@dataclass
class ConditioningItem(ABC):
    """Base class for all conditioning items.

    Subclasses define how a piece of conditioning information
    (e.g. an encoded reference image) is merged into a LatentState.
    """

    strength: float = 1.0

    @abstractmethod
    def apply(self, latent: torch.Tensor, denoise_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply this conditioning to the latent and mask.

        Args:
            latent: Current latent tensor [B, seq, C].
            denoise_mask: Current denoise mask [B, seq, 1].

        Returns:
            Tuple of (modified_latent, modified_mask).
        """
        ...

    def apply_to(self, latent_state: LatentState, latent_tools=None) -> LatentState:
        """Apply this conditioning to a LatentState (pipeline API).

        Delegates to :meth:`apply` and returns an updated state.

        Args:
            latent_state: Current state container.
            latent_tools: Optional tools (currently unused, accepted for API compat).
        """
        mask = latent_state.denoise_mask
        if mask is None:
            mask = torch.ones(
                latent_state.latent.shape[0],
                latent_state.latent.shape[1],
                1,
                device=latent_state.latent.device,
                dtype=latent_state.latent.dtype,
            )
        new_latent, new_mask = self.apply(latent_state.latent, mask)
        return replace(latent_state, latent=new_latent, denoise_mask=new_mask)


@dataclass
class VideoConditionByLatentIndex(ConditioningItem):
    """Replace a specific latent position with an encoded image.

    Used for image-to-video: the first (or any) frame is replaced
    by the VAE-encoded image latent and its denoise mask is set to 0
    so it remains clean throughout the denoising loop.

    Args:
        latent: Encoded image latent [B, 1, C] or [B, spatial, C].
        latent_idx: Frame index in the latent sequence to replace.
        strength: Conditioning strength (1.0 = fully replace).
    """

    latent: torch.Tensor | None = None  # Will be set after encoding
    latent_idx: int = 0

    def apply(self, latent: torch.Tensor, denoise_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.latent is None:
            return latent, denoise_mask

        cond = self.latent.to(latent.device, latent.dtype)
        # Replace at the specified index(es)
        cond_len = cond.shape[1]
        start = self.latent_idx * (latent.shape[1] // max(1, denoise_mask.shape[1]))

        latent = latent.clone()
        denoise_mask = denoise_mask.clone()

        # Blend with strength
        latent[:, start : start + cond_len] = (
            self.strength * cond + (1 - self.strength) * latent[:, start : start + cond_len]
        )
        denoise_mask[:, start : start + cond_len] = 1.0 - self.strength

        return latent, denoise_mask


@dataclass
class VideoConditionByKeyframeIndex(ConditioningItem):
    """Add guiding keyframe latent at a specific frame index.

    Unlike LatentIndex conditioning, this does NOT replace the latent —
    instead it adds a soft guidance signal that the denoiser can use.

    Args:
        keyframes: Encoded keyframe latent [B, spatial, C].
        frame_idx: Target frame index.
        strength: Guidance strength.
    """

    keyframes: torch.Tensor | None = None
    frame_idx: int = 0

    def apply(self, latent: torch.Tensor, denoise_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keyframes is None:
            return latent, denoise_mask

        kf = self.keyframes.to(latent.device, latent.dtype)
        kf_len = kf.shape[1]
        start = self.frame_idx * kf_len

        latent = latent.clone()
        # Soft blend — don't zero out denoise mask (still denoise, but guided)
        latent[:, start : start + kf_len] = (
            self.strength * kf + (1 - self.strength) * latent[:, start : start + kf_len]
        )
        return latent, denoise_mask


@dataclass
class VideoConditionByReferenceLatent(ConditioningItem):
    """Condition the generation with a full reference video latent.

    Used by IC-LoRA pipelines: a reference video (e.g. depth map, pose)
    is encoded and concatenated/blended into the latent stream so that
    the denoiser can use it as spatial guidance.

    Args:
        latent: Encoded reference video latent [B, seq, C].
        downscale_factor: Ratio between target and reference resolution.
        strength: Conditioning strength.
    """

    latent: torch.Tensor | None = None
    downscale_factor: int = 1

    def apply(self, latent: torch.Tensor, denoise_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.latent is None:
            return latent, denoise_mask

        ref = self.latent.to(latent.device, latent.dtype)

        # If the reference is shorter, pad; if longer, truncate
        ref_len = ref.shape[1]
        tgt_len = latent.shape[1]

        latent = latent.clone()
        denoise_mask = denoise_mask.clone()

        use_len = min(ref_len, tgt_len)
        # Blend reference into latent and reduce denoise mask
        latent[:, :use_len] = (
            self.strength * ref[:, :use_len] + (1 - self.strength) * latent[:, :use_len]
        )
        denoise_mask[:, :use_len] = denoise_mask[:, :use_len] * (1 - self.strength)

        return latent, denoise_mask
