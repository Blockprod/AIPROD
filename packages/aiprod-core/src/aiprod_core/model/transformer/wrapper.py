# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
X0Model — Thin wrapper that predicts the clean latent x₀.

Wraps the raw SHDT backbone and converts its output into a
clean-sample prediction usable by the Euler flow step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .modality import Modality


class X0Model(nn.Module):
    """Wraps SHDT backbone to produce x₀ (clean sample) predictions.

    The underlying SHDT predicts velocity v such that:
        x₀ = x_t - σ · v

    This wrapper takes the raw output and applies the conversion.

    Args:
        backbone: The raw SHDT (or any diffusion transformer) model.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        video: Modality,
        audio: Modality | None = None,
        perturbation_config: object | None = None,
        perturbations: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict denoised latents from noisy inputs.

        Args:
            video: Video modality input.
            audio: Optional audio modality input.
            perturbation_config: Optional STG perturbation config.
            perturbations: Legacy alias for perturbation_config.

        Returns:
            Tuple of (denoised_video, denoised_audio) tensors.
            Audio may be None if no audio modality was provided.
        """
        # Legacy alias
        if perturbation_config is None and perturbations is not None:
            perturbation_config = perturbations
        # Build backbone inputs
        # The backbone (SHDT) expects (latent, timestep, context) at minimum
        sigma = video.sigma
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor([sigma], device=video.latent.device, dtype=video.latent.dtype)

        # Forward through backbone
        if hasattr(self.backbone, "forward_multimodal"):
            # Full multimodal forward
            video_out, audio_out = self.backbone.forward_multimodal(
                video_latent=video.latent,
                audio_latent=audio.latent if audio is not None else None,
                video_positions=video.positions,
                audio_positions=audio.positions if audio is not None else None,
                context=video.context,
                timestep=sigma,
            )
        else:
            # Single-modality fallback
            video_out = self.backbone(
                video.latent,
                sigma,
                video.context,
            )
            audio_out = None

        return video_out, audio_out
