# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Noiser — Gaussian noise injection for flow-matching diffusion.

Applies noise to latent states following the flow-matching formulation:
    noisy = (1 - sigma) * clean + sigma * noise

Respects `denoise_mask` in LatentState so that conditioning regions
(mask == 0) remain clean while generation regions (mask == 1) are noised.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from aiprod_core.types import LatentState


class GaussianNoiser:
    """Add Gaussian noise to latent states for flow-matching diffusion.

    The noise is weighted by ``noise_scale`` and respects the ``denoise_mask``
    field of the :class:`LatentState`.  Where the mask is 0 the latent is
    kept clean; where it is 1 the full noise is applied.

    Args:
        generator: Optional torch generator for reproducible noise.
    """

    def __init__(self, generator: torch.Generator | None = None) -> None:
        self._generator = generator

    def __call__(
        self,
        latent_state: LatentState,
        noise_scale: float = 1.0,
    ) -> LatentState:
        """Add noise to the latent state.

        Args:
            latent_state: Input state containing ``latent`` and optionally
                ``denoise_mask`` and ``clean_latent``.
            noise_scale: Sigma value (0 = clean, 1 = pure noise).

        Returns:
            New LatentState with noisy ``latent``.
        """
        latent = latent_state.latent
        noise = torch.randn(
            latent.shape,
            device=latent.device,
            dtype=latent.dtype,
            generator=self._generator,
        )

        # flow-matching interpolation: x_t = (1 - σ) * x_0 + σ * ε
        clean = latent_state.clean_latent if latent_state.clean_latent is not None else latent
        noisy = (1.0 - noise_scale) * clean + noise_scale * noise

        # Apply denoise mask: only noise regions marked for denoising
        if latent_state.denoise_mask is not None:
            mask = latent_state.denoise_mask  # [B, seq, 1]
            noisy = mask * noisy + (1.0 - mask) * clean

        return replace(latent_state, latent=noisy, noise_level=noise_scale)
