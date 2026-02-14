# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Diffusion Protocols

Abstract protocol interfaces for diffusion pipeline components.
These protocols define the contracts between scheduler, guider,
noiser, and stepper components.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from aiprod_core.types import LatentState


@runtime_checkable
class DiffusionStepProtocol(Protocol):
    """Protocol for a single step of the diffusion process.

    Implementations advance the noisy sample one step closer to the
    clean prediction using the diffusion schedule (sigmas).
    """

    def step(
        self,
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        """Advance one diffusion step.

        Args:
            noisy: Current noisy latent [B, seq, C].
            denoised: Model's clean prediction [B, seq, C].
            sigmas: Full noise schedule (1-D tensor).
            step_index: Current position in the schedule.

        Returns:
            Updated latent for the next step.
        """
        ...


@runtime_checkable
class GuiderProtocol(Protocol):
    """Protocol for classifier-free guidance strategies."""

    def guide(
        self,
        cond_output: torch.Tensor,
        uncond_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply guidance to model outputs."""
        ...


@runtime_checkable
class NoiserProtocol(Protocol):
    """Protocol for adding noise to latent states."""

    def __call__(
        self,
        latent_state: LatentState,
        noise_scale: float,
    ) -> LatentState:
        """Add noise to a latent state, respecting denoise_mask."""
        ...


@runtime_checkable
class SchedulerProtocol(Protocol):
    """Protocol for diffusion noise schedules."""

    def get_sigmas(self, num_steps: int) -> torch.Tensor:
        """Return sigma values for each diffusion step."""
        ...
