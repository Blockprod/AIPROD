# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
Euler Flow Step

Performs a single Euler denoising step in flow-matching space.
"""

from __future__ import annotations

import torch


class EulerFlowStep:
    """Single Euler integration step for flow-matching diffusion.

    Updates the sample from noise level σ_t to σ_{t-1}:
        x_{t-1} = x_t + (σ_{t-1} - σ_t) * model_output

    Where model_output is the predicted velocity (dx/dt).

    Supports two calling conventions:
        1. step(model_output, sigma_t, sigma_next, sample)  — explicit sigmas
        2. step(noisy, denoised, sigmas, step_index)        — schedule-based
    """

    def step(
        self,
        noisy_or_output: torch.Tensor,
        denoised_or_sigma_t: torch.Tensor | float,
        sigmas_or_sigma_next: torch.Tensor | float,
        sample_or_step_index: torch.Tensor | int,
    ) -> torch.Tensor:
        """Perform one Euler step.

        Convention 1 (explicit):
            step(model_output, sigma_t, sigma_next, sample) → updated sample

        Convention 2 (schedule-based, used by pipelines):
            step(noisy, denoised, sigmas, step_index) → updated sample

        Returns:
            [B, ...] updated sample at the next noise level.
        """
        if isinstance(sample_or_step_index, int):
            # Convention 2: schedule-based
            noisy = noisy_or_output
            denoised = denoised_or_sigma_t
            sigmas = sigmas_or_sigma_next
            step_idx = sample_or_step_index

            sigma_t = sigmas[step_idx]
            sigma_next = sigmas[step_idx + 1]

            # Compute velocity: v = (noisy - denoised) / sigma_t
            if sigma_t > 0:
                velocity = (noisy - denoised) / sigma_t
            else:
                velocity = torch.zeros_like(noisy)

            dt = sigma_next - sigma_t
            return noisy + dt * velocity
        else:
            # Convention 1: explicit
            model_output = noisy_or_output
            sigma_t = denoised_or_sigma_t
            sigma_next = sigmas_or_sigma_next
            sample = sample_or_step_index

            dt = sigma_next - sigma_t
            return sample + dt * model_output
