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
    """

    def step(
        self,
        model_output: torch.Tensor,
        sigma_t: torch.Tensor | float,
        sigma_next: torch.Tensor | float,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one Euler step.

        Args:
            model_output: [B, ...] predicted velocity from the model.
            sigma_t: Current noise level.
            sigma_next: Next (lower) noise level.
            sample: [B, ...] current noisy sample.

        Returns:
            [B, ...] denoised sample at sigma_next.
        """
        dt = sigma_next - sigma_t  # negative (decreasing noise)
        return sample + dt * model_output
