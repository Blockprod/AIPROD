# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
Adaptive Flow Scheduler

AIPROD's noise schedule for flow-matching diffusion.
Uses a configurable sigma schedule with optional learned
adjustments per training step.

Supports multiple schedule types:
    - linear: linearly spaced noise levels
    - cosine: cosine annealing schedule
    - sigmoid: sigmoid-shaped schedule (concentrates steps around mid-noise)
    - adaptive: learned schedule (requires training)
"""

from __future__ import annotations

from typing import Literal
import math

import torch
import torch.nn as nn


class AdaptiveFlowScheduler(nn.Module):
    """Flow-matching noise scheduler.

    In flow-matching, we define a path from noise (t=1) to data (t=0).
    The scheduler generates timestep sequences and provides methods
    to add noise and compute velocity targets.

    Args:
        schedule_type: Type of noise schedule.
        shift: Shift parameter for the sigmoid schedule.
    """

    def __init__(
        self,
        schedule_type: Literal["linear", "cosine", "sigmoid"] = "sigmoid",
        shift: float = 3.0,
    ):
        super().__init__()
        self.schedule_type = schedule_type
        self.shift = shift

    def get_schedule(self, num_steps: int, device: torch.device | None = None) -> torch.Tensor:
        """Generate a noise schedule.

        Args:
            num_steps: Number of denoising steps.
            device: Target device.

        Returns:
            [num_steps+1] tensor of noise levels from 1.0 (pure noise) to 0.0 (clean).
        """
        t = torch.linspace(0, 1, num_steps + 1, device=device)

        if self.schedule_type == "linear":
            sigmas = 1.0 - t
        elif self.schedule_type == "cosine":
            sigmas = torch.cos(t * math.pi / 2)
        elif self.schedule_type == "sigmoid":
            # Sigmoid schedule: concentrates steps in the middle
            sigmas = torch.sigmoid(self.shift * (0.5 - t))
            sigmas = (sigmas - sigmas[-1]) / (sigmas[0] - sigmas[-1])
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return sigmas

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to a clean sample at timestep t.

        Uses the flow-matching interpolation:
            x_t = (1 - t) * x0 + t * noise

        Args:
            x0: [B, ...] clean sample.
            noise: [B, ...] Gaussian noise.
            t: [B] or scalar timestep in [0, 1].

        Returns:
            [B, ...] noisy sample.
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        # Broadcast t to match x0 shape
        while t.dim() < x0.dim():
            t = t.unsqueeze(-1)

        return (1 - t) * x0 + t * noise

    def get_velocity(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the velocity target for flow-matching.

        velocity = noise - x0

        Args:
            x0: [B, ...] clean sample.
            noise: [B, ...] noise.

        Returns:
            [B, ...] velocity target.
        """
        return noise - x0
