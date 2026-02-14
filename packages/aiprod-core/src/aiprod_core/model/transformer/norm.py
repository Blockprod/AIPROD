# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
Adaptive RMS Normalization

Normalization conditioned on timestep embedding, enabling the model
to modulate its behavior based on the noise level. Uses RMSNorm
(more efficient than LayerNorm) with learned scale and shift from
the timestep embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveRMSNorm(nn.Module):
    """RMSNorm with adaptive scale and shift from conditioning signal.

    Unlike standard LayerNorm:
        - No mean subtraction (RMSNorm)
        - Scale and shift are predicted from timestep embedding
        - More parameter-efficient than AdaLN (single projection)

    Args:
        dim: Feature dimension to normalize.
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.dim = dim

        # Learnable base scale
        self.weight = nn.Parameter(torch.ones(dim))

        # Adaptive modulation from conditioning
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim),  # predict scale and shift
        )

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] input tensor.
            conditioning: [B, D] conditioning signal (e.g., timestep embedding).

        Returns:
            [B, N, D] normalized and modulated tensor.
        """
        x_norm = self._rms_norm(x)

        if conditioning is not None:
            # Predict adaptive scale and shift
            mod = self.modulation(conditioning)  # [B, 2*D]
            scale, shift = mod.chunk(2, dim=-1)  # each [B, D]
            x_norm = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return x_norm
