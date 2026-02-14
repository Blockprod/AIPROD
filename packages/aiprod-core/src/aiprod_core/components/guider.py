# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
Classifier-Free Guidance

Standard CFG implementation with optional dynamic rescaling
to prevent oversaturation at high guidance scales.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassifierFreeGuider(nn.Module):
    """Classifier-Free Guidance with optional rescaling.

    Combines conditional and unconditional model predictions:
        output = uncond + scale * (cond - uncond)

    With dynamic rescaling (prevents oversaturation):
        output = rescale_factor * output_rescaled + (1 - rescale_factor) * output

    Args:
        guidance_scale: CFG scale factor (typical: 3.0-15.0).
        rescale_factor: Dynamic rescaling strength (0.0 = off, 0.7 = recommended).
    """

    def __init__(self, guidance_scale: float = 7.5, rescale_factor: float = 0.0):
        super().__init__()
        self.guidance_scale = guidance_scale
        self.rescale_factor = rescale_factor

    def guide(
        self,
        cond_output: torch.Tensor,
        uncond_output: torch.Tensor,
        guidance_scale: float | None = None,
    ) -> torch.Tensor:
        """Apply classifier-free guidance.

        Args:
            cond_output: [B, ...] conditional model output.
            uncond_output: [B, ...] unconditional model output.
            guidance_scale: Override default guidance scale.

        Returns:
            [B, ...] guided output.
        """
        scale = guidance_scale if guidance_scale is not None else self.guidance_scale

        # Standard CFG
        guided = uncond_output + scale * (cond_output - uncond_output)

        # Optional dynamic rescaling
        if self.rescale_factor > 0:
            # Compute per-channel std ratio
            std_cond = cond_output.std(dim=list(range(1, cond_output.dim())), keepdim=True)
            std_guided = guided.std(dim=list(range(1, guided.dim())), keepdim=True)

            rescaled = guided * (std_cond / (std_guided + 1e-8))
            guided = (
                self.rescale_factor * rescaled
                + (1 - self.rescale_factor) * guided
            )

        return guided
