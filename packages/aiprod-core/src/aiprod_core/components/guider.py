# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
Classifier-Free Guidance

Standard CFG implementation with optional dynamic rescaling
to prevent oversaturation at high guidance scales.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    def enabled(self) -> bool:
        """Return True if guidance is active (scale != 1.0)."""
        return self.guidance_scale != 1.0

    def delta(
        self,
        cond_output: torch.Tensor,
        uncond_output: torch.Tensor,
    ) -> torch.Tensor:
        """Return the guidance delta (correction term only).

        ``guided = cond + delta(cond, uncond)``
        """
        return (self.guidance_scale - 1.0) * (cond_output - uncond_output)


# ─── Aliases ──────────────────────────────────────────────────────────────────

# Short alias used throughout the pipelines
CFGGuider = ClassifierFreeGuider


class STGGuider(ClassifierFreeGuider):
    """Spatiotemporal Guidance guider.

    Extends CFG with an additional STG term that uses a perturbed model
    output as a third signal.

    Args:
        guidance_scale: CFG scale.
        stg_scale: STG scale.
        rescale_factor: Dynamic rescaling.
    """

    def __init__(
        self,
        guidance_scale: float = 7.5,
        stg_scale: float = 1.0,
        rescale_factor: float = 0.0,
    ):
        super().__init__(guidance_scale=guidance_scale, rescale_factor=rescale_factor)
        self.stg_scale = stg_scale

    def guide_stg(
        self,
        cond_output: torch.Tensor,
        uncond_output: torch.Tensor,
        perturbed_output: torch.Tensor,
    ) -> torch.Tensor:
        """Apply CFG + STG guidance.

        result = uncond + cfg_scale * (cond - uncond) + stg_scale * (cond - perturbed)
        """
        cfg = self.guidance_scale * (cond_output - uncond_output)
        stg = self.stg_scale * (cond_output - perturbed_output)
        return uncond_output + cfg + stg


@dataclass
class MultiModalGuiderParams:
    """Parameters for multi-modal (video + audio) guidance.

    Args:
        cfg_scale: Classifier-free guidance scale.
        stg_scale: Spatiotemporal guidance scale.
        rescale_scale: Dynamic rescaling factor.
        modality_scale: Cross-modal guidance scale.
        skip_step: Skip guidance for the first N steps.
        stg_blocks: Transformer block indices to perturb for STG.
    """
    cfg_scale: float = 3.0
    stg_scale: float = 1.0
    rescale_scale: float = 0.7
    modality_scale: float = 3.0
    skip_step: int = 0
    stg_blocks: list[int] | None = None


class MultiModalGuider(nn.Module):
    """Multi-modal guidance for joint video+audio generation.

    Supports two construction patterns:

    New API (dual-modal):
        ``MultiModalGuider(video_params=..., audio_params=...)``

    Legacy API (single-modal, used by pipeline helpers):
        ``MultiModalGuider(params=..., negative_context=...)``
        Stores ``params`` and ``negative_context`` for use by pipeline
        helper functions (``should_skip_step``, ``calculate``, etc.).
    """

    def __init__(
        self,
        video_params: MultiModalGuiderParams | None = None,
        audio_params: MultiModalGuiderParams | None = None,
        *,
        params: MultiModalGuiderParams | None = None,
        negative_context: torch.Tensor | None = None,
    ):
        super().__init__()

        # Legacy single-modal mode
        if params is not None:
            self.params = params
            self.negative_context = negative_context
            video_params = params

        if not hasattr(self, "params"):
            self.params = video_params or MultiModalGuiderParams()
            self.negative_context = negative_context

        self.video_params = video_params or MultiModalGuiderParams()
        self.audio_params = audio_params or MultiModalGuiderParams()
        self._video_guider = ClassifierFreeGuider(
            guidance_scale=self.video_params.cfg_scale,
            rescale_factor=self.video_params.rescale_scale,
        )
        self._audio_guider = ClassifierFreeGuider(
            guidance_scale=self.audio_params.cfg_scale,
            rescale_factor=self.audio_params.rescale_scale,
        )

    # ── Legacy single-modal API (pipeline helpers) ────────────────────────

    def should_skip_step(self, step_index: int) -> bool:
        """Return True if guidance should be skipped for this step."""
        return step_index < self.params.skip_step

    def do_unconditional_generation(self) -> bool:
        """Return True if CFG unconditional pass is needed."""
        return self.params.cfg_scale != 1.0

    def do_perturbed_generation(self) -> bool:
        """Return True if STG perturbed pass is needed."""
        return (
            self.params.stg_scale != 0.0
            and self.params.stg_blocks is not None
            and len(self.params.stg_blocks) > 0
        )

    def do_isolated_modality_generation(self) -> bool:
        """Return True if modality isolation pass is needed."""
        return self.params.modality_scale != 0.0

    def calculate(
        self,
        denoised: torch.Tensor,
        neg_denoised: torch.Tensor | float,
        ptb_denoised: torch.Tensor | float,
        mod_denoised: torch.Tensor | float,
    ) -> torch.Tensor:
        """Combine CFG, STG, and modality guidance signals.

        result = denoised
            + cfg_scale * (denoised - neg_denoised)
            + stg_scale * (denoised - ptb_denoised)
            + modality_scale * (denoised - mod_denoised)
        """
        result = denoised
        p = self.params
        if isinstance(neg_denoised, torch.Tensor):
            result = result + p.cfg_scale * (denoised - neg_denoised)
        if isinstance(ptb_denoised, torch.Tensor):
            result = result + p.stg_scale * (denoised - ptb_denoised)
        if isinstance(mod_denoised, torch.Tensor):
            result = result + p.modality_scale * (denoised - mod_denoised)
        return result

    # ── Dual-modal API ────────────────────────────────────────────────────

    def guide(
        self,
        video_cond: torch.Tensor,
        video_uncond: torch.Tensor,
        audio_cond: torch.Tensor | None = None,
        audio_uncond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply guidance to both modalities."""
        guided_video = self._video_guider.guide(video_cond, video_uncond)
        guided_audio = None
        if audio_cond is not None and audio_uncond is not None:
            guided_audio = self._audio_guider.guide(audio_cond, audio_uncond)
        return guided_video, guided_audio
