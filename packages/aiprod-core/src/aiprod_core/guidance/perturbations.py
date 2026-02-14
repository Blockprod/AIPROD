# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Perturbation-Based Guidance

Implements Spatiotemporal Guidance (STG) and related perturbation strategies
for improving generation quality during the denoising loop.

STG works by perturbing specific transformer blocks (e.g. zeroing out
attention) and using the difference between the perturbed and unperturbed
outputs as an additional guidance signal.

Reference: https://arxiv.org/abs/2411.18664
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Sequence

import torch


class PerturbationType(Enum):
    """Types of perturbation that can be applied to transformer blocks."""
    IDENTITY = auto()          # Replace attention output with identity (zero attention)
    ATTENTION_ZERO = auto()    # Zero out attention weights
    ATTENTION_NOISE = auto()   # Add noise to attention weights
    # Legacy / cross-modal perturbation types (pipeline compat)
    SKIP_VIDEO_SELF_ATTN = auto()   # Skip video self-attention in a block
    SKIP_AUDIO_SELF_ATTN = auto()   # Skip audio self-attention in a block
    SKIP_A2V_CROSS_ATTN = auto()    # Skip audio-to-video cross-attention
    SKIP_V2A_CROSS_ATTN = auto()    # Skip video-to-audio cross-attention


@dataclass
class PerturbationConfig:
    """Configuration for a single perturbation to apply.

    Can be constructed in two ways:

    New API (per-block):
        ``PerturbationConfig(block_index=3, perturbation_type=PerturbationType.ATTENTION_ZERO)``

    Legacy API (multi-block wrapper):
        ``PerturbationConfig(perturbations=[perturbation_obj_1, ...])``

    Args:
        block_index: Which transformer block to perturb (-1 = all).
        perturbation_type: How to perturb the block.
        scale: Strength of the perturbation.
        perturbations: Legacy — list of ``Perturbation`` data objects.
    """
    block_index: int = 0
    perturbation_type: PerturbationType = PerturbationType.IDENTITY
    scale: float = 1.0
    # Legacy field: when constructing via old API, Perturbation objects go here
    perturbations: list | None = None


@dataclass
class BatchedPerturbationConfig:
    """A batch of perturbation configs for multi-block STG.

    Enables perturbing multiple blocks in a single forward pass.

    Can be constructed with either:
        ``BatchedPerturbationConfig(configs=[...])``    — new API
        ``BatchedPerturbationConfig(perturbations=[...])``  — legacy (mapped to configs)
    """
    configs: list[PerturbationConfig] = field(default_factory=list)

    def __init__(
        self,
        configs: list[PerturbationConfig] | None = None,
        perturbations: list | None = None,
    ):
        if configs is not None:
            self.configs = configs
        elif perturbations is not None:
            # Legacy: perturbations is a list of PerturbationConfig objects
            self.configs = perturbations if perturbations else []
        else:
            self.configs = []

    @classmethod
    def from_blocks(
        cls,
        block_indices: Sequence[int],
        perturbation_type: PerturbationType = PerturbationType.IDENTITY,
        scale: float = 1.0,
    ) -> BatchedPerturbationConfig:
        """Create configs for multiple blocks with the same settings."""
        return cls(
            configs=[
                PerturbationConfig(
                    block_index=idx,
                    perturbation_type=perturbation_type,
                    scale=scale,
                )
                for idx in block_indices
            ]
        )

    def __bool__(self) -> bool:
        return len(self.configs) > 0


class Perturbation:
    """Applies perturbation-based guidance during inference.

    During the forward pass, selected transformer blocks are perturbed
    (e.g. attention set to identity).  The difference between the
    unperturbed and perturbed outputs provides an additional guidance
    signal scaled by ``stg_scale``.

    Supports two construction patterns:

    New API:
        ``Perturbation(config=BatchedPerturbationConfig(...), stg_scale=1.0)``

    Legacy API:
        ``Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=[1,2])``
    """

    def __init__(
        self,
        config: BatchedPerturbationConfig | None = None,
        stg_scale: float = 1.0,
        *,
        type: PerturbationType | None = None,
        blocks: list[int] | None = None,
    ) -> None:
        self.stg_scale = stg_scale
        # Legacy constructor
        self.type = type
        self.blocks = blocks

        if config is not None:
            self.config = config
        elif type is not None:
            # Build config from legacy args
            if blocks:
                self.config = BatchedPerturbationConfig(
                    configs=[PerturbationConfig(block_index=b, perturbation_type=type) for b in blocks]
                )
            else:
                # blocks=None means apply to all (use sentinel -1)
                self.config = BatchedPerturbationConfig(
                    configs=[PerturbationConfig(block_index=-1, perturbation_type=type)]
                )
        else:
            self.config = BatchedPerturbationConfig()

    def should_perturb(self, block_index: int) -> bool:
        """Check if a given block should be perturbed."""
        return any(c.block_index == block_index for c in self.config.configs)

    def get_perturbation(self, block_index: int) -> PerturbationConfig | None:
        """Get perturbation config for a specific block."""
        for c in self.config.configs:
            if c.block_index == block_index:
                return c
        return None

    def apply_guidance(
        self,
        output_cond: torch.Tensor,
        output_uncond: torch.Tensor,
        output_perturbed: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Combine CFG and STG guidance signals.

        result = uncond + cfg_scale * (cond - uncond) + stg_scale * (cond - perturbed)

        Args:
            output_cond: Conditional model output.
            output_uncond: Unconditional model output.
            output_perturbed: Perturbed (STG) model output.
            cfg_scale: CFG guidance scale.

        Returns:
            Guided output tensor.
        """
        cfg_component = cfg_scale * (output_cond - output_uncond)
        stg_component = self.stg_scale * (output_cond - output_perturbed)
        return output_uncond + cfg_component + stg_component
