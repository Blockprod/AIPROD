# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Type System

Core data types, shapes, and protocols used across all AIPROD modules.
Designed for multi-modal generation (video + audio + text).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol, TypeVar, Optional, Sequence
import torch


# ─── Enums ────────────────────────────────────────────────────────────────────

class ModalityType(Enum):
    """Supported generation modalities."""
    VIDEO = auto()
    AUDIO = auto()
    TEXT = auto()
    IMAGE = auto()


class PrecisionMode(Enum):
    """Model precision modes."""
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    FP8 = "float8_e4m3fn"


class DeviceType(Enum):
    """Target compute devices."""
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


# ─── Shape Definitions ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class VideoShape:
    """Describes pixel-space video dimensions.

    AIPROD uses a [B, C, T, H, W] convention throughout.
    """
    batch: int
    channels: int = 3
    frames: int = 49
    height: int = 512
    width: int = 768

    @property
    def num_pixels(self) -> int:
        return self.frames * self.height * self.width


@dataclass(frozen=True)
class LatentShape:
    """Describes latent-space tensor dimensions.

    AIPROD's HWVAE uses a configurable compression ratio defined by
    `spatial_factor` and `temporal_factor`.
    """
    batch: int
    channels: int = 64
    frames: int = 7
    height: int = 64
    width: int = 96

    @property
    def spatial_factor(self) -> int:
        """Spatial compression ratio (pixel / latent)."""
        return 8  # configurable per model variant

    @property
    def temporal_factor(self) -> int:
        """Temporal compression ratio (frames / latent_frames)."""
        return 7


@dataclass(frozen=True)
class AudioShape:
    """Describes audio tensor dimensions."""
    batch: int
    channels: int = 1
    samples: int = 48000  # 1 second at 48kHz
    sample_rate: int = 48000


# ─── State Containers ────────────────────────────────────────────────────────

@dataclass
class LatentState:
    """Container for latent tensors during diffusion.

    Holds the noisy latent, timestep info, and optional conditioning.
    """
    latent: torch.Tensor
    timestep: torch.Tensor
    noise_level: float = 1.0
    conditioning: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Top-level configuration for a generation request."""
    prompt: str
    negative_prompt: str = ""
    num_frames: int = 49
    height: int = 512
    width: int = 768
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    audio_enabled: bool = True
    precision: PrecisionMode = PrecisionMode.BF16
    device: DeviceType = DeviceType.CUDA


# ─── Protocols ────────────────────────────────────────────────────────────────

ModelT = TypeVar("ModelT")


class ModelConfigurator(Protocol[ModelT]):
    """Protocol for model factory/configurator classes."""

    def build_model(self, device: torch.device, dtype: torch.dtype) -> ModelT:
        """Construct the model on the given device with given dtype."""
        ...

    def load_weights(self, model: ModelT, checkpoint_path: str) -> ModelT:
        """Load trained weights into the model."""
        ...


class DiffusionStep(Protocol):
    """Protocol for a single diffusion denoising step."""

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one denoising step."""
        ...


class Scheduler(Protocol):
    """Protocol for noise schedule generators."""

    def get_schedule(self, num_steps: int) -> torch.Tensor:
        """Return a tensor of noise levels for each timestep."""
        ...

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to a clean sample at the given timestep."""
        ...


class Guider(Protocol):
    """Protocol for classifier-free guidance strategies."""

    def guide(
        self,
        cond_output: torch.Tensor,
        uncond_output: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Apply guidance to model outputs."""
        ...
