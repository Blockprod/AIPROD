# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Type System

Core data types, shapes, and protocols used across all AIPROD modules.
Designed for multi-modal generation (video + audio + text).
"""

from __future__ import annotations

import math
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


# ─── Scale Factors ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SpatioTemporalScaleFactors:
    """Compression ratios between pixel space and latent space.

    These factors define how much the VAE compresses along each axis.
    """
    temporal: int = 7
    spatial_h: int = 8
    spatial_w: int = 8

    @classmethod
    def default(cls) -> SpatioTemporalScaleFactors:
        """Return default AIPROD HWVAE scale factors."""
        return cls(temporal=7, spatial_h=8, spatial_w=8)


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
class VideoPixelShape:
    """Describes output video dimensions including frame rate.

    Used as the source-of-truth for generation target resolution.
    """
    height: int = 512
    width: int = 768
    num_frames: int = 49
    fps: float = 24.0

    def __init__(
        self,
        height: int = 512,
        width: int = 768,
        num_frames: int = 49,
        fps: float = 24.0,
        *,
        batch: int | None = None,
        frames: int | None = None,
    ):
        # Accept legacy kwargs: batch (ignored), frames → num_frames
        if frames is not None and num_frames == 49:
            num_frames = frames
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "num_frames", num_frames)
        object.__setattr__(self, "fps", fps)

    @property
    def frames(self) -> int:
        """Legacy alias for num_frames."""
        return self.num_frames

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @property
    def duration_seconds(self) -> float:
        return self.num_frames / self.fps


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
        return 8

    @property
    def temporal_factor(self) -> int:
        """Temporal compression ratio (frames / latent_frames)."""
        return 7


@dataclass(frozen=True)
class VideoLatentShape:
    """Shape of video latents after VAE encoding + patchification.

    Computed from a VideoPixelShape and the VAE scale factors.
    """
    batch_size: int = 1
    channels: int = 64
    num_frames: int = 7
    height: int = 64
    width: int = 96

    @classmethod
    def from_pixel_shape(
        cls,
        shape: VideoPixelShape,
        latent_channels: int = 64,
        scale_factors: SpatioTemporalScaleFactors | None = None,
        batch_size: int = 1,
    ) -> VideoLatentShape:
        sf = scale_factors or SpatioTemporalScaleFactors.default()
        return cls(
            batch_size=batch_size,
            channels=latent_channels,
            num_frames=max(1, math.ceil(shape.num_frames / sf.temporal)),
            height=shape.height // sf.spatial_h,
            width=shape.width // sf.spatial_w,
        )

    @property
    def seq_len(self) -> int:
        """Total number of latent tokens (after flattening spatial-temporal)."""
        return self.num_frames * self.height * self.width


@dataclass(frozen=True)
class AudioLatentShape:
    """Shape of audio latents after audio codec encoding + patchification."""
    batch_size: int = 1
    channels: int = 64
    length: int = 50  # number of latent frames

    @classmethod
    def from_video_pixel_shape(
        cls,
        shape: VideoPixelShape,
        latent_channels: int = 64,
        audio_sample_rate: int = 24000,
        audio_compression: int = 480,
        batch_size: int = 1,
    ) -> AudioLatentShape:
        duration = shape.num_frames / shape.fps
        total_audio_samples = int(duration * audio_sample_rate)
        latent_len = max(1, total_audio_samples // audio_compression)
        return cls(
            batch_size=batch_size,
            channels=latent_channels,
            length=latent_len,
        )

    @property
    def seq_len(self) -> int:
        return self.length


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

    Holds the noisy latent, conditioning information, and denoising metadata.
    Designed for use with both video and audio modalities.
    """
    latent: torch.Tensor                        # [B, seq_len, C] patchified latent
    denoise_mask: torch.Tensor | None = None    # [B, seq_len, 1] — 1.0 = denoise, 0.0 = keep clean
    positions: torch.Tensor | None = None       # [B, 3, seq_len, 2] — (t, h, w) start/end coords
    clean_latent: torch.Tensor | None = None    # [B, seq_len, C] — reference clean latent (for conditioning)
    timestep: torch.Tensor | None = None        # scalar or [B] timestep
    noise_level: float = 1.0
    conditioning: Optional[torch.Tensor] = None
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
