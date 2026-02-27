#!/usr/bin/env python3
# Copyright (c) 2025-2026 AIPROD. All rights reserved.
"""
validate_pipeline_e2e.py — End-to-End Pipeline Validation

Uses tiny randomly-initialized models to prove the complete AIPROD video
generation code path works end-to-end on any hardware (CPU or GPU):

    prompt → text_encode → noise → denoise_loop → decode_video → .mp4

Output: output/validation_test.mp4

Models used: MOCK (random weights, ~10MB total). This validates the CODE PATH,
not model quality. Once real models fit in VRAM, swap mock → real and
the same pipeline produces actual video.

Usage:
    python scripts/validate_pipeline_e2e.py
    python scripts/validate_pipeline_e2e.py --device cpu
    python scripts/validate_pipeline_e2e.py --steps 8 --frames 17
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK MODELS — Tiny random models that match the real pipeline interfaces
# ═══════════════════════════════════════════════════════════════════════════════


class MockSHDTBackbone(nn.Module):
    """Minimal transformer backbone compatible with X0Model wrapper.

    Accepts patchified multimodal inputs via ``forward_multimodal``
    and returns identity-scaled outputs. This mimics the shape contract
    of the real 19B SHDT transformer without doing any real denoising.
    """

    def forward_multimodal(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor | None,
        video_positions: torch.Tensor,
        audio_positions: torch.Tensor | None,
        context: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return slightly perturbed input (mock denoising)."""
        video_out = video_latent * 0.98 + 0.02 * torch.randn_like(video_latent)
        audio_out = None
        if audio_latent is not None:
            audio_out = audio_latent * 0.98 + 0.02 * torch.randn_like(audio_latent)
        return video_out, audio_out


class MockVideoDecoder(nn.Module):
    """Decodes latent [B, C, T', H', W'] → pixel video [B, 3, T, H, W].

    Uses a single 1×1×1 conv + trilinear upsampling to expand latent
    to pixel space. Output in [-1, 1] (tanh).
    """

    def __init__(
        self,
        latent_channels: int = 128,
        temporal_scale: int = 7,
        spatial_scale: int = 8,
    ):
        super().__init__()
        self.ts = temporal_scale
        self.ss = spatial_scale
        self.proj = nn.Conv3d(latent_channels, 3, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.proj(z)  # [B, 3, T', H', W']
        x = torch.nn.functional.interpolate(
            x,
            scale_factor=(self.ts, self.ss, self.ss),
            mode="trilinear",
            align_corners=False,
        )
        return torch.tanh(x)


class MockVideoEncoder(nn.Module):
    """Encodes pixel video [B, 3, T, H, W] → latent [B, C, T', H', W']."""

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.proj = nn.Conv3d(3, latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MockAudioDecoder(nn.Module):
    """Decodes audio latent [B, C, L] → waveform [B, 1, L*compression]."""

    def __init__(self, latent_channels: int = 64, compression: int = 480):
        super().__init__()
        self.compression = compression
        self.proj = nn.Linear(latent_channels, compression)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, c, length = z.shape
        x = z.permute(0, 2, 1)  # [B, L, C]
        x = self.proj(x)  # [B, L, compression]
        x = x.reshape(b, 1, length * self.compression)
        return torch.tanh(x)


class MockTextEncoder(nn.Module):
    """Mock text encoder that produces random embeddings.

    Returns the same signature as LLMBridge._encode_prompt():
        (video_context, audio_context, attention_mask)
    """

    def __init__(self, output_dim: int = 128, seq_len: int = 16):
        super().__init__()
        self.dim = output_dim
        self.seq_len = seq_len
        # Need at least one parameter so .to(device) works
        self._dummy = nn.Linear(1, 1)

    def _encode_prompt(
        self, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self._dummy.weight.device
        dtype = self._dummy.weight.dtype
        embeddings = torch.randn(
            1, self.seq_len, self.dim, device=device, dtype=dtype
        )
        mask = torch.ones(1, self.seq_len, device=device, dtype=dtype)
        return embeddings, embeddings.clone(), mask


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def latent_to_pixel_video(raw: torch.Tensor) -> torch.Tensor:
    """Convert decoder output [B, 3, T, H, W] float → [T, H, W, 3] uint8."""
    video = raw[0]  # [3, T, H, W]
    video = video.permute(1, 2, 3, 0)  # [T, H, W, 3]
    video = ((video.float() + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
    return video


@torch.inference_mode()
def run_validation(
    device: torch.device,
    height: int = 64,
    width: int = 64,
    num_frames: int = 9,
    fps: float = 8.0,
    num_steps: int = 4,
    seed: int = 42,
    output_path: str = "output/validation_test.mp4",
) -> None:
    t0 = time.time()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # ── Imports ───────────────────────────────────────────────────────────
    from aiprod_core.components.diffusion_steps import EulerDiffusionStep
    from aiprod_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
    from aiprod_core.components.noisers import GaussianNoiser
    from aiprod_core.components.schedulers import AIPROD2Scheduler
    from aiprod_core.model.transformer import X0Model
    from aiprod_core.types import VideoPixelShape
    from aiprod_pipelines.utils.helpers import (
        denoise_audio_video,
        euler_denoising_loop,
        multi_modal_guider_denoising_func,
    )
    from aiprod_pipelines.utils.media_io import encode_video
    from aiprod_pipelines.utils.types import PipelineComponents

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Create Mock Models ────────────────────────────────────────
    log.info(f"[1/6] Creating mock models on {device} ({dtype})")

    transformer = X0Model(MockSHDTBackbone()).to(device=device, dtype=dtype).eval()
    video_decoder = MockVideoDecoder(latent_channels=128).to(device=device, dtype=dtype).eval()
    text_encoder = MockTextEncoder(output_dim=128, seq_len=16).to(device=device, dtype=dtype)

    components = PipelineComponents(dtype=dtype, device=device)

    # ── Step 2: Text Encoding ─────────────────────────────────────────────
    log.info("[2/6] Encoding text prompt (mock)")

    prompt = "A drone flies slowly over a mountain lake at golden hour"
    v_ctx_p, a_ctx_p, _ = text_encoder._encode_prompt(prompt)
    v_ctx_n, a_ctx_n, _ = text_encoder._encode_prompt("")

    # ── Step 3: Diffusion Schedule ────────────────────────────────────────
    log.info(f"[3/6] Building diffusion schedule ({num_steps} steps)")

    generator = torch.Generator(device=device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()

    # Note: AIPROD2Scheduler.execute() is a known bug in pipeline code.
    # The correct method is get_schedule().
    sigmas = (
        AIPROD2Scheduler()
        .get_schedule(num_steps=num_steps, device=device)
        .to(dtype=torch.float32)
    )
    log.info(f"   Sigmas: {sigmas.tolist()}")

    # Minimal guidance — CFG only, no STG, no cross-modal
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=0.0,
        rescale_scale=0.7,
        modality_scale=0.0,
        skip_step=0,
        stg_blocks=[],
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=3.0,
        stg_scale=0.0,
        rescale_scale=0.7,
        modality_scale=0.0,
        skip_step=0,
        stg_blocks=[],
    )

    def denoising_loop(sigmas, video_state, audio_state, stepper):
        return euler_denoising_loop(
            sigmas=sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=stepper,
            denoise_fn=multi_modal_guider_denoising_func(
                video_guider=MultiModalGuider(
                    params=video_guider_params, negative_context=v_ctx_n
                ),
                audio_guider=MultiModalGuider(
                    params=audio_guider_params, negative_context=a_ctx_n
                ),
                v_context=v_ctx_p,
                a_context=a_ctx_p,
                transformer=transformer,
            ),
        )

    # ── Step 4: Denoising Loop ────────────────────────────────────────────
    log.info(
        f"[4/6] Running denoising loop ({height}x{width}, {num_frames}f, {num_steps} steps)"
    )

    output_shape = VideoPixelShape(
        batch=1, frames=num_frames, width=width, height=height, fps=fps
    )

    video_state, audio_state = denoise_audio_video(
        output_shape=output_shape,
        conditionings=[],
        noiser=noiser,
        sigmas=sigmas,
        stepper=stepper,
        denoising_loop_fn=denoising_loop,
        components=components,
        dtype=dtype,
        device=device,
    )

    log.info(
        f"   Video latent: {video_state.latent.shape}, "
        f"Audio latent: {audio_state.latent.shape}"
    )

    # ── Step 5: Decode Video ──────────────────────────────────────────────
    log.info("[5/6] Decoding latent → pixel video")

    raw_video = video_decoder(video_state.latent)  # [B, 3, T, H, W]
    log.info(f"   Raw decoded video: {raw_video.shape}")

    pixel_video = latent_to_pixel_video(raw_video)  # [T, H, W, 3] uint8
    log.info(f"   Pixel video: {pixel_video.shape} ({pixel_video.dtype})")

    # ── Step 6: Encode MP4 ────────────────────────────────────────────────
    log.info(f"[6/6] Writing MP4 → {output_path}")

    encode_video(
        video=pixel_video,
        fps=int(fps),
        audio=None,  # Video-only for validation
        audio_sample_rate=None,
        output_path=output_path,
        video_chunks_number=1,
    )

    elapsed = time.time() - t0
    size = Path(output_path).stat().st_size
    log.info("")
    log.info("=" * 60)
    log.info("  PIPELINE VALIDATION PASSED")
    log.info("=" * 60)
    log.info(f"  Output : {output_path} ({size:,} bytes)")
    log.info(f"  Config : {height}x{width}, {num_frames}f @ {fps}fps, {num_steps} steps")
    log.info(f"  Device : {device} ({dtype})")
    log.info(f"  Time   : {elapsed:.1f}s")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="AIPROD end-to-end pipeline validation (mock models)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cuda', 'cpu', or 'auto' (default: auto)",
    )
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default="output/validation_test.mp4"
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    run_validation(
        device=device,
        height=args.height,
        width=args.width,
        num_frames=args.frames,
        fps=args.fps,
        num_steps=args.steps,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
