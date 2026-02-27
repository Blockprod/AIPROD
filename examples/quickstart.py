#!/usr/bin/env python3
# Copyright (c) 2025-2026 AIPROD. All rights reserved.
"""
AIPROD Quick Start — Text-to-Video Generation

Usage (one-stage, recommended for single GPU):
    python examples/quickstart.py \\
        --prompt "A drone flies over a forest at sunset" \\
        --checkpoint-path models/ltx2_research \\
        --text-encoder-root models/aiprod-sovereign/aiprod-text-encoder-v1 \\
        --output output/quickstart.mp4

    # Use --enable-fp8 for the FP8 checkpoint (ltx-2-19b-dev-fp8.safetensors)
    python examples/quickstart.py \\
        --prompt "A cat sitting on a windowsill" \\
        --checkpoint-path models/ltx2_research \\
        --text-encoder-root models/aiprod-sovereign/aiprod-text-encoder-v1 \\
        --enable-fp8 --output output/cat.mp4

    # Validate with tiny mock models (no GPU/checkpoint needed):
    python scripts/validate_pipeline_e2e.py --device cpu

Requirements:
    - GPU with >=24 GB VRAM for the 19B model (A5000, RTX 4090, etc.)
    - Or use scripts/validate_pipeline_e2e.py --device cpu to validate on any hardware
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from aiprod_core.components.guiders import MultiModalGuiderParams
from aiprod_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from aiprod_pipelines.utils.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_1_STAGE_HEIGHT,
    DEFAULT_1_STAGE_WIDTH,
    DEFAULT_AUDIO_GUIDER_PARAMS,
    DEFAULT_FRAME_RATE,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_VIDEO_GUIDER_PARAMS,
)
from aiprod_pipelines.utils.media_io import encode_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def latent_to_pixel_video(raw: torch.Tensor) -> torch.Tensor:
    """Convert decoder output [B, 3, T, H, W] float → [T, H, W, 3] uint8."""
    video = raw[0]  # [3, T, H, W]
    video = video.permute(1, 2, 3, 0)  # [T, H, W, 3]
    video = ((video.float() + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
    return video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AIPROD — Text-to-Video Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="models/ltx2_research",
        help="Path to checkpoint directory or file",
    )
    parser.add_argument(
        "--text-encoder-root",
        type=str,
        default="models/aiprod-sovereign/aiprod-text-encoder-v1",
        help="Path to text encoder directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful sunset over mountains with vibrant orange and purple colors",
        help="Text prompt for video generation",
    )

    # Output
    parser.add_argument(
        "--output", type=str, default="output/quickstart.mp4", help="Output video path"
    )

    # Generation params
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--height", type=int, default=DEFAULT_1_STAGE_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_1_STAGE_WIDTH)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--frame-rate", type=float, default=DEFAULT_FRAME_RATE)
    parser.add_argument(
        "--num-inference-steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS
    )
    parser.add_argument(
        "--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT
    )

    # Model options
    parser.add_argument(
        "--enable-fp8",
        action="store_true",
        default=False,
        help="Use FP8 quantization for the transformer",
    )
    parser.add_argument(
        "--lora",
        type=str,
        nargs="*",
        default=[],
        help="LoRA checkpoint paths (optional)",
    )

    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    t0 = time.time()

    # ── Validate paths ────────────────────────────────────────────────────
    ckpt = Path(args.checkpoint_path)
    te = Path(args.text_encoder_root)

    if not ckpt.exists():
        log.error(f"Checkpoint path not found: {ckpt}")
        log.error("Download models first: python scripts/download_models.py")
        sys.exit(1)
    if not te.exists():
        log.error(f"Text encoder path not found: {te}")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ── Print config ──────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  AIPROD — Text-to-Video Generation")
    log.info("=" * 60)
    log.info(f"  Prompt  : {args.prompt[:80]}...")
    log.info(f"  Res     : {args.height}x{args.width}, {args.num_frames}f @ {args.frame_rate}fps")
    log.info(f"  Steps   : {args.num_inference_steps}")
    log.info(f"  FP8     : {args.enable_fp8}")
    log.info(f"  Output  : {args.output}")
    log.info("=" * 60)

    # ── Load pipeline ─────────────────────────────────────────────────────
    log.info("Loading pipeline...")

    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=str(ckpt),
        text_encoder_root=str(te),
        loras=tuple(args.lora),
        fp8transformer=args.enable_fp8,
    )

    log.info("Pipeline loaded")

    # ── Generate ──────────────────────────────────────────────────────────
    log.info("Generating video...")

    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=DEFAULT_VIDEO_GUIDER_PARAMS,
        audio_guider_params=DEFAULT_AUDIO_GUIDER_PARAMS,
        images=[],
    )

    log.info("Generation complete, decoding video...")

    # ── Post-process & save ───────────────────────────────────────────────
    # Convert [B, 3, T, H, W] float → [T, H, W, 3] uint8
    pixel_video = latent_to_pixel_video(video)

    encode_video(
        video=pixel_video,
        fps=int(args.frame_rate),
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE if audio is not None else None,
        output_path=args.output,
        video_chunks_number=1,
    )

    elapsed = time.time() - t0
    size = Path(args.output).stat().st_size
    log.info("")
    log.info("=" * 60)
    log.info("  VIDEO GENERATED SUCCESSFULLY")
    log.info("=" * 60)
    log.info(f"  Output : {args.output} ({size:,} bytes)")
    log.info(f"  Time   : {elapsed:.1f}s")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
