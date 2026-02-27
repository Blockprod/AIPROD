#!/usr/bin/env python3
# Copyright (c) 2025-2026 AIPROD. All rights reserved.
"""
AIPROD — Video Generation on AWS GPU

Optimized for g5.xlarge (A10G 24GB VRAM). Generates a video from a text
prompt using the full AIPROD pipeline with real pre-trained weights.

Usage:
    python scripts/generate_video_aws.py --prompt "A drone over mountains at sunset"
    python scripts/generate_video_aws.py --prompt "A cat on a windowsill" --steps 30 --seed 42
    python scripts/generate_video_aws.py --prompt "Ocean waves" --height 512 --width 768 --frames 61

Output: output/<sanitized_prompt>.mp4

Estimated time on g5.xlarge (A10G):
    - 512×768, 61 frames, 40 steps: ~3-5 min
    - 512×768, 121 frames, 40 steps: ~5-10 min
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def sanitize_filename(text: str, max_len: int = 60) -> str:
    """Convert a prompt to a safe filename."""
    clean = re.sub(r"[^a-zA-Z0-9_\- ]", "", text)
    clean = re.sub(r"\s+", "_", clean.strip())
    return clean[:max_len].rstrip("_")


def check_gpu():
    """Verify GPU is available and print info."""
    if not torch.cuda.is_available():
        log.error("No CUDA GPU detected. This script requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    if vram_gb < 20:
        log.warning(
            f"Only {vram_gb:.1f} GB VRAM detected. "
            "The 19B FP8 model needs ~20 GB. Generation may fail with OOM."
        )
    return vram_gb


def check_models(checkpoint_path: str, text_encoder_root: str):
    """Verify model files exist."""
    ckpt = Path(checkpoint_path)
    te = Path(text_encoder_root)

    if not ckpt.exists():
        log.error(f"Checkpoint not found: {ckpt}")
        log.error("Upload models first. See deploy/aws/setup_ec2_gpu.sh")
        sys.exit(1)

    # Find the actual safetensors file
    if ckpt.is_dir():
        st_files = list(ckpt.glob("*.safetensors"))
        if not st_files:
            log.error(f"No .safetensors file found in {ckpt}")
            sys.exit(1)
        log.info(f"Checkpoint: {st_files[0].name} ({st_files[0].stat().st_size / 1e9:.1f} GB)")

    if not te.exists():
        log.error(f"Text encoder not found: {te}")
        sys.exit(1)

    te_model = te / "model.safetensors"
    if te_model.exists():
        log.info(f"Text encoder: {te_model.stat().st_size / 1e9:.1f} GB")


@torch.inference_mode()
def generate(
    prompt: str,
    negative_prompt: str,
    checkpoint_path: str,
    text_encoder_root: str,
    output_path: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    num_inference_steps: int,
    enable_fp8: bool,
) -> None:
    """Run the full generation pipeline."""
    from aiprod_core.components.guiders import MultiModalGuiderParams
    from aiprod_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from aiprod_pipelines.utils.constants import (
        AUDIO_SAMPLE_RATE,
        DEFAULT_AUDIO_GUIDER_PARAMS,
        DEFAULT_VIDEO_GUIDER_PARAMS,
    )
    from aiprod_pipelines.utils.media_io import encode_video

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Load Pipeline ─────────────────────────────────────────────────────
    log.info("Loading pipeline (this takes 2-5 minutes for the 19B model)...")
    t_load = time.time()

    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=checkpoint_path,
        text_encoder_root=text_encoder_root,
        loras=(),
        fp8transformer=enable_fp8,
    )

    log.info(f"Pipeline loaded in {time.time() - t_load:.0f}s")

    # ── Generate ──────────────────────────────────────────────────────────
    log.info(f"Generating: {height}x{width}, {num_frames}f @ {frame_rate}fps, {num_inference_steps} steps")
    t_gen = time.time()

    video_result, audio_result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        num_inference_steps=num_inference_steps,
        video_guider_params=DEFAULT_VIDEO_GUIDER_PARAMS,
        audio_guider_params=DEFAULT_AUDIO_GUIDER_PARAMS,
        images=[],
    )

    log.info(f"Generation done in {time.time() - t_gen:.0f}s")

    # ── Decode & Save ─────────────────────────────────────────────────────
    log.info("Encoding MP4...")

    # video_result is an iterator of [T, H, W, 3] chunks or a single tensor
    # The pipeline returns decoded_video (from vae_decode_video) which returns
    # the decoder output directly.
    # Convert: [B, 3, T, H, W] float → [T, H, W, 3] uint8
    if isinstance(video_result, torch.Tensor):
        video = video_result[0]  # Remove batch dim: [3, T, H, W]
        video = video.permute(1, 2, 3, 0)  # [T, H, W, 3]
        video = ((video.float() + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)
    else:
        # Iterator — collect chunks
        chunks = list(video_result)
        video = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        if video.ndim == 5:  # [B, 3, T, H, W]
            video = video[0].permute(1, 2, 3, 0)
            video = ((video.float() + 1.0) / 2.0 * 255.0).clamp(0, 255).to(torch.uint8)

    encode_video(
        video=video,
        fps=int(frame_rate),
        audio=audio_result if audio_result is not None else None,
        audio_sample_rate=AUDIO_SAMPLE_RATE if audio_result is not None else None,
        output_path=output_path,
        video_chunks_number=1,
    )

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    duration = num_frames / frame_rate
    log.info(f"Saved: {output_path} ({size_mb:.1f} MB, {duration:.1f}s)")


def main():
    from aiprod_pipelines.utils.constants import (
        DEFAULT_1_STAGE_HEIGHT,
        DEFAULT_1_STAGE_WIDTH,
        DEFAULT_FRAME_RATE,
        DEFAULT_NEGATIVE_PROMPT,
        DEFAULT_NUM_FRAMES,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_SEED,
    )

    parser = argparse.ArgumentParser(
        description="AIPROD — Generate video on AWS GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--checkpoint-path", type=str, default="models/ltx2_research")
    parser.add_argument("--text-encoder-root", type=str, default="models/aiprod-sovereign/aiprod-text-encoder-v1")
    parser.add_argument("--output", type=str, default=None, help="Output path (auto-generated from prompt if omitted)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--height", type=int, default=DEFAULT_1_STAGE_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_1_STAGE_WIDTH)
    parser.add_argument("--num-frames", type=int, default=61, help="Number of frames (61 = ~2.5s at 24fps)")
    parser.add_argument("--frame-rate", type=float, default=DEFAULT_FRAME_RATE)
    parser.add_argument("--steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS, dest="num_inference_steps")
    parser.add_argument("--enable-fp8", action="store_true", default=True, help="FP8 quantization (default: on)")
    parser.add_argument("--no-fp8", action="store_false", dest="enable_fp8", help="Disable FP8")

    args = parser.parse_args()

    # Auto-generate output path from prompt
    if args.output is None:
        args.output = f"output/{sanitize_filename(args.prompt)}.mp4"

    t0 = time.time()

    # ── Pre-flight checks ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("  AIPROD — Video Generation on AWS")
    log.info("=" * 60)
    log.info(f"  Prompt : {args.prompt}")
    log.info(f"  Config : {args.height}x{args.width}, {args.num_frames}f @ {args.frame_rate}fps")
    log.info(f"  Steps  : {args.num_inference_steps}, FP8: {args.enable_fp8}")
    log.info(f"  Output : {args.output}")
    log.info("=" * 60)

    vram = check_gpu()
    check_models(args.checkpoint_path, args.text_encoder_root)

    # ── Generate ──────────────────────────────────────────────────────────
    generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        checkpoint_path=args.checkpoint_path,
        text_encoder_root=args.text_encoder_root,
        output_path=args.output,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        enable_fp8=args.enable_fp8,
    )

    elapsed = time.time() - t0
    log.info("")
    log.info("=" * 60)
    log.info("  VIDEO GENERATED SUCCESSFULLY")
    log.info("=" * 60)
    log.info(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info(f"  Output: {args.output}")
    log.info("")
    log.info("  Download to your PC:")
    log.info(f"    scp ubuntu@<ip>:~/aiprod/{args.output} .")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
