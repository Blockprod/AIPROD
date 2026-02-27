# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Video Generator — Proprietary generation interface.

High-level wrapper for audio-video generation that abstracts the underlying
model backend.  Current backend: LTX-2 via ``diffusers``.  When a sovereign
SHDT checkpoint is available it can be swapped transparently.

Usage::

    from aiprod_pipelines.generate import AIPRODVideoGenerator

    gen = AIPRODVideoGenerator()
    gen.generate(prompt="A cinematic drone shot …", output_path="out.mp4")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    """All tuneable knobs for a single generation run."""

    prompt: str = "A cinematic drone shot over misty mountains at golden hour"
    negative_prompt: str = (
        "worst quality, inconsistent motion, blurry, jittery, distorted, "
        "shaky, glitchy, low quality, deformed, motion smear, motion "
        "artifacts, fused fingers, bad anatomy, weird hand, ugly, "
        "transition, static"
    )
    width: int = 768
    height: int = 512
    num_frames: int = 121
    frame_rate: float = 24.0
    num_inference_steps: int = 40
    guidance_scale: float = 4.0
    seed: int = 42
    output_type: str = "np"  # "np" | "pil" | "latent"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class AIPRODVideoGenerator:
    """
    AIPROD proprietary video + audio generator.

    Wraps the diffusion backend behind a stable, AIPROD-owned API so that the
    notebook and downstream code never import ``diffusers`` directly.

    Parameters
    ----------
    model_id : str
        HuggingFace repo or local path for the diffusion model.
    device : str
        Target CUDA device (e.g. ``"cuda:0"``).
    dtype : torch.dtype
        Compute dtype — ``torch.bfloat16`` recommended for A100.
    cpu_offload : bool
        Enable sequential CPU off-load to save VRAM (recommended).
    enable_tiling : bool
        Enable VAE tiling for large resolutions.
    local_files_only : bool
        If *True* (default), never contact the network — use local
        weights only (sovereign mode).  Set to *False* on Colab/Cloud
        to allow downloading weights from HuggingFace.
    """

    def __init__(
        self,
        model_id: str = "Lightricks/LTX-2",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        cpu_offload: bool = True,
        enable_tiling: bool = True,
        local_files_only: bool = True,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self._pipe: Any | None = None
        self._cpu_offload = cpu_offload
        self._enable_tiling = enable_tiling
        self._local_files_only = local_files_only

    # -- lazy init --------------------------------------------------------

    def _ensure_pipeline(self) -> Any:
        """Load the pipeline on first use (lazy)."""
        if self._pipe is not None:
            return self._pipe

        logger.info("Loading AIPROD pipeline (model=%s) …", self.model_id)
        t0 = time.time()

        # Internal implementation: diffusers LTX2Pipeline
        from diffusers import LTX2Pipeline  # type: ignore[import-unresolved]  # noqa: WPS433

        self._pipe = LTX2Pipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            local_files_only=self._local_files_only,
        )

        if self._cpu_offload:
            self._pipe.enable_sequential_cpu_offload(device=self.device)
        else:
            self._pipe.to(self.device)

        if self._enable_tiling:
            self._pipe.vae.enable_tiling()

        logger.info("Pipeline loaded in %.0fs", time.time() - t0)
        return self._pipe

    # -- public API -------------------------------------------------------

    @property
    def audio_sample_rate(self) -> int:
        """Audio sample rate advertised by the vocoder."""
        pipe = self._ensure_pipeline()
        return int(pipe.vocoder.config.output_sampling_rate)

    def generate(
        self,
        prompt: str | None = None,
        output_path: str | Path | None = None,
        config: GenerationConfig | None = None,
        **overrides: Any,
    ) -> tuple:
        """
        Generate a video (+ audio) from a text prompt.

        Parameters
        ----------
        prompt : str, optional
            Text prompt.  Overrides ``config.prompt`` when given.
        output_path : str | Path, optional
            If provided the result is encoded as MP4 to this path.
        config : GenerationConfig, optional
            Full generation config.  Defaults are used when ``None``.
        **overrides
            Any ``GenerationConfig`` field (e.g. ``seed=123``).

        Returns
        -------
        tuple[ndarray, ndarray | None]
            ``(video, audio)`` — numpy arrays.
        """
        cfg = config or GenerationConfig()
        if prompt is not None:
            cfg.prompt = prompt
        # apply overrides
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        pipe = self._ensure_pipeline()
        generator = torch.Generator("cpu").manual_seed(cfg.seed)

        logger.info(
            "Generating %d frames @ %dx%d (%d steps, guidance=%.1f) …",
            cfg.num_frames,
            cfg.width,
            cfg.height,
            cfg.num_inference_steps,
            cfg.guidance_scale,
        )
        t0 = time.time()

        video, audio = pipe(
            prompt=cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            width=cfg.width,
            height=cfg.height,
            num_frames=cfg.num_frames,
            frame_rate=cfg.frame_rate,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            generator=generator,
            output_type=cfg.output_type,
            return_dict=False,
        )

        elapsed = time.time() - t0
        logger.info("Generation done in %.0fs", elapsed)

        # optionally save
        if output_path is not None:
            self.save_video(
                video=video,
                audio=audio,
                output_path=str(output_path),
                fps=cfg.frame_rate,
            )

        return video, audio

    def save_video(
        self,
        video,
        audio,
        output_path: str,
        fps: float = 24.0,
    ) -> Path:
        """
        Encode video + audio arrays to MP4.

        Returns the absolute path of the written file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            from diffusers.pipelines.ltx2.export_utils import encode_video as _enc  # type: ignore[import-unresolved]
        except ImportError:
            _enc = None

        if _enc is not None:
            _enc(
                video[0],
                fps=fps,
                audio=audio[0].float().cpu() if audio is not None else None,
                audio_sample_rate=self.audio_sample_rate,
                output_path=str(out),
            )
        else:
            # Fallback: AIPROD proprietary encoder
            from aiprod_pipelines.utils.media_io import encode_video as _aiprod_enc

            _aiprod_enc(
                video=video,
                fps=fps,
                audio=audio[0].float().cpu() if audio is not None else None,
                audio_sample_rate=self.audio_sample_rate,
                output_path=str(out),
                video_chunks_number=1,
            )
        logger.info("Saved %s (%.1f MB)", out, out.stat().st_size / 1024**2)
        return out

    def release(self) -> None:
        """Free GPU memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        torch.cuda.empty_cache()
