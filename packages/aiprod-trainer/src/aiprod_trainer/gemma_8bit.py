# ruff: noqa: PLC0415

"""
8-bit text encoder loading utilities.

This module provides functionality for loading the LLMBridge text encoder in 8-bit precision
using bitsandbytes, which significantly reduces GPU memory usage.

This replaces the old Gemma-specific 8-bit loader with a generic approach that
uses the LLMBridge's pluggable backend architecture.

Example usage::

    from aiprod_trainer.gemma_8bit import load_8bit_text_encoder

    text_encoder = load_8bit_text_encoder(
        checkpoint_path="/path/to/AIPROD2.safetensors",
        text_model_path="/path/to/gemma-or-llama",
    )
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from aiprod_core.model.text_encoder import LLMBridge


def load_8bit_text_encoder(
    checkpoint_path: str | Path,
    text_model_path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
) -> "LLMBridge":
    """Load the text encoder in 8-bit precision using bitsandbytes.

    This function creates an LLMBridge whose internal LLM encoder is
    loaded in 8-bit quantisation via the ``BitsAndBytesConfig``.

    Args:
        checkpoint_path: Path to the AIPROD safetensors checkpoint file
            (used for loading projection weights).
        text_model_path: Path to the HuggingFace model directory
            (Gemma, LLaMA, Mistral, etc.).
        dtype: Data type for non-quantized model weights (projection layers).

    Returns:
        Loaded LLMBridge with 8-bit quantized LLM backbone.

    Raises:
        ImportError: If bitsandbytes or transformers are not installed.
        FileNotFoundError: If the model directory does not exist.
    """
    try:
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    except ImportError as e:
        raise ImportError(
            "8-bit text encoder loading requires transformers and bitsandbytes. "
            "Install with: uv pip install transformers bitsandbytes"
        ) from e

    from aiprod_core.model.text_encoder import LLMBridge, LLMBridgeConfig

    text_model_path = Path(text_model_path)
    if not text_model_path.is_dir():
        raise FileNotFoundError(f"Text model path not found: {text_model_path}")

    # Load quantised LLM backbone
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    with _suppress_accelerate_memory_warnings():
        encoder = AutoModel.from_pretrained(
            str(text_model_path),
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(text_model_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-detect hidden dimension from the loaded model
    hidden_dim = getattr(encoder.config, "hidden_size", 2048)

    # Create LLMBridge with matching config
    config = LLMBridgeConfig(
        model_name=str(text_model_path),
        hidden_dim=hidden_dim,
    )
    bridge = LLMBridge(config)

    # Inject the pre-loaded 8-bit encoder and tokenizer
    bridge._encoder = encoder
    bridge._tokenizer = tokenizer

    # Move projection layers to the same device as the encoder,
    # using the non-quantized dtype
    encoder_device = next(encoder.parameters()).device
    bridge.projection = bridge.projection.to(device=encoder_device, dtype=dtype)

    # Optionally load projection weights from the checkpoint
    _load_projection_weights(bridge, checkpoint_path, encoder_device, dtype)

    return bridge


def _load_projection_weights(
    bridge: "LLMBridge",
    checkpoint_path: str | Path,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Attempt to load projection weights from the AIPROD checkpoint.

    If the checkpoint contains keys matching the projection layer,
    they are loaded. Otherwise, the randomly-initialised projection is kept.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return

    try:
        from safetensors.torch import load_file
    except ImportError:
        return

    state_dict = load_file(str(checkpoint_path))

    # Look for projection-related keys (various naming conventions)
    prefixes = [
        "text_encoder.projection.",
        "feature_extractor_linear.",
        "embeddings_connector.",
    ]
    proj_sd: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                clean_key = key[len(prefix):]
                proj_sd[clean_key] = value.to(device=device, dtype=dtype)
                break

    if proj_sd:
        try:
            bridge.projection.load_state_dict(proj_sd, strict=False)
        except RuntimeError:
            pass  # Silently ignore mismatches — projection will train from scratch


@contextmanager
def _suppress_accelerate_memory_warnings() -> Generator[None, None, None]:
    """Temporarily suppress INFO warnings from accelerate about memory allocation."""
    accelerate_logger = logging.getLogger("accelerate.utils.modeling")
    old_level = accelerate_logger.level
    accelerate_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        accelerate_logger.setLevel(old_level)


# ── Backward-compatible alias ────────────────────────────────────────────────
load_8bit_gemma = load_8bit_text_encoder
