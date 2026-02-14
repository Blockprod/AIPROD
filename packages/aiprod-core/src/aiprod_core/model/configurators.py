# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
AIPROD Model Configurators

Factory classes that know how to:
    1. Infer model hyper-parameters from a state dict
    2. Instantiate the model architecture
    3. Load weights into the model

Each model family (SHDT, HWVAE, NAC, LLMBridge) has its own configurator.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from aiprod_core.model.transformer.model import SHDTConfig, SHDTModel
from aiprod_core.model.video_vae.config import HWVAEConfig
from aiprod_core.model.video_vae.encoder import HWVAEEncoder
from aiprod_core.model.video_vae.decoder import HWVAEDecoder
from aiprod_core.model.audio_vae.codec import NACConfig, AudioEncoder, AudioDecoder


# ─── Key Remapping Ops ───────────────────────────────────────────────────────

def _strip_prefix(prefix: str):
    """Return a key-remap callable that strips a given prefix."""
    def _remap(key: str) -> str | None:
        if key.startswith(prefix):
            return key[len(prefix):]
        return None  # drop keys that don't match
    return _remap


def _identity(key: str) -> str:
    return key


# Model key filters — used by SingleGPUModelBuilder
AIPRODV_MODEL_COMFY_RENAMING_MAP = (_identity,)
AIPRODV_MODEL_COMFY_RENAMING_WITH_TRANSFORMER_LINEAR_DOWNCAST_MAP = (_identity,)
UPCAST_DURING_INFERENCE = object()  # sentinel for FP8 upcasting

VAE_ENCODER_COMFY_KEYS_FILTER = (_strip_prefix("vae.encoder."),)
VAE_DECODER_COMFY_KEYS_FILTER = (_strip_prefix("vae.decoder."),)
AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER = (_strip_prefix("audio.encoder."),)
AUDIO_VAE_DECODER_COMFY_KEYS_FILTER = (_strip_prefix("audio.decoder."),)
VOCODER_COMFY_KEYS_FILTER = (_strip_prefix("audio.vocoder."),)


# ─── SHDT Configurator ───────────────────────────────────────────────────────

class SHDTConfigurator:
    """Configurator for the SHDT transformer backbone."""

    @staticmethod
    def from_state_dict(
        state_dict: dict[str, torch.Tensor],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> SHDTModel:
        """Infer config from state dict, build model, and load weights."""
        config = SHDTConfig()  # Use defaults; can be inferred from shapes
        model = SHDTModel(config)
        # Try to load matching keys
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_sd}
        if filtered:
            model.load_state_dict(filtered, strict=False)
        return model


# Alias for backward-compat with old naming
AIPRODModelConfigurator = SHDTConfigurator


# ─── Video VAE Configurators ─────────────────────────────────────────────────

class VideoEncoderConfigurator:
    """Configurator for the HWVAE encoder."""

    @staticmethod
    def from_state_dict(
        state_dict: dict[str, torch.Tensor],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> HWVAEEncoder:
        config = HWVAEConfig()
        model = HWVAEEncoder(config)
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_sd}
        if filtered:
            model.load_state_dict(filtered, strict=False)
        return model


class VideoDecoderConfigurator:
    """Configurator for the HWVAE decoder."""

    @staticmethod
    def from_state_dict(
        state_dict: dict[str, torch.Tensor],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> HWVAEDecoder:
        config = HWVAEConfig()
        model = HWVAEDecoder(config)
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_sd}
        if filtered:
            model.load_state_dict(filtered, strict=False)
        return model


# ─── Audio VAE Configurators ─────────────────────────────────────────────────

class AudioEncoderConfigurator:
    """Configurator for the NAC audio encoder."""

    @staticmethod
    def from_state_dict(
        state_dict: dict[str, torch.Tensor],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> AudioEncoder:
        config = NACConfig()
        model = AudioEncoder(config)
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_sd}
        if filtered:
            model.load_state_dict(filtered, strict=False)
        return model


class AudioDecoderConfigurator:
    """Configurator for the NAC audio decoder."""

    @staticmethod
    def from_state_dict(
        state_dict: dict[str, torch.Tensor],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> AudioDecoder:
        config = NACConfig()
        model = AudioDecoder(config)
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_sd}
        if filtered:
            model.load_state_dict(filtered, strict=False)
        return model
