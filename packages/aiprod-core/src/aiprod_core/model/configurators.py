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
    def _infer_config(state_dict: dict[str, torch.Tensor]) -> SHDTConfig:
        """Infer SHDTConfig hyper-parameters from checkpoint weight shapes.

        Probes well-known weight keys to determine:
            - latent_channels  (from patch_embed.0.weight)
            - hidden_dim       (from patch_embed.0.weight)
            - text_embed_dim   (from text_proj.weight)
            - num_layers       (from max block index in keys)
            - num_heads / num_kv_heads / head_dim (from attention projections)
        Falls back to SHDTConfig defaults for anything not found.
        """
        config_kwargs: dict[str, object] = {}

        # --- latent_channels & hidden_dim from patch_embed.0.weight [hidden_dim, latent_channels] ---
        pe_key = "patch_embed.0.weight"
        if pe_key in state_dict:
            w = state_dict[pe_key]
            config_kwargs["hidden_dim"] = w.shape[0]
            config_kwargs["latent_channels"] = w.shape[1]

        # --- text_embed_dim from text_proj.weight [hidden_dim, text_embed_dim] ---
        tp_key = "text_proj.weight"
        if tp_key in state_dict:
            config_kwargs["text_embed_dim"] = state_dict[tp_key].shape[1]

        # --- num_layers from block indices ---
        block_indices: set[int] = set()
        for key in state_dict:
            if key.startswith("blocks."):
                try:
                    idx = int(key.split(".")[1])
                    block_indices.add(idx)
                except (IndexError, ValueError):
                    pass
        if block_indices:
            config_kwargs["num_layers"] = max(block_indices) + 1

        # --- num_heads / num_kv_heads / head_dim from attention projections ---
        q_key = "blocks.0.spatial_attn.attn.q_proj.weight"
        k_key = "blocks.0.spatial_attn.attn.k_proj.weight"
        if q_key in state_dict and k_key in state_dict:
            hidden_dim = config_kwargs.get("hidden_dim", SHDTConfig.hidden_dim)
            q_out = state_dict[q_key].shape[0]  # num_heads * head_dim
            k_out = state_dict[k_key].shape[0]  # num_kv_heads * head_dim
            # head_dim divides both q_out and k_out
            from math import gcd
            hd = gcd(q_out, k_out)
            # Ensure head_dim is reasonable (32..256)
            while hd > 256:
                hd //= 2
            if hd >= 32:
                config_kwargs["head_dim"] = hd
                config_kwargs["num_heads"] = q_out // hd
                config_kwargs["num_kv_heads"] = k_out // hd

        return SHDTConfig(**config_kwargs)

    @staticmethod
    def from_state_dict(
        state_dict: dict[str, torch.Tensor],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> SHDTModel:
        """Infer config from state dict, build model, and load weights."""
        config = SHDTConfigurator._infer_config(state_dict)
        model = SHDTModel(config)
        # Try to load matching keys
        model_sd = model.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_sd}
        if filtered:
            model.load_state_dict(filtered, strict=False)
        else:
            import logging
            logging.warning(
                "SHDTConfigurator: no checkpoint keys matched the SHDT model "
                "(%d keys in checkpoint vs %d in model). The model will have "
                "random weights. This usually means the checkpoint comes from "
                "an incompatible architecture and requires key remapping.",
                len(state_dict),
                len(model_sd),
            )
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
