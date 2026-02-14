# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
NAC — Neural Audio Codec

AIPROD's proprietary audio encoding/decoding, using a residual
vector-quantized (RVQ) codec architecture. Distinct from HiFi-GAN
vocoder-based approaches.

Key features:
    - RVQ-based discrete audio tokens (not continuous latents)
    - Multi-band spectral processing
    - Configurable bitrate (1.5-24 kbps)
    - 48kHz support
"""

from .codec import AudioEncoder, AudioDecoder, NACConfig

# The NAC codec handles both encoding and decoding — no separate vocoder needed.
# For pipeline compatibility, Vocoder is aliased to AudioDecoder.
Vocoder = AudioDecoder


def __getattr__(name: str):
    """Lazy imports to avoid circular dependency with configurators."""
    _configurator_names = {
        "AudioEncoderConfigurator",
        "AudioDecoderConfigurator",
        "VocoderConfigurator",
        "AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER",
        "AUDIO_VAE_DECODER_COMFY_KEYS_FILTER",
        "VOCODER_COMFY_KEYS_FILTER",
    }
    if name in _configurator_names:
        from aiprod_core.model import configurators as _cfg
        if name == "VocoderConfigurator":
            return _cfg.AudioDecoderConfigurator
        return getattr(_cfg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def decode_audio(latent, decoder, vocoder=None):
    """Decode audio latent to waveform.

    ``vocoder`` is accepted for backward compat; when the NAC codec
    is used, the decoder itself handles the full pipeline.
    """
    if vocoder is not None and vocoder is not decoder:
        intermediate = decoder(latent)
        return vocoder(intermediate)
    return decoder(latent)


__all__ = [
    "AudioEncoder",
    "AudioDecoder",
    "NACConfig",
    "Vocoder",
    "AudioEncoderConfigurator",
    "AudioDecoderConfigurator",
    "VocoderConfigurator",
    "AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER",
    "AUDIO_VAE_DECODER_COMFY_KEYS_FILTER",
    "VOCODER_COMFY_KEYS_FILTER",
    "decode_audio",
]
