# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
TTS Module — Text-to-Speech Engine

AIPROD's proprietary multi-speaker, multi-language TTS system.

Modules:
    TTSModel         – Full encoder-decoder TTS pipeline
    TextFrontend     – Text normalization & G2P
    ProsodyModeler   – Duration / F0 / energy prediction
    SpeakerEmbedding – Multi-speaker & zero-shot cloning
    VocoderTTS       – HiFi-GAN waveform synthesis
"""

from .model import TTSModel, TTSConfig
from .text_frontend import TextFrontend, FrontendConfig
from .prosody import ProsodyModeler, ProsodyConfig
from .speaker_embedding import SpeakerEmbedding, SpeakerConfig
from .vocoder_tts import VocoderTTS, VocoderConfig

__all__ = [
    "TTSModel", "TTSConfig",
    "TextFrontend", "FrontendConfig",
    "ProsodyModeler", "ProsodyConfig",
    "SpeakerEmbedding", "SpeakerConfig",
    "VocoderTTS", "VocoderConfig",
]
