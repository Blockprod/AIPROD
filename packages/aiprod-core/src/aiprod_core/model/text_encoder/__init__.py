# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
LLM Bridge — Flexible Text Encoder Integration

AIPROD's pluggable text encoding system. Unlike hardcoded integrations
with specific LLMs, the LLM Bridge provides:
    - A unified interface for any text encoder backend
    - Automatic projection to the model's expected dimension
    - Support for multiple LLM backends (local or API)
    - Caching and batching of text embeddings
"""

from .bridge import LLMBridge, LLMBridgeConfig, TextEncoderBackend

# Pipeline-compatible alias: the old API used AVGemmaTextEncoderModel
# everywhere — point it to our pluggable LLMBridge.
AVGemmaTextEncoderModel = LLMBridge
GemmaTextEncoderModelBase = LLMBridge

__all__ = [
    "LLMBridge",
    "LLMBridgeConfig",
    "TextEncoderBackend",
    "AVGemmaTextEncoderModel",
    "GemmaTextEncoderModelBase",
]
