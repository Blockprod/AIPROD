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

# Pipeline-compatible aliases
AIPRODTextEncoder = LLMBridge
AIPRODTextEncoderBase = LLMBridge
AIPRODTextEncoderModel = LLMBridge


def encode_text(
    text_encoder: LLMBridge,
    prompts: list[str],
) -> list[tuple]:
    """Encode a list of prompts into (video_context, audio_context) tuples."""
    results = []
    for prompt in prompts:
        video_ctx, audio_ctx, _mask = text_encoder._encode_prompt(prompt)
        results.append((video_ctx, audio_ctx))
    return results


__all__ = [
    "LLMBridge",
    "LLMBridgeConfig",
    "TextEncoderBackend",
    "AIPRODTextEncoder",
    "AIPRODTextEncoderBase",
    "AIPRODTextEncoderModel",
    "encode_text",
]
