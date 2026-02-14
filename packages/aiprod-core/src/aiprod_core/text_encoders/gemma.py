# Backward-compat shim — old import path ``aiprod_core.text_encoders.gemma``
# Use ``aiprod_core.model.text_encoder`` for new code.

from __future__ import annotations

from typing import TYPE_CHECKING

from aiprod_core.model.text_encoder import LLMBridge

# Legacy aliases
GemmaTextEncoderModelBase = LLMBridge
AVGemmaTextEncoderModel = LLMBridge


def encode_text(
    text_encoder: LLMBridge,
    prompts: list[str],
) -> list[tuple]:
    """Backward-compat helper that encodes a list of prompts.

    Returns a list of ``(video_context, audio_context)`` tuples,
    one per prompt.
    """
    results = []
    for prompt in prompts:
        video_ctx, audio_ctx, _mask = text_encoder._encode_prompt(prompt)
        results.append((video_ctx, audio_ctx))
    return results


__all__ = [
    "GemmaTextEncoderModelBase",
    "AVGemmaTextEncoderModel",
    "encode_text",
]
