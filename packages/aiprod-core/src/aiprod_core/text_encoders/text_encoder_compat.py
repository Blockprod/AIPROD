# Backward-compat shim — old import path ``aiprod_core.text_encoders.text_encoder_compat``
# Use ``aiprod_core.model.text_encoder`` for new code.

from __future__ import annotations

from typing import TYPE_CHECKING

from aiprod_core.model.text_encoder import LLMBridge

# Legacy aliases — kept for backward compatibility only
AIPRODTextEncoderBase = LLMBridge
AIPRODTextEncoderModel = LLMBridge


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
    "AIPRODTextEncoderBase",
    "AIPRODTextEncoderModel",
    "encode_text",
]
