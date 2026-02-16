"""
Gemini Flash Captioner — Cloud external captioning.
=====================================================

Cloud-based captioner that depends on the Google Gemini API.

This module lives in ``aiprod-cloud`` and is re-exported by the
backward-compatible shim at
``aiprod_trainer.captioning_external``.
"""

import re
from pathlib import Path

from aiprod_trainer.captioning import (
    DEFAULT_CAPTION_INSTRUCTION,
    VIDEO_ONLY_CAPTION_INSTRUCTION,
    MediaCaptioningModel,
)


class GeminiFlashCaptioner(MediaCaptioningModel):
    """Audio-visual captioning using Google's Gemini Flash API.

    Gemini Flash is a cloud-based multimodal model that natively supports
    audio and video understanding. Requires a Google API key.

    **This is an EXTERNAL dependency — not part of the sovereign stack.**
    """

    MODEL_ID = "gemini-flash-lite-latest"

    def __init__(
        self,
        api_key: str | None = None,
        instruction: str | None = None,
    ):
        self.instruction = instruction
        self._init_client(api_key)

    @property
    def supports_audio(self) -> bool:
        return True

    def caption(
        self,
        path: str | Path,
        fps: int = 3,  # noqa: ARG002
        include_audio: bool = True,
        clean_caption: bool = True,
    ) -> str:
        import time  # noqa: PLC0415

        path = Path(path)
        is_video = self._is_video_file(path)
        use_audio = include_audio and is_video

        if self.instruction is not None:
            instruction = self.instruction
        else:
            instruction = DEFAULT_CAPTION_INSTRUCTION if use_audio else VIDEO_ONLY_CAPTION_INSTRUCTION

        uploaded_file = self._genai.upload_file(path)

        while uploaded_file.state.name == "PROCESSING":
            time.sleep(1)
            uploaded_file = self._genai.get_file(uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise RuntimeError(f"File processing failed: {uploaded_file.state.name}")

        response = self._model.generate_content([uploaded_file, instruction])
        caption_raw = response.text

        self._genai.delete_file(uploaded_file.name)

        return self._clean_raw_caption(caption_raw) if clean_caption else caption_raw

    def _init_client(self, api_key: str | None) -> None:
        import os  # noqa: PLC0415

        import google.generativeai as genai  # type: ignore[import-not-found]  # noqa: PLC0415

        resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "Gemini API key is required. Provide it via the `api_key` argument "
                "or set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable."
            )

        genai.configure(api_key=resolved_api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(self.MODEL_ID)
