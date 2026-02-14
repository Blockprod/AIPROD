# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
LLM Bridge — Text Encoding with pluggable LLM backends.

Provides a unified interface for encoding text prompts into
embeddings suitable for conditioning the SHDT transformer.

Supported backends:
    - LOCAL: Load a local HuggingFace model (any causal or seq2seq LLM)
    - API: Use an external API (OpenAI, Anthropic, etc.) — text only
    - SENTENCE_TRANSFORMER: Use a sentence-transformer model

The bridge handles tokenization, encoding, and projection to
the target embedding dimension transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn as nn


class TextEncoderBackend(Enum):
    """Supported text encoder backends."""
    LOCAL_CAUSAL_LM = auto()        # Local HuggingFace CausalLM
    LOCAL_SEQ2SEQ = auto()          # Local HuggingFace Seq2SeqLM
    SENTENCE_TRANSFORMER = auto()   # Sentence-transformer model
    CUSTOM = auto()                 # Custom encoder (user-provided)


@dataclass
class LLMBridgeConfig:
    """Configuration for the LLM Bridge.

    Args:
        backend: Which type of text encoder to use.
        model_name: HuggingFace model name or path.
        output_dim: Target embedding dimension (must match SHDT config).
        max_length: Maximum token sequence length.
        hidden_dim: LLM hidden dimension (auto-detected if possible).
        num_hidden_layers: Number of LLM layers to use for features.
        feature_layer: Which hidden layer to extract features from (-1 = last).
        freeze: Whether to freeze the LLM weights.
    """
    backend: TextEncoderBackend = TextEncoderBackend.LOCAL_CAUSAL_LM
    model_name: str = "meta-llama/Llama-3.2-1B"
    output_dim: int = 2048
    max_length: int = 512
    hidden_dim: int = 2048       # will be auto-detected
    feature_layer: int = -1      # last hidden layer
    freeze: bool = True
    dtype: str = "bfloat16"


class LLMBridge(nn.Module):
    """Flexible text encoder that wraps any LLM backend.

    The bridge:
        1. Tokenizes input text
        2. Runs the LLM encoder (frozen or trainable)
        3. Extracts features from a specified hidden layer
        4. Projects to the target dimension for the SHDT

    This is NOT a wrapper around a specific model. It's a pluggable
    system that can work with any HuggingFace-compatible LLM.

    Backward-compatible attributes:
        ``model`` — alias for the internal LLM encoder
        ``tokenizer`` — alias for the internal tokenizer
        ``feature_extractor_linear`` — alias for the projection head
    """

    def __init__(self, config: LLMBridgeConfig):
        super().__init__()
        self.config = config

        # Projection from LLM hidden dim to SHDT model dim
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.GELU(),
            nn.Linear(config.output_dim, config.output_dim),
        )

        # The actual LLM will be loaded lazily
        self._encoder = None
        self._tokenizer = None

    # ── Backward-compatible properties ────────────────────────────────────

    @property
    def model(self):
        """Backward-compat: alias for the internal LLM encoder."""
        return self._encoder

    @model.setter
    def model(self, value):
        self._encoder = value

    @property
    def tokenizer(self):
        """Backward-compat: alias for the internal tokenizer."""
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    @property
    def feature_extractor_linear(self):
        """Backward-compat: alias for the projection head."""
        return self.projection

    @feature_extractor_linear.setter
    def feature_extractor_linear(self, value):
        self.projection = value

    def _load_encoder(self):
        """Lazily load the LLM encoder and tokenizer."""
        if self._encoder is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "The 'transformers' package is required for LLMBridge. "
                "Install with: pip install transformers"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        self._encoder = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        if self.config.freeze:
            for param in self._encoder.parameters():
                param.requires_grad = False
            self._encoder.eval()

        # Auto-detect hidden dim
        if hasattr(self._encoder.config, "hidden_size"):
            actual_dim = self._encoder.config.hidden_size
            if actual_dim != self.config.hidden_dim:
                # Rebuild projection with correct dim
                self.projection = nn.Sequential(
                    nn.Linear(actual_dim, self.config.output_dim),
                    nn.LayerNorm(self.config.output_dim),
                    nn.GELU(),
                    nn.Linear(self.config.output_dim, self.config.output_dim),
                ).to(next(self.projection.parameters()).device)
                self.config.hidden_dim = actual_dim

    def encode_text(
        self,
        texts: list[str],
        device: Optional[torch.device] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of text strings to embeddings.

        Args:
            texts: List of text prompts.
            device: Target device.

        Returns:
            embeddings: [B, S, output_dim] text embeddings.
            mask: [B, S] attention mask.
        """
        self._load_encoder()

        if device is None:
            device = next(self.projection.parameters()).device

        # Tokenize
        tokens = self._tokenizer(
            texts,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Encode
        with torch.no_grad() if self.config.freeze else torch.enable_grad():
            outputs = self._encoder(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                output_hidden_states=True,
            )

        # Extract features from specified layer
        hidden_states = outputs.hidden_states[self.config.feature_layer]

        # Project to target dimension
        embeddings = self.projection(hidden_states)

        return embeddings, tokens["attention_mask"]

    def forward(
        self,
        input_ids_or_prompt: torch.Tensor | str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass — accepts pre-tokenized inputs OR a text prompt.

        When called with a **string**, the prompt is encoded and the result
        is returned as ``(video_context, audio_context, attention_mask)``
        (video and audio share the same embeddings).

        When called with a **tensor**, behaves as a standard forward pass
        returning ``[B, S, output_dim]`` projected text embeddings.
        """
        if isinstance(input_ids_or_prompt, str):
            return self._encode_prompt(input_ids_or_prompt)

        self._load_encoder()

        with torch.no_grad() if self.config.freeze else torch.enable_grad():
            outputs = self._encoder(
                input_ids=input_ids_or_prompt,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[self.config.feature_layer]
        return self.projection(hidden_states)

    # ── Backward-compatible high-level helpers ────────────────────────────

    def _encode_prompt(
        self,
        prompt: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode a single prompt, returning separate video & audio contexts.

        In the new unified architecture both streams share the same text
        embeddings.  The separation is maintained for API compatibility.

        Returns:
            ``(video_context, audio_context, attention_mask)``
        """
        device = next(self.projection.parameters()).device
        embeddings, mask = self.encode_text([prompt], device=device)
        # Both modalities share the same context
        return embeddings, embeddings.clone(), mask

    def _run_connectors(
        self,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply embedding connectors to pre-computed text embeddings.

        In the old architecture this ran separate video/audio connector
        heads.  In the unified LLMBridge both streams share the same
        projection, so this is effectively a pass-through.

        Args:
            prompt_embeds: [B, S, D] pre-computed hidden states.
            prompt_attention_mask: [B, S] attention mask.

        Returns:
            ``(video_embeds, audio_embeds, attention_mask)``
        """
        projected = self.projection(prompt_embeds)
        return projected, projected.clone(), prompt_attention_mask
