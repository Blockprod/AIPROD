"""
Speaker Embedding — Multi-speaker Identity Encoding

Provides learned speaker embeddings for conditioning the TTS decoder.
Supports lookup-table embeddings and external d-vector / x-vector
speaker representations.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpeakerConfig:
    """Speaker embedding configuration."""
    num_speakers: int = 100
    embedding_dim: int = 512
    use_external_embeddings: bool = False
    # If using a speaker encoder network
    encoder_input_dim: int = 80   # mel bins
    encoder_hidden_dim: int = 256
    encoder_num_layers: int = 3


class SpeakerLookup(nn.Module):
    """Simple lookup-table speaker embeddings (used during training)."""

    def __init__(self, config: SpeakerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.num_speakers, config.embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, speaker_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            speaker_id: [batch] integer speaker IDs
        Returns:
            speaker_emb: [batch, embedding_dim]
        """
        return self.embedding(speaker_id)


class SpeakerEncoder(nn.Module):
    """
    LSTM-based speaker encoder (d-vector style).
    
    Takes a mel-spectrogram from a reference utterance and produces
    a fixed-size speaker embedding.  Used for *zero-shot* voice
    cloning when the speaker is not in the lookup table.
    """

    def __init__(self, config: SpeakerConfig):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.encoder_input_dim,
            hidden_size=config.encoder_hidden_dim,
            num_layers=config.encoder_num_layers,
            batch_first=True,
        )
        self.projection = nn.Linear(config.encoder_hidden_dim, config.embedding_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: reference mel-spectrogram [batch, time, mel_bins]
        Returns:
            speaker_emb: [batch, embedding_dim]
        """
        _, (hidden, _) = self.lstm(mel)          # hidden: [layers, batch, hidden]
        top_hidden = hidden[-1]                   # [batch, hidden]
        emb = self.projection(top_hidden)         # [batch, emb_dim]
        return F.normalize(emb, p=2, dim=-1)      # L2-normalise


class SpeakerEmbedding(nn.Module):
    """
    Unified speaker conditioning module.

    Mode A – lookup:  speaker_id → embedding  (training with known speakers)
    Mode B – encoder: reference_mel → embedding  (zero-shot / cloning)
    """

    def __init__(self, config: Optional[SpeakerConfig] = None):
        super().__init__()
        self.config = config or SpeakerConfig()
        self.lookup = SpeakerLookup(self.config)
        self.encoder = SpeakerEncoder(self.config)
        self._speaker_names: Dict[str, int] = {"default": 0}

    # ── Public API ─────────────────────────────────────────────────────

    def register_speaker(self, name: str, speaker_id: int) -> None:
        """Register a named speaker."""
        self._speaker_names[name] = speaker_id

    def get_speaker_id(self, name: str) -> int:
        """Resolve speaker name → ID (default = 0)."""
        return self._speaker_names.get(name, 0)

    def from_id(self, speaker_id: torch.Tensor) -> torch.Tensor:
        """Get embedding by speaker ID."""
        return self.lookup(speaker_id)

    def from_reference(self, reference_mel: torch.Tensor) -> torch.Tensor:
        """Get embedding from reference audio mel-spectrogram."""
        return self.encoder(reference_mel)

    def forward(
        self,
        speaker_id: Optional[torch.Tensor] = None,
        reference_mel: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Produce a speaker embedding (from ID or reference audio).

        Args:
            speaker_id:    [batch] int tensor   – known speaker
            reference_mel: [batch, T, mel_bins] – unknown speaker (cloning)
        Returns:
            embedding: [batch, embedding_dim]
        """
        if reference_mel is not None:
            return self.from_reference(reference_mel)
        if speaker_id is not None:
            return self.from_id(speaker_id)
        raise ValueError("Provide either speaker_id or reference_mel")
