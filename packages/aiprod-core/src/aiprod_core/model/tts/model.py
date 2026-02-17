"""
Text-to-Speech Model Implementation

Main TTS encoder-decoder architecture (FastSpeech 2 + HiFi-GAN vocoder).
Supports multi-language, multi-speaker voice synthesis.

Architecture:
    text → TextFrontend → phoneme IDs
         → TextEncoder (Transformer)
         → ProsodyModeler (duration / F0 / energy)
         → MelDecoder (Transformer)
         → VocoderTTS (HiFi-GAN) → waveform
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_frontend import TextFrontend, FrontendConfig
from .prosody import ProsodyModeler, ProsodyConfig
from .speaker_embedding import SpeakerEmbedding, SpeakerConfig
from .vocoder_tts import VocoderTTS, VocoderConfig


# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class TTSConfig:
    """TTS Model Configuration."""
    # Encoder
    encoder_hidden: int = 384
    encoder_layers: int = 4
    encoder_heads: int = 2
    encoder_ff_dim: int = 1024
    # Decoder
    decoder_hidden: int = 384
    decoder_layers: int = 6
    decoder_heads: int = 2
    decoder_ff_dim: int = 1024
    # Text
    vocab_size: int = 150
    max_seq_length: int = 512
    # Speaker
    num_speakers: int = 100
    speaker_emb_dim: int = 512
    # Audio
    sample_rate: int = 24000
    num_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    # Training
    learning_rate: float = 1e-3
    warmup_steps: int = 4000
    dropout: float = 0.1

    def __post_init__(self):
        assert self.encoder_heads > 0
        assert self.decoder_heads > 0
        assert self.sample_rate > 0


# ───────────────────────────────────────────────────────────────────────────
# Positional encoding
# ───────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ───────────────────────────────────────────────────────────────────────────
# Text encoder
# ───────────────────────────────────────────────────────────────────────────

class TextEncoder(nn.Module):
    """Transformer encoder for phoneme sequences."""

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.encoder_hidden)
        self.pos_enc = SinusoidalPositionalEncoding(config.encoder_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_hidden,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.encoder_layers)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, phoneme_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phoneme_ids: [batch, seq_len]
        Returns:
            encoder_output: [batch, seq_len, hidden]
        """
        x = self.embedding(phoneme_ids)
        x = self.pos_enc(x)
        x = self.dropout(x)
        return self.encoder(x)


# ───────────────────────────────────────────────────────────────────────────
# Mel decoder
# ───────────────────────────────────────────────────────────────────────────

class MelDecoder(nn.Module):
    """Transformer decoder producing mel-spectrogram frames."""

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(config.decoder_hidden)
        decoder_layer = nn.TransformerEncoderLayer(  # self-attention only
            d_model=config.decoder_hidden,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, config.decoder_layers)
        self.mel_linear = nn.Linear(config.decoder_hidden, config.num_mels)
        self.postnet = PostNet(config.num_mels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: frame-level features [batch, frames, hidden]
        Returns:
            mel: [batch, num_mels, frames]
        """
        x = self.pos_enc(x)
        x = self.decoder(x)
        mel_before = self.mel_linear(x).transpose(1, 2)  # [B, mels, T]
        mel_after = mel_before + self.postnet(mel_before)
        return mel_after


class PostNet(nn.Module):
    """5-layer Conv1D postnet for mel refinement."""

    def __init__(self, num_mels: int, channels: int = 512, kernel_size: int = 5):
        super().__init__()
        layers = [
            nn.Sequential(
                nn.Conv1d(num_mels, channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(channels),
                nn.Tanh(),
            )
        ]
        for _ in range(3):
            layers.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(channels),
                nn.Tanh(),
            ))
        layers.append(nn.Sequential(
            nn.Conv1d(channels, num_mels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_mels),
        ))
        self.layers = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.layers(mel)


# ───────────────────────────────────────────────────────────────────────────
# Full TTS model
# ───────────────────────────────────────────────────────────────────────────

class TTSModel(nn.Module):
    """
    Full Text-to-Speech Model.

    Pipeline:
        phoneme_ids → TextEncoder → ProsodyModeler → MelDecoder → VocoderTTS
                                      ↑ speaker embedding added
    """

    def __init__(self, config: TTSConfig):
        super().__init__()
        self.config = config

        # Sub-modules
        self.text_frontend = TextFrontend(FrontendConfig())
        self.text_encoder = TextEncoder(config)
        self.prosody = ProsodyModeler(
            encoder_dim=config.encoder_hidden,
            config=ProsodyConfig(),
        )
        self.speaker = SpeakerEmbedding(SpeakerConfig(
            num_speakers=config.num_speakers,
            embedding_dim=config.speaker_emb_dim,
        ))
        # Project speaker embedding to encoder dim
        self.speaker_proj = nn.Linear(config.speaker_emb_dim, config.encoder_hidden)
        self.mel_decoder = MelDecoder(config)
        self.vocoder = VocoderTTS(VocoderConfig(
            num_mels=config.num_mels,
            sample_rate=config.sample_rate,
        ))

    # ── Training forward ───────────────────────────────────────────────

    def forward(
        self,
        phoneme_ids: torch.Tensor,       # [B, seq_len]
        speaker_id: torch.Tensor,        # [B]
        target_mel: Optional[torch.Tensor] = None,  # [B, mels, T]
        target_durations: Optional[torch.Tensor] = None,
        target_f0: Optional[torch.Tensor] = None,
        target_energy: Optional[torch.Tensor] = None,
        speed: float = 1.0,
    ) -> dict:
        """
        Full forward pass.

        Returns dict with:
            mel_output:       [B, mels, T]
            waveform:         [B, 1, T_audio]
            predicted_duration, predicted_f0, predicted_energy
        """
        # 1. Encode phonemes
        enc_out = self.text_encoder(phoneme_ids)  # [B, S, D]

        # 2. Add speaker conditioning
        spk_emb = self.speaker.from_id(speaker_id)         # [B, spk_dim]
        spk_proj = self.speaker_proj(spk_emb).unsqueeze(1)  # [B, 1, D]
        enc_out = enc_out + spk_proj

        # 3. Prosody (duration / pitch / energy)
        frame_features, prosody_info = self.prosody(
            enc_out,
            target_durations=target_durations,
            target_f0=target_f0,
            target_energy=target_energy,
            speed=speed,
        )

        # 4. Decode mel-spectrogram
        mel = self.mel_decoder(frame_features)  # [B, mels, T]

        # 5. Vocoder
        waveform = self.vocoder(mel)  # [B, 1, T_audio]

        return {
            "mel_output": mel,
            "waveform": waveform,
            **prosody_info,
        }

    # ── Inference (high-level) ─────────────────────────────────────────

    @torch.no_grad()
    def infer(
        self,
        text: str,
        speaker: str = "default",
        language: str = "en",
        speed: float = 1.0,
        pitch_shift: float = 0.0,
    ) -> Tuple[torch.Tensor, int]:
        """
        End-to-end inference: text → waveform.

        Args:
            text:        Raw input text
            speaker:     Speaker name (registered via speaker module)
            language:    Language code (en, fr, es, …)
            speed:       Speed multiplier (0.5–2.0)
            pitch_shift: Not yet used (reserved for future F0 shifting)

        Returns:
            waveform:    [audio_len]
            sample_rate: int
        """
        self.eval()
        device = next(self.parameters()).device

        # Text → phoneme IDs
        ids = self.text_frontend.text_to_phoneme_ids(text)
        phoneme_ids = torch.tensor([ids], dtype=torch.long, device=device)

        # Speaker
        sid = self.speaker.get_speaker_id(speaker)
        speaker_id = torch.tensor([sid], dtype=torch.long, device=device)

        # Forward
        result = self.forward(
            phoneme_ids=phoneme_ids,
            speaker_id=speaker_id,
            speed=speed,
        )

        waveform = result["waveform"].squeeze(0).squeeze(0)  # [T]
        return waveform, self.config.sample_rate
