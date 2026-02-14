"""
TTS Prosody Modelling — Duration, Pitch (F0), and Energy Prediction

Predicts prosodic features from encoded text representations
to drive natural-sounding speech synthesis.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class ProsodyConfig:
    """Prosody predictor configuration."""
    hidden_dim: int = 256
    kernel_size: int = 3
    num_conv_layers: int = 2
    dropout: float = 0.1
    # F0 range (Hz)
    f0_min: float = 60.0
    f0_max: float = 900.0
    f0_bins: int = 256
    # Energy range
    energy_min: float = 0.0
    energy_max: float = 1.0
    energy_bins: int = 256


# ───────────────────────────────────────────────────────────────────────────
# Variance predictor (shared between duration / pitch / energy)
# ───────────────────────────────────────────────────────────────────────────

class VariancePredictor(nn.Module):
    """
    Conv1D stack → linear → scalar per timestep.
    Used for duration, pitch, and energy prediction.
    """

    def __init__(self, in_dim: int, config: ProsodyConfig):
        super().__init__()
        layers = []
        ch = in_dim
        for _ in range(config.num_conv_layers):
            layers.extend([
                nn.Conv1d(ch, config.hidden_dim, config.kernel_size,
                          padding=config.kernel_size // 2),
                nn.ReLU(inplace=True),
                nn.LayerNorm(config.hidden_dim),
                nn.Dropout(config.dropout),
            ])
            ch = config.hidden_dim
        self.convs = nn.Sequential(*layers)
        self.linear = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, in_dim]
        Returns:
            prediction: [batch, seq_len]
        """
        # Conv1D expects [batch, channels, seq]
        out = self.convs(x.transpose(1, 2)).transpose(1, 2)
        return self.linear(out).squeeze(-1)


# ───────────────────────────────────────────────────────────────────────────
# Length regulator (expand phonemes by predicted durations)
# ───────────────────────────────────────────────────────────────────────────

class LengthRegulator(nn.Module):
    """Expand phoneme-level features to frame-level using durations."""

    def forward(
        self,
        x: torch.Tensor,           # [batch, phoneme_len, dim]
        durations: torch.Tensor,    # [batch, phoneme_len]  (integer counts)
    ) -> torch.Tensor:
        """
        Returns:
            expanded: [batch, frame_len, dim]
        """
        outputs: list[torch.Tensor] = []
        for b in range(x.size(0)):
            reps = durations[b].long().clamp(min=0)
            expanded = torch.repeat_interleave(x[b], reps, dim=0)
            outputs.append(expanded)
        # Pad to same length in the batch
        max_len = max(o.size(0) for o in outputs)
        padded = torch.zeros(len(outputs), max_len, x.size(2),
                             device=x.device, dtype=x.dtype)
        for b, o in enumerate(outputs):
            padded[b, :o.size(0)] = o
        return padded


# ───────────────────────────────────────────────────────────────────────────
# Pitch (F0) predictor with quantised embedding
# ───────────────────────────────────────────────────────────────────────────

class PitchPredictor(nn.Module):
    """Predicts continuous F0 and provides quantised pitch embedding."""

    def __init__(self, in_dim: int, config: ProsodyConfig):
        super().__init__()
        self.config = config
        self.predictor = VariancePredictor(in_dim, config)
        self.embedding = nn.Embedding(config.f0_bins, in_dim)
        # Pre-compute bin edges
        self.register_buffer(
            "bins",
            torch.linspace(
                math.log(config.f0_min + 1),
                math.log(config.f0_max + 1),
                config.f0_bins - 1,
            ),
        )

    def _quantise(self, f0: torch.Tensor) -> torch.Tensor:
        """Map continuous F0 (Hz) → bin index."""
        log_f0 = torch.log(f0.clamp(min=1.0) + 1)
        return torch.bucketize(log_f0, self.bins)

    def forward(
        self,
        x: torch.Tensor,
        target_f0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: encoder output [batch, seq, dim]
            target_f0: ground-truth F0 in Hz (training only)
        Returns:
            pitch_embedding: [batch, seq, dim]
            predicted_f0: [batch, seq] in Hz
        """
        predicted_f0 = self.predictor(x).exp()  # predict in log-space
        if target_f0 is not None:
            bins = self._quantise(target_f0)
        else:
            bins = self._quantise(predicted_f0)
        emb = self.embedding(bins)
        return emb, predicted_f0


# ───────────────────────────────────────────────────────────────────────────
# Energy predictor
# ───────────────────────────────────────────────────────────────────────────

class EnergyPredictor(nn.Module):
    """Predicts energy (amplitude) per phoneme / frame."""

    def __init__(self, in_dim: int, config: ProsodyConfig):
        super().__init__()
        self.config = config
        self.predictor = VariancePredictor(in_dim, config)
        self.embedding = nn.Embedding(config.energy_bins, in_dim)
        self.register_buffer(
            "bins",
            torch.linspace(config.energy_min, config.energy_max,
                           config.energy_bins - 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        target_energy: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_energy = self.predictor(x)
        source = target_energy if target_energy is not None else pred_energy
        bins = torch.bucketize(source, self.bins)
        emb = self.embedding(bins)
        return emb, pred_energy


# ───────────────────────────────────────────────────────────────────────────
# Prosody modeller (aggregates duration + pitch + energy)
# ───────────────────────────────────────────────────────────────────────────

class ProsodyModeler(nn.Module):
    """
    Full prosody module:
        encoder_output → duration + F0 + energy → frame-level features
    """

    def __init__(self, encoder_dim: int, config: Optional[ProsodyConfig] = None):
        super().__init__()
        self.config = config or ProsodyConfig()

        self.duration_predictor = VariancePredictor(encoder_dim, self.config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = PitchPredictor(encoder_dim, self.config)
        self.energy_predictor = EnergyPredictor(encoder_dim, self.config)

    def forward(
        self,
        encoder_out: torch.Tensor,       # [B, phoneme_len, D]
        target_durations: Optional[torch.Tensor] = None,
        target_f0: Optional[torch.Tensor] = None,
        target_energy: Optional[torch.Tensor] = None,
        speed: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            frame_features: [B, frame_len, D]
            prosody_info: dict with predicted durations, F0, energy
        """
        # ── Duration ───────────────────────────────────────────────────
        pred_dur = self.duration_predictor(encoder_out)
        if target_durations is not None:
            dur_for_regulate = target_durations
        else:
            dur_for_regulate = (pred_dur.exp() * speed).round().clamp(min=1)

        regulated = self.length_regulator(encoder_out, dur_for_regulate)

        # ── Pitch ──────────────────────────────────────────────────────
        pitch_emb, pred_f0 = self.pitch_predictor(regulated, target_f0)

        # ── Energy ─────────────────────────────────────────────────────
        energy_emb, pred_energy = self.energy_predictor(regulated, target_energy)

        # ── Combine ────────────────────────────────────────────────────
        frame_features = regulated + pitch_emb + energy_emb

        return frame_features, {
            "predicted_duration": pred_dur,
            "predicted_f0": pred_f0,
            "predicted_energy": pred_energy,
        }
