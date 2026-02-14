"""Lip-Sync Module — Audio-to-Facial Animation

Synchronises audio to facial movements using neural networks.
Supports real-time inference and high-quality rendering.

Architecture:
    mel-spectrogram → AudioEncoder (Conv1D stack)
                    → TemporalProcessor (BiLSTM)
                    → FacialDecoder (Linear → FLAME 52-params)

Metrics:
    LSE-D  — Lip Sync Error Distance  (↓ target ≤ 7.0)
    LSE-C  — Lip Sync Error Confidence (↑ higher is better)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class LipSyncConfig:
    """Lip-Sync Configuration."""
    # Audio processing
    sample_rate: int = 16000
    audio_dim: int = 80          # mel-spectrogram bins
    audio_context: int = 5       # frames of context each side

    # Video processing
    video_fps: int = 30
    num_facial_params: int = 52  # FLAME model blend-shapes

    # Model architecture
    hidden_dim: int = 256
    num_layers: int = 3
    conv_channels: int = 128
    dropout: float = 0.1

    # Loss & metrics
    sync_loss_type: str = "lse"  # "lse" | "mse"
    target_lse_d: float = 7.0    # Sync distance threshold


# ───────────────────────────────────────────────────────────────────────────
# Audio encoder
# ───────────────────────────────────────────────────────────────────────────

class AudioEncoder(nn.Module):
    """Conv1D stack that encodes mel-spectrogram frames → audio features."""

    def __init__(self, audio_dim: int, out_dim: int, channels: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            # block 1:  [B, audio_dim, T] → [B, channels, T]
            nn.Conv1d(audio_dim, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            # block 2
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            # block 3
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),

            # project → out_dim
            nn.Conv1d(channels, out_dim, kernel_size=1),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [batch, frames, audio_dim]
        Returns:
            [batch, frames, out_dim]
        """
        x = mel.transpose(1, 2)     # → [B, audio_dim, T]
        x = self.net(x)             # → [B, out_dim, T]
        return x.transpose(1, 2)    # → [B, T, out_dim]


# ───────────────────────────────────────────────────────────────────────────
# Temporal processor (BiLSTM)
# ───────────────────────────────────────────────────────────────────────────

class TemporalProcessor(nn.Module):
    """Bi-directional LSTM for audio-visual temporal modelling."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)  # merge directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D_in] → [B, T, hidden]"""
        out, _ = self.lstm(x)
        return self.proj(out)


# ───────────────────────────────────────────────────────────────────────────
# Facial parameter decoder
# ───────────────────────────────────────────────────────────────────────────

class FacialDecoder(nn.Module):
    """Decodes temporal features into FLAME blend-shape parameters."""

    def __init__(self, hidden_dim: int, num_params: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_params),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, hidden] → [B, T, num_params]"""
        return self.net(x)


# ───────────────────────────────────────────────────────────────────────────
# Full Lip-Sync model
# ───────────────────────────────────────────────────────────────────────────

class LipSyncModel(nn.Module):
    """
    Lip-Sync Neural Network.

    Given audio mel-spectrogram features, predicts per-frame FLAME 52
    blend-shape parameters that are synchronous with the audio.
    """

    def __init__(self, config: LipSyncConfig):
        super().__init__()
        self.config = config

        self.audio_encoder = AudioEncoder(
            audio_dim=config.audio_dim,
            out_dim=config.hidden_dim,
            channels=config.conv_channels,
        )
        self.temporal = TemporalProcessor(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.decoder = FacialDecoder(
            hidden_dim=config.hidden_dim,
            num_params=config.num_facial_params,
        )

    # ── Forward ────────────────────────────────────────────────────────

    def forward(
        self,
        audio_features: torch.Tensor,              # [B, T, audio_dim]
        facial_params: Optional[torch.Tensor] = None,  # [B, T, 52]
    ) -> dict:
        """
        Args:
            audio_features: mel-spectrogram input
            facial_params:  ground-truth (training only)
        Returns:
            dict with predicted_params, and optionally loss / metrics
        """
        enc = self.audio_encoder(audio_features)   # [B, T, D]
        temporal = self.temporal(enc)               # [B, T, D]
        predicted = self.decoder(temporal)          # [B, T, 52]

        result: Dict[str, torch.Tensor] = {"predicted_params": predicted}

        if facial_params is not None:
            loss, metrics = self.sync_loss(predicted, facial_params)
            result["loss"] = loss
            result["metrics"] = metrics

        return result

    # ── Loss functions ─────────────────────────────────────────────────

    def sync_loss(
        self, predicted: torch.Tensor, ground_truth: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute synchronisation loss.

        Returns:
            loss:    scalar loss tensor
            metrics: dict with lse_d, lse_c, mse
        """
        # MSE component — frame-level blend-shape match
        mse = F.mse_loss(predicted, ground_truth)

        # LSE-D (Lip Sync Error – Distance)
        # Euclidean distance between predicted and GT params (lower-face subset)
        # FLAME jaw/lips are indices 0-17 by convention
        lip_pred = predicted[:, :, :18]
        lip_gt = ground_truth[:, :, :18]
        dist = torch.norm(lip_pred - lip_gt, dim=-1)  # [B, T]
        lse_d = dist.mean()

        # LSE-C (Lip Sync Error – Confidence)
        # Cosine similarity between predicted and GT lip params
        cos_sim = F.cosine_similarity(lip_pred, lip_gt, dim=-1)  # [B, T]
        lse_c = cos_sim.mean()

        # Combined loss
        loss = mse + 0.5 * lse_d

        metrics = {
            "mse": mse.item(),
            "lse_d": lse_d.item(),
            "lse_c": lse_c.item(),
        }
        return loss, metrics

    # ── Inference ──────────────────────────────────────────────────────

    @torch.no_grad()
    def infer(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Predict facial params from audio (no GT required).

        Args:
            audio_features: [1, T, audio_dim] or [T, audio_dim]
        Returns:
            facial_params: [T, 52]
        """
        self.eval()
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
        result = self.forward(audio_features)
        return result["predicted_params"].squeeze(0)
