"""
TTS Vocoder — Mel-Spectrogram to Waveform Conversion

Implements a HiFi-GAN–style generator for high-quality waveform synthesis.
Supports multi-period and multi-scale discriminators for training.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VocoderConfig:
    """Vocoder (HiFi-GAN generator) configuration."""
    # Input
    num_mels: int = 80
    # Generator
    upsample_initial_channel: int = 512
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilations: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    # Audio
    sample_rate: int = 24000


# ───────────────────────────────────────────────────────────────────────────
# Residual blocks
# ───────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Multi-dilation residual block (HiFi-GAN style)."""

    def __init__(self, channels: int, kernel_size: int, dilations: List[int]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilations:
            padding = (kernel_size * d - d) // 2
            self.convs1.append(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size,
                              dilation=d, padding=padding)
                )
            )
            self.convs2.append(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size,
                              dilation=1, padding=(kernel_size - 1) // 2)
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            residual = x
            x = F.leaky_relu(x, 0.1)
            x = c1(x)
            x = F.leaky_relu(x, 0.1)
            x = c2(x)
            x = x + residual
        return x


# ───────────────────────────────────────────────────────────────────────────
# Generator
# ───────────────────────────────────────────────────────────────────────────

class VocoderGenerator(nn.Module):
    """
    HiFi-GAN–style generator.

    mel [B, 80, T_mel] → waveform [B, 1, T_audio]
    """

    def __init__(self, config: VocoderConfig):
        super().__init__()
        self.config = config

        ch = config.upsample_initial_channel
        self.pre_conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(config.num_mels, ch, 7, padding=3)
        )

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u_rate, u_ksize) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            self.ups.append(
                nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(
                        ch, ch // 2, u_ksize,
                        stride=u_rate,
                        padding=(u_ksize - u_rate) // 2,
                    )
                )
            )
            ch_out = ch // 2
            for k, d in zip(config.resblock_kernel_sizes, config.resblock_dilations):
                self.resblocks.append(ResBlock(ch_out, k, d))
            ch = ch_out

        self.post_conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(ch, 1, 7, padding=3)
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [batch, num_mels, mel_len]
        Returns:
            waveform: [batch, 1, audio_len]
        """
        x = self.pre_conv(mel)
        num_res_per_up = len(self.config.resblock_kernel_sizes)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            # Sum of residual blocks at this scale
            xs = torch.zeros_like(x)
            for j in range(num_res_per_up):
                xs += self.resblocks[i * num_res_per_up + j](x)
            x = xs / num_res_per_up

        x = F.leaky_relu(x, 0.1)
        x = self.post_conv(x)
        x = torch.tanh(x)
        return x


# ───────────────────────────────────────────────────────────────────────────
# Discriminators (for GAN training)
# ───────────────────────────────────────────────────────────────────────────

class PeriodDiscriminator(nn.Module):
    """Sub-discriminator operating on a specific period of the waveform."""

    def __init__(self, period: int, channels: int = 32):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList([
            nn.utils.parametrizations.weight_norm(
                nn.Conv2d(1, channels, (5, 1), (3, 1), padding=(2, 0))
            ),
            nn.utils.parametrizations.weight_norm(
                nn.Conv2d(channels, channels * 2, (5, 1), (3, 1), padding=(2, 0))
            ),
            nn.utils.parametrizations.weight_norm(
                nn.Conv2d(channels * 2, channels * 4, (5, 1), (3, 1), padding=(2, 0))
            ),
            nn.utils.parametrizations.weight_norm(
                nn.Conv2d(channels * 4, channels * 4, (5, 1), 1, padding=(2, 0))
            ),
        ])
        self.final = nn.utils.parametrizations.weight_norm(
            nn.Conv2d(channels * 4, 1, (3, 1), 1, padding=(1, 0))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: waveform [batch, 1, time]
        Returns:
            logit, feature_maps
        """
        features = []
        b, c, t = x.shape
        # Pad and reshape to [batch, 1, period_len, period]
        pad_len = (self.period - (t % self.period)) % self.period
        x = F.pad(x, (0, pad_len), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)
        x = self.final(x)
        features.append(x)
        return x.flatten(1, -1), features


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (MPD) for waveform quality assessment."""

    def __init__(self, periods: Optional[List[int]] = None):
        super().__init__()
        periods = periods or [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList(
            [PeriodDiscriminator(p) for p in periods]
        )

    def forward(
        self, y: torch.Tensor, y_hat: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
               List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        Returns:
            real_logits, fake_logits, real_features, fake_features
        """
        real_logits, fake_logits = [], []
        real_features, fake_features = [], []
        for d in self.discriminators:
            r_logit, r_feat = d(y)
            f_logit, f_feat = d(y_hat)
            real_logits.append(r_logit)
            fake_logits.append(f_logit)
            real_features.append(r_feat)
            fake_features.append(f_feat)
        return real_logits, fake_logits, real_features, fake_features


# ───────────────────────────────────────────────────────────────────────────
# Unified vocoder wrapper
# ───────────────────────────────────────────────────────────────────────────

class VocoderTTS(nn.Module):
    """
    High-level vocoder wrapper.

    Usage:
        vocoder = VocoderTTS()
        waveform = vocoder(mel_spectrogram)
    """

    def __init__(self, config: Optional[VocoderConfig] = None):
        super().__init__()
        self.config = config or VocoderConfig()
        self.generator = VocoderGenerator(self.config)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [batch, num_mels, mel_len]
        Returns:
            waveform: [batch, 1, audio_len]
        """
        return self.generator(mel)

    @torch.no_grad()
    def infer(self, mel: torch.Tensor) -> torch.Tensor:
        """Inference mode (no grad, removes channel dim)."""
        self.eval()
        wav = self.generator(mel)  # [B, 1, T]
        return wav.squeeze(1)      # [B, T]
