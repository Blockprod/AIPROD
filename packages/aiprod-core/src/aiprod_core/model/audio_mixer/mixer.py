"""Audio Mixer — Multi-track Audio Processing

Handles mixing voice, music, ambient sounds, and effects.
Supports spatial audio (stereo, 5.1, binaural) and dynamic processing.

Pipeline per track:
    volume → pan → EQ → compressor → reverb → bus sum → master limiter
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class AudioMixerConfig:
    """Audio Mixer Configuration."""
    sample_rate: int = 48000
    num_channels: int = 2   # stereo
    bit_depth: int = 32
    max_tracks: int = 16

    # Effects
    enable_compression: bool = True
    enable_eq: bool = True
    enable_reverb: bool = True
    enable_limiting: bool = True

    # Spatial audio
    spatial_format: str = "stereo"  # stereo | 5.1 | binaural


@dataclass
class AudioTrack:
    """Single audio track (voice, music, ambient, FX)."""
    name: str
    audio_data: torch.Tensor        # [channels, samples]
    track_type: str                  # voice | music | ambient | fx
    volume: float = 1.0              # linear gain
    pan: float = 0.0                 # -1.0 (L) … 0.0 (C) … 1.0 (R)
    mute: bool = False
    solo: bool = False


# ───────────────────────────────────────────────────────────────────────────
# DSP helpers
# ───────────────────────────────────────────────────────────────────────────

def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _linear_to_db(linear: float, floor: float = 1e-8) -> float:
    return 20.0 * math.log10(max(linear, floor))


def _equal_power_pan(pan: float) -> Tuple[float, float]:
    """Equal-power pan law.  pan ∈ [-1, 1] → (left_gain, right_gain)."""
    angle = (pan + 1.0) * 0.25 * math.pi   # 0 … π/2
    return math.cos(angle), math.sin(angle)


# ───────────────────────────────────────────────────────────────────────────
# Biquad EQ
# ───────────────────────────────────────────────────────────────────────────

def _biquad_peaking(
    freq_hz: float, gain_db: float, q: float, sr: int,
) -> Tuple[List[float], List[float]]:
    """Design one peaking-EQ biquad section.  Returns (b, a) lists len-3."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * freq_hz / sr
    alpha = math.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * math.cos(w0)
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * math.cos(w0)
    a2 = 1.0 - alpha / A

    return [b0 / a0, b1 / a0, b2 / a0], [1.0, a1 / a0, a2 / a0]


def _apply_biquad(audio: torch.Tensor, b: List[float], a: List[float]) -> torch.Tensor:
    """Apply a second-order IIR filter (Direct Form II transposed).

    audio: [C, N]   b, a: length-3 lists.
    """
    b0, b1, b2 = b
    a1, a2 = a[1], a[2]
    C, N = audio.shape
    out = torch.zeros_like(audio)
    z1 = torch.zeros(C, device=audio.device, dtype=audio.dtype)
    z2 = torch.zeros(C, device=audio.device, dtype=audio.dtype)
    for n in range(N):
        x_n = audio[:, n]
        y_n = b0 * x_n + z1
        z1 = b1 * x_n - a1 * y_n + z2
        z2 = b2 * x_n - a2 * y_n
        out[:, n] = y_n
    return out


# ───────────────────────────────────────────────────────────────────────────
# AudioMixer
# ───────────────────────────────────────────────────────────────────────────

class AudioMixer(nn.Module):
    """
    Multi-track Audio Mixer.

    Capabilities:
        • Multi-track mixing (voice, music, ambient, FX)
        • Per-track volume / pan (equal-power pan law)
        • Dynamics: compressor, limiter
        • Parametric EQ (peaking biquad sections)
        • Algorithmic reverb (Schroeder–Moorer)
    """

    def __init__(self, config: AudioMixerConfig):
        super().__init__()
        self.config = config
        self.tracks: Dict[str, AudioTrack] = {}

    # ── Track management ──────────────────────────────────────────────

    def add_track(self, track: AudioTrack) -> None:
        if len(self.tracks) >= self.config.max_tracks:
            raise RuntimeError(f"Maximum tracks ({self.config.max_tracks}) reached")
        self.tracks[track.name] = track

    def remove_track(self, name: str) -> None:
        self.tracks.pop(name, None)

    # ── Mix ────────────────────────────────────────────────────────────

    def mix(self) -> torch.Tensor:
        """Mix all active tracks → stereo output [2, samples]."""
        if not self.tracks:
            raise ValueError("No tracks to mix")

        # Determine output length (longest track)
        max_len = max(t.audio_data.shape[-1] for t in self.tracks.values())
        out = torch.zeros(2, max_len, device=self._device())

        # Solo logic: if any track is soloed, mute all non-soloed
        any_solo = any(t.solo for t in self.tracks.values())

        for track in self.tracks.values():
            if track.mute:
                continue
            if any_solo and not track.solo:
                continue

            # Ensure stereo
            audio = track.audio_data
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            if audio.shape[0] == 1:
                audio = audio.expand(2, -1)

            # Pad / trim to output length
            if audio.shape[-1] < max_len:
                audio = F.pad(audio, (0, max_len - audio.shape[-1]))
            else:
                audio = audio[:, :max_len]

            # EQ
            if self.config.enable_eq:
                audio = self._default_eq(audio, track.track_type)

            # Compressor
            if self.config.enable_compression:
                audio = self.apply_compression(
                    audio, threshold=-18.0, ratio=4.0,
                    attack_ms=10.0, release_ms=100.0,
                )

            # Reverb (ambient & fx get some)
            if self.config.enable_reverb and track.track_type in ("ambient", "fx"):
                audio = self.apply_reverb(audio, room_size=0.6)

            # Volume
            audio = audio * track.volume

            # Pan (equal-power)
            l_gain, r_gain = _equal_power_pan(track.pan)
            out[0] += audio[0] * l_gain
            out[1] += audio[1] * r_gain

        # Master limiter
        if self.config.enable_limiting:
            out = self._hard_limiter(out, ceiling_db=-1.0)

        return out

    # ── Compressor ─────────────────────────────────────────────────────

    def apply_compression(
        self,
        audio: torch.Tensor,
        threshold: float = -20.0,   # dB
        ratio: float = 4.0,
        attack_ms: float = 10.0,
        release_ms: float = 100.0,
    ) -> torch.Tensor:
        """Feed-forward dynamic-range compressor with envelope follower."""
        sr = self.config.sample_rate
        attack_coeff = math.exp(-1.0 / (attack_ms * 0.001 * sr))
        release_coeff = math.exp(-1.0 / (release_ms * 0.001 * sr))
        threshold_lin = _db_to_linear(threshold)

        C, N = audio.shape
        envelope = torch.zeros(C, device=audio.device)
        out = torch.zeros_like(audio)

        for n in range(N):
            sample = audio[:, n]
            level = sample.abs()
            # Envelope follower (peak)
            coeff = torch.where(level > envelope, attack_coeff, release_coeff)
            envelope = coeff * envelope + (1.0 - coeff) * level

            # Gain computer (soft-knee)
            over = (envelope / (threshold_lin + 1e-8)).clamp(min=1.0)
            gain_reduction = over ** (1.0 / ratio - 1.0)
            out[:, n] = sample * gain_reduction

        return out

    # ── EQ ─────────────────────────────────────────────────────────────

    def apply_eq(
        self,
        audio: torch.Tensor,
        bands: List[Tuple[float, float, float]],
    ) -> torch.Tensor:
        """Apply parametric EQ.

        Args:
            audio: [C, N]
            bands: list of (freq_hz, gain_db, Q) tuples
        """
        for freq, gain, q in bands:
            b, a = _biquad_peaking(freq, gain, q, self.config.sample_rate)
            audio = _apply_biquad(audio, b, a)
        return audio

    def _default_eq(self, audio: torch.Tensor, track_type: str) -> torch.Tensor:
        """Apply sensible default EQ per track type."""
        presets = {
            "voice": [(300.0, -3.0, 0.7), (3000.0, 3.0, 1.0), (8000.0, 2.0, 0.7)],
            "music": [(80.0, 2.0, 0.7), (5000.0, 1.0, 1.0)],
            "ambient": [(200.0, -2.0, 0.7), (6000.0, -1.0, 0.7)],
            "fx": [(1000.0, 2.0, 1.0)],
        }
        bands = presets.get(track_type, [])
        if bands:
            return self.apply_eq(audio, bands)
        return audio

    # ── Reverb ─────────────────────────────────────────────────────────

    def apply_reverb(
        self, audio: torch.Tensor, room_size: float = 0.5,
        wet_mix: float = 0.3,
    ) -> torch.Tensor:
        """Schroeder–Moorer algorithmic reverb (4 comb + 2 allpass).

        Args:
            audio:     [C, N]
            room_size: 0.0–1.0
            wet_mix:   dry/wet ratio
        """
        sr = self.config.sample_rate
        # Delay lengths in samples (prime-ish spacing)
        base = int(sr * 0.03)
        comb_delays = [base, int(base * 1.13), int(base * 1.27), int(base * 1.41)]
        ap_delays = [int(sr * 0.005), int(sr * 0.0017)]
        feedback = 0.3 + 0.5 * room_size

        C, N = audio.shape
        wet = torch.zeros_like(audio)

        # Parallel comb filters
        for delay in comb_delays:
            buf = torch.zeros(C, N + delay, device=audio.device, dtype=audio.dtype)
            for n in range(N):
                buf[:, n + delay] = audio[:, n] + feedback * buf[:, n]
            wet += buf[:, delay: delay + N]
        wet /= len(comb_delays)

        # Series allpass filters
        for delay in ap_delays:
            buf_in = wet.clone()
            buf_out = torch.zeros_like(wet)
            for n in range(delay, N):
                buf_out[:, n] = -feedback * buf_in[:, n] + buf_in[:, n - delay] + feedback * buf_out[:, n - delay]
            wet = buf_out

        return (1.0 - wet_mix) * audio + wet_mix * wet

    # ── Limiter ────────────────────────────────────────────────────────

    @staticmethod
    def _hard_limiter(audio: torch.Tensor, ceiling_db: float = -1.0) -> torch.Tensor:
        ceiling = _db_to_linear(ceiling_db)
        return audio.clamp(-ceiling, ceiling)

    # ── Helpers ────────────────────────────────────────────────────────

    def _device(self):
        for t in self.tracks.values():
            return t.audio_data.device
        return torch.device("cpu")


# ───────────────────────────────────────────────────────────────────────────
# Spatial Audio
# ───────────────────────────────────────────────────────────────────────────

class SpatialAudio(nn.Module):
    """Spatial audio processor (stereo, 5.1, binaural)."""

    def __init__(self, config: AudioMixerConfig):
        super().__init__()
        self.config = config

    def to_stereo(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert any channel layout to stereo [2, N]."""
        if audio.dim() == 1:
            return audio.unsqueeze(0).expand(2, -1)
        C = audio.shape[0]
        if C == 1:
            return audio.expand(2, -1)
        if C == 2:
            return audio
        if C == 6:
            # 5.1 → stereo downmix  (ITU-R BS.775)
            L, R, C_ch, LFE, Ls, Rs = audio[0], audio[1], audio[2], audio[3], audio[4], audio[5]
            left = L + 0.707 * C_ch + 0.707 * Ls
            right = R + 0.707 * C_ch + 0.707 * Rs
            return torch.stack([left, right])
        # Generic: take first two channels
        return audio[:2]

    def to_5_1(self, audio: torch.Tensor) -> torch.Tensor:
        """Upmix stereo → 5.1 surround [6, N].

        Channels: L, R, C, LFE, Ls, Rs
        """
        stereo = self.to_stereo(audio)
        L, R = stereo[0], stereo[1]
        C_ch = (L + R) * 0.5
        LFE = F.avg_pool1d(
            ((L + R) * 0.5).unsqueeze(0).unsqueeze(0),
            kernel_size=48, stride=1, padding=24,
        ).squeeze()[:L.shape[0]]
        # Decorrelate surrounds with comb filter
        sr = self.config.sample_rate
        delay = int(sr * 0.02)  # 20 ms
        Ls = torch.zeros_like(L)
        Rs = torch.zeros_like(R)
        Ls[delay:] = L[:-delay] * 0.5
        Rs[delay:] = R[:-delay] * 0.5
        return torch.stack([L, R, C_ch, LFE, Ls, Rs])

    def to_binaural(
        self, audio: torch.Tensor, azimuth_deg: float = 0.0,
    ) -> torch.Tensor:
        """Simple binaural panning using ITD + ILD (no HRTF convolution).

        Args:
            audio:        mono or stereo [C, N]
            azimuth_deg:  source angle (−90 left … +90 right)
        Returns:
            binaural: [2, N]
        """
        stereo = self.to_stereo(audio)
        L, R = stereo[0], stereo[1]
        sr = self.config.sample_rate

        # Interaural Time Difference (~0.65 ms max at ±90°)
        max_itd_samples = int(0.00065 * sr)
        itd = int(max_itd_samples * math.sin(math.radians(azimuth_deg)))

        # Interaural Level Difference (up to ~8 dB)
        ild_db = 8.0 * math.sin(math.radians(azimuth_deg))
        l_gain = _db_to_linear(-ild_db / 2.0)
        r_gain = _db_to_linear(ild_db / 2.0)

        # Apply ITD via delay
        N = L.shape[0]
        out_L = torch.zeros(N, device=audio.device, dtype=audio.dtype)
        out_R = torch.zeros(N, device=audio.device, dtype=audio.dtype)

        if itd >= 0:
            # Sound from right: delay left ear
            out_L[itd:] = L[:N - itd] * l_gain
            out_R[:] = R * r_gain
        else:
            out_L[:] = L * l_gain
            d = -itd
            out_R[d:] = R[:N - d] * r_gain

        return torch.stack([out_L, out_R])
