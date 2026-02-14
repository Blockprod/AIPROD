"""Multi-Format Video Export Engine

Exports video to multiple formats via FFmpeg subprocess:
    H.264/H.265 (.mp4, .mkv) · ProRes (.mov) · DNxHR (.mxf)
    VP9/AV1 (.webm) · EXR/DPX image sequences

Pipeline:
    tensor frames → raw pipe → FFmpeg → container
    tensor audio  → raw PCM   → FFmpeg → muxed output
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional
import os
import shutil
import struct
import subprocess
import tempfile

import torch


# ───────────────────────────────────────────────────────────────────────────
# Enums & Data classes
# ───────────────────────────────────────────────────────────────────────────

class VideoCodec(Enum):
    H264 = "h264"
    H265 = "hevc"
    PRORES_422 = "prores_422"
    PRORES_4444 = "prores_4444"
    DNxHD = "dnxhd"
    DNxHR = "dnxhr"
    VP9 = "vp9"
    AV1 = "av1"


class AudioCodec(Enum):
    AAC = "aac"
    OPUS = "opus"
    FLAC = "flac"
    PCM = "pcm"
    DOLBY_DIGITAL = "ac3"
    DOLBY_DIGITAL_PLUS = "eac3"
    ATMOS = "eac3_joc"


@dataclass
class ExportProfile:
    """Video export profile / preset."""
    name: str
    video_codec: VideoCodec
    audio_codec: AudioCodec
    resolution: tuple = (1920, 1080)
    fps: int = 30
    bitrate_video: int = 20_000   # kbps
    bitrate_audio: int = 256      # kbps
    preset: str = "slower"
    color_space: str = "rec709"
    color_range: str = "full"
    hdr_enabled: bool = False
    hdr_format: Optional[str] = None
    container: str = "mp4"


@dataclass
class ExportConfig:
    """Global export configuration."""
    profiles: Optional[Dict[str, ExportProfile]] = None
    default_profile: str = "web_mp4"
    temp_directory: str = ""
    use_gpu_encoding: bool = True
    num_workers: int = 4


# ───────────────────────────────────────────────────────────────────────────
# FFmpeg codec / option mapping
# ───────────────────────────────────────────────────────────────────────────

_FFMPEG_VCODEC = {
    VideoCodec.H264:        "libx264",
    VideoCodec.H265:        "libx265",
    VideoCodec.PRORES_422:  "prores_ks",
    VideoCodec.PRORES_4444: "prores_ks",
    VideoCodec.DNxHD:       "dnxhd",
    VideoCodec.DNxHR:       "dnxhd",
    VideoCodec.VP9:         "libvpx-vp9",
    VideoCodec.AV1:         "libaom-av1",
}

_FFMPEG_ACODEC = {
    AudioCodec.AAC:          "aac",
    AudioCodec.OPUS:         "libopus",
    AudioCodec.FLAC:         "flac",
    AudioCodec.PCM:          "pcm_s16le",
    AudioCodec.DOLBY_DIGITAL: "ac3",
    AudioCodec.DOLBY_DIGITAL_PLUS: "eac3",
    AudioCodec.ATMOS:        "eac3",
}


def _find_ffmpeg() -> str:
    """Return path to FFmpeg binary (raises if missing)."""
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError(
            "FFmpeg not found on PATH.  Install FFmpeg to enable export."
        )
    return path


# ───────────────────────────────────────────────────────────────────────────
# Low-level encoders
# ───────────────────────────────────────────────────────────────────────────

class VideoEncoder:
    """Encodes a sequence of tensor frames via FFmpeg raw-pipe."""

    @staticmethod
    def encode_sequence(
        video: torch.Tensor,       # [3, H, W, T] float32 0-1
        output_path: str,
        codec: VideoCodec = VideoCodec.H264,
        fps: int = 30,
        bitrate_kbps: int = 20_000,
        preset: str = "slower",
        pix_fmt: str = "yuv420p",
    ) -> bool:
        ffmpeg = _find_ffmpeg()
        _, H, W, T = video.shape
        lib = _FFMPEG_VCODEC[codec]

        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", lib,
            "-pix_fmt", pix_fmt,
            "-b:v", f"{bitrate_kbps}k",
        ]

        # Codec-specific flags
        if codec in (VideoCodec.H264, VideoCodec.H265):
            cmd += ["-preset", preset]
        if codec == VideoCodec.PRORES_422:
            cmd += ["-profile:v", "2"]          # ProRes 422
        elif codec == VideoCodec.PRORES_4444:
            cmd += ["-profile:v", "4"]          # ProRes 4444

        cmd.append(output_path)

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # Feed frames as raw RGB24
        for t_idx in range(T):
            frame = video[:, :, :, t_idx]            # [3, H, W]
            frame_uint8 = (frame.clamp(0, 1) * 255).byte()
            frame_hwc = frame_uint8.permute(1, 2, 0).contiguous()
            proc.stdin.write(frame_hwc.numpy().tobytes())

        proc.stdin.close()
        proc.wait()
        return proc.returncode == 0


class AudioEncoder:
    """Encodes tensor audio via FFmpeg raw-pipe."""

    @staticmethod
    def encode(
        audio: torch.Tensor,       # [C, N] float32, range [-1, 1]
        output_path: str,
        codec: AudioCodec = AudioCodec.AAC,
        sample_rate: int = 48_000,
        bitrate_kbps: int = 256,
    ) -> bool:
        ffmpeg = _find_ffmpeg()
        C, N = audio.shape
        lib = _FFMPEG_ACODEC[codec]

        cmd = [
            ffmpeg, "-y",
            "-f", "f32le",
            "-ar", str(sample_rate),
            "-ac", str(C),
            "-i", "pipe:0",
            "-c:a", lib,
        ]
        if codec not in (AudioCodec.PCM, AudioCodec.FLAC):
            cmd += ["-b:a", f"{bitrate_kbps}k"]
        cmd.append(output_path)

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        # Interleave channels: [C, N] → [N, C] → bytes
        interleaved = audio.T.contiguous()  # [N, C]
        proc.stdin.write(interleaved.numpy().tobytes())

        proc.stdin.close()
        proc.wait()
        return proc.returncode == 0


class Muxer:
    """Multiplexes separate video and audio files into a container."""

    @staticmethod
    def mux(
        video_file: str, audio_file: str, output_file: str,
        container: str = "mp4",
    ) -> bool:
        ffmpeg = _find_ffmpeg()
        cmd = [
            ffmpeg, "-y",
            "-i", video_file,
            "-i", audio_file,
            "-c", "copy",
        ]
        if container == "mp4":
            cmd += ["-movflags", "+faststart"]
        cmd.append(output_file)
        return subprocess.call(cmd, stderr=subprocess.DEVNULL) == 0


# ───────────────────────────────────────────────────────────────────────────
# Export Engine
# ───────────────────────────────────────────────────────────────────────────

class ExportEngine:
    """Multi-format video export engine."""

    STANDARD_PROFILES: Dict[str, ExportProfile] = {
        "web_mp4": ExportProfile(
            name="web_mp4",
            video_codec=VideoCodec.H264,
            audio_codec=AudioCodec.AAC,
            resolution=(1920, 1080), fps=30,
            bitrate_video=8_000, preset="medium",
            container="mp4",
        ),
        "streaming_hq": ExportProfile(
            name="streaming_hq",
            video_codec=VideoCodec.H265,
            audio_codec=AudioCodec.AAC,
            resolution=(3840, 2160), fps=30,
            bitrate_video=25_000, preset="slower",
            container="mp4",
        ),
        "prores_editing": ExportProfile(
            name="prores_editing",
            video_codec=VideoCodec.PRORES_422,
            audio_codec=AudioCodec.PCM,
            resolution=(1920, 1080), fps=30,
            bitrate_video=500_000, preset="fast",
            container="mov",
        ),
        "dnxhr_mxf": ExportProfile(
            name="dnxhr_mxf",
            video_codec=VideoCodec.DNxHR,
            audio_codec=AudioCodec.PCM,
            resolution=(1920, 1080), fps=30,
            container="mxf",
        ),
        "web_av1": ExportProfile(
            name="web_av1",
            video_codec=VideoCodec.AV1,
            audio_codec=AudioCodec.OPUS,
            resolution=(1920, 1080), fps=30,
            bitrate_video=6_000, preset="slower",
            container="webm",
        ),
    }

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()
        if not self.config.profiles:
            self.config.profiles = dict(self.STANDARD_PROFILES)

    # ── Main export ────────────────────────────────────────────────────

    def export(
        self,
        video: torch.Tensor,                 # [3, H, W, T]
        audio: Optional[torch.Tensor],       # [C, N]
        output_path: str,
        profile: str = "web_mp4",
        callback_progress: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Export video (+ optional audio) to *output_path* using *profile*."""
        if profile not in self.config.profiles:
            raise ValueError(f"Unknown profile: {profile}")
        p = self.config.profiles[profile]

        tmp = self.config.temp_directory or tempfile.mkdtemp(prefix="aiprod_export_")
        os.makedirs(tmp, exist_ok=True)

        video_tmp = os.path.join(tmp, "video_raw.mkv")
        audio_tmp = os.path.join(tmp, "audio_raw.mka")

        # 1. Encode video
        if callback_progress:
            callback_progress(0.1)
        ok_v = VideoEncoder.encode_sequence(
            video, video_tmp,
            codec=p.video_codec,
            fps=p.fps,
            bitrate_kbps=p.bitrate_video,
            preset=p.preset,
        )
        if not ok_v:
            return False

        if callback_progress:
            callback_progress(0.6)

        # 2. Encode audio (if present)
        if audio is not None:
            ok_a = AudioEncoder.encode(
                audio, audio_tmp,
                codec=p.audio_codec,
                bitrate_kbps=p.bitrate_audio,
            )
            if not ok_a:
                return False
            if callback_progress:
                callback_progress(0.8)
            # 3. Mux
            ok = Muxer.mux(video_tmp, audio_tmp, output_path, p.container)
        else:
            # No audio: just move video
            shutil.move(video_tmp, output_path)
            ok = True

        if callback_progress:
            callback_progress(1.0)
        return ok

    # ── Image-sequence export ──────────────────────────────────────────

    def export_to_sequence(
        self,
        video: torch.Tensor,
        output_dir: str,
        sequence_format: str = "exr",
        bit_depth: int = 16,
    ) -> bool:
        """Export each frame as an individual image (EXR/DPX via FFmpeg)."""
        ffmpeg = _find_ffmpeg()
        os.makedirs(output_dir, exist_ok=True)
        _, H, W, T = video.shape

        ext = sequence_format.lower()
        pattern = os.path.join(output_dir, f"frame_%06d.{ext}")

        pix_fmt = "rgb48le" if bit_depth == 16 else "gbrpf32le"
        cmd = [
            ffmpeg, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}",
            "-r", "1",
            "-i", "pipe:0",
            "-pix_fmt", pix_fmt,
            pattern,
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for t_idx in range(T):
            frame = video[:, :, :, t_idx]
            frame_u8 = (frame.clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous()
            proc.stdin.write(frame_u8.numpy().tobytes())
        proc.stdin.close()
        proc.wait()
        return proc.returncode == 0

    # ── Profile management ─────────────────────────────────────────────

    def add_custom_profile(self, profile: ExportProfile) -> None:
        self.config.profiles[profile.name] = profile

    def list_profiles(self) -> List[str]:
        return list(self.config.profiles.keys())

    def get_profile_info(self, profile_name: str) -> Dict:
        profile = self.config.profiles.get(profile_name)
        if not profile:
            return {}
        return {
            "name": profile.name,
            "codec": profile.video_codec.value,
            "resolution": profile.resolution,
            "fps": profile.fps,
            "bitrate": f"{profile.bitrate_video} kbps",
            "container": profile.container,
            "color_space": profile.color_space,
            "hdr": profile.hdr_enabled,
        }
