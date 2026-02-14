"""Automatic Montage / Editing Engine

Generates editing timelines with cuts, transitions, and pacing.
Supports automatic scene assembly and narrative rhythm control.

Pipeline:
    scenario → PacingEngine → TimelineGenerator → TransitionsLib → export (EDL/FCPXML)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────
# Enums & data classes
# ───────────────────────────────────────────────────────────────────────────

class TransitionType(Enum):
    CUT = "cut"
    FADE = "fade"
    DISSOLVE = "dissolve"
    WIPE = "wipe"
    MATCH_CUT = "match_cut"


@dataclass
class TimelineClip:
    """Single clip in the timeline."""
    name: str
    start_sec: float
    duration_sec: float
    video_source: str
    audio_sources: List[str] = field(default_factory=list)
    transition_in: TransitionType = TransitionType.CUT
    transition_duration: float = 0.0
    volume: float = 1.0
    playback_speed: float = 1.0


@dataclass
class EditingConfig:
    """Editing engine configuration."""
    video_fps: int = 30
    audio_sample_rate: int = 48000

    # Transitions
    default_transition: TransitionType = TransitionType.DISSOLVE
    default_transition_duration: float = 0.5

    # Pacing
    enable_auto_pacing: bool = True
    emotion_aware_timing: bool = True

    # Export
    export_formats: List[str] = field(
        default_factory=lambda: ["mp4", "xml", "edl"],
    )


# ───────────────────────────────────────────────────────────────────────────
# Pacing Engine
# ───────────────────────────────────────────────────────────────────────────

# Emotion → (min_sec, max_sec) base shot duration range
_EMOTION_DURATIONS: Dict[str, Tuple[float, float]] = {
    "action": (0.5, 2.0),
    "suspense": (1.5, 4.0),
    "calm": (3.0, 8.0),
    "buildup": (1.0, 3.0),
    "dramatic": (2.0, 5.0),
    "comedy": (1.0, 3.0),
}


class PacingEngine:
    """Controls narrative rhythm and shot duration."""

    def __init__(self, config: EditingConfig):
        self.config = config

    def compute_optimal_duration(
        self,
        scene_emotion: str,
        action_intensity: float = 0.5,
        visual_complexity: float = 0.5,
    ) -> float:
        """Compute optimal shot duration (seconds).

        High action → shorter shots.  Calm → longer.
        Complex visuals → slightly longer for comprehension.
        """
        lo, hi = _EMOTION_DURATIONS.get(scene_emotion, (2.0, 5.0))
        # Intensity shortens
        base = hi - (hi - lo) * action_intensity
        # Complexity lengthens (up to +30 %)
        base *= 1.0 + 0.3 * visual_complexity
        return round(max(lo, min(hi * 1.3, base)), 2)

    def compute_pacing_curve(
        self, emotions: List[str], intensities: List[float],
    ) -> List[float]:
        """Return a list of optimal durations for each scene."""
        return [
            self.compute_optimal_duration(e, i)
            for e, i in zip(emotions, intensities)
        ]


# ───────────────────────────────────────────────────────────────────────────
# Transitions library
# ───────────────────────────────────────────────────────────────────────────

class TransitionsLib:
    """Library of video transition effects (tensor-based)."""

    @staticmethod
    def cross_fade(
        clip_a: torch.Tensor,   # [C, H, W, T_a]
        clip_b: torch.Tensor,   # [C, H, W, T_b]
        transition_frames: int,
    ) -> torch.Tensor:
        """Alpha cross-fade between trailing frames of A and leading frames of B.

        Returns combined tensor [C, H, W, T_a + T_b − transition_frames].
        """
        T_a = clip_a.shape[-1]
        T_b = clip_b.shape[-1]
        t = min(transition_frames, T_a, T_b)

        # Fade-out tail of A, fade-in head of B
        alpha = torch.linspace(0.0, 1.0, t, device=clip_a.device)
        alpha = alpha.reshape(1, 1, 1, t)

        blend = (1.0 - alpha) * clip_a[..., -t:] + alpha * clip_b[..., :t]

        parts = []
        if T_a > t:
            parts.append(clip_a[..., : T_a - t])
        parts.append(blend)
        if T_b > t:
            parts.append(clip_b[..., t:])

        return torch.cat(parts, dim=-1)

    @staticmethod
    def wipe(
        clip_a: torch.Tensor,
        clip_b: torch.Tensor,
        transition_frames: int,
        direction: str = "left_to_right",
    ) -> torch.Tensor:
        """Directional wipe transition.

        Supported directions: left_to_right, right_to_left,
                              top_to_bottom, bottom_to_top.
        """
        C, H, W, T_a = clip_a.shape
        T_b = clip_b.shape[-1]
        t = min(transition_frames, T_a, T_b)
        result_frames = []

        for i in range(t):
            progress = (i + 1) / t
            mask = torch.zeros(1, H, W, 1, device=clip_a.device)

            if direction == "left_to_right":
                boundary = int(W * progress)
                mask[:, :, :boundary, :] = 1.0
            elif direction == "right_to_left":
                boundary = int(W * (1.0 - progress))
                mask[:, :, boundary:, :] = 1.0
            elif direction == "top_to_bottom":
                boundary = int(H * progress)
                mask[:, :boundary, :, :] = 1.0
            elif direction == "bottom_to_top":
                boundary = int(H * (1.0 - progress))
                mask[:, boundary:, :, :] = 1.0
            else:
                mask[:, :, : int(W * progress), :] = 1.0

            frame_a = clip_a[..., T_a - t + i : T_a - t + i + 1]
            frame_b = clip_b[..., i : i + 1]
            blended = (1.0 - mask) * frame_a + mask * frame_b
            result_frames.append(blended)

        parts = []
        if T_a > t:
            parts.append(clip_a[..., : T_a - t])
        parts.append(torch.cat(result_frames, dim=-1))
        if T_b > t:
            parts.append(clip_b[..., t:])

        return torch.cat(parts, dim=-1)

    @staticmethod
    def match_cut(
        clip_a: torch.Tensor,
        clip_b: torch.Tensor,
        match_frame_a: int = -1,
        match_frame_b: int = 0,
    ) -> torch.Tensor:
        """Match cut: instant cut with spatial continuity alignment.

        Applies a short (3-frame) micro-dissolve for perceived smoothness.
        """
        if match_frame_a < 0:
            match_frame_a = clip_a.shape[-1] + match_frame_a
        # Micro-dissolve (3 frames)
        a_end = clip_a[..., : match_frame_a + 1]
        b_start = clip_b[..., match_frame_b:]
        return TransitionsLib.cross_fade(a_end, b_start, transition_frames=3)


# ───────────────────────────────────────────────────────────────────────────
# Timeline Generator
# ───────────────────────────────────────────────────────────────────────────

class TimelineGenerator:
    """Generates editing timeline from a scenario / script."""

    def __init__(self, config: EditingConfig):
        self.config = config
        self.clips: List[TimelineClip] = []
        self.timeline_duration: float = 0.0
        self._pacing = PacingEngine(config)

    # ── Build from scenario ────────────────────────────────────────────

    def from_scenario(self, scenario: dict) -> "TimelineGenerator":
        """Build timeline from a scenario dict.

        Expected format::
            {
                "scenes": [
                    {"id": "s1", "duration": 5, "emotion": "action",
                     "video_source": "path.mp4", "intensity": 0.8},
                    ...
                ],
                "style": "cinematic",
            }
        """
        scenes = scenario.get("scenes", [])
        cursor: float = 0.0

        for i, scene in enumerate(scenes):
            emotion = scene.get("emotion", "calm")
            intensity = scene.get("intensity", 0.5)
            complexity = scene.get("visual_complexity", 0.5)

            # Duration: explicit or auto-paced
            if self.config.enable_auto_pacing and "duration" not in scene:
                dur = self._pacing.compute_optimal_duration(
                    emotion, intensity, complexity,
                )
            else:
                dur = float(scene.get("duration", 3.0))

            # Transition logic
            if i == 0:
                trans = TransitionType.CUT
                trans_dur = 0.0
            else:
                trans = self.config.default_transition
                trans_dur = self.config.default_transition_duration
                # Action → hard cuts
                if emotion == "action" and intensity > 0.7:
                    trans = TransitionType.CUT
                    trans_dur = 0.0

            self.add_clip(
                video_source=scene.get("video_source", scene.get("id", f"scene_{i}")),
                start_sec=cursor,
                duration_sec=dur,
                transition=trans,
                transition_duration=trans_dur,
            )
            # Advance cursor (subtract overlap for dissolves)
            cursor += dur - trans_dur

        return self

    # ── Add clip ───────────────────────────────────────────────────────

    def add_clip(
        self,
        video_source: str,
        start_sec: float,
        duration_sec: float,
        transition: TransitionType = TransitionType.DISSOLVE,
        transition_duration: float = 0.5,
    ) -> None:
        clip = TimelineClip(
            name=f"clip_{len(self.clips)}",
            start_sec=start_sec,
            duration_sec=duration_sec,
            video_source=video_source,
            transition_in=transition,
            transition_duration=transition_duration,
        )
        self.clips.append(clip)
        self.timeline_duration = max(
            self.timeline_duration, start_sec + duration_sec,
        )

    # ── Pacing analysis ────────────────────────────────────────────────

    def compute_pacing(self) -> dict:
        """Return pacing metadata for the current timeline."""
        if not self.clips:
            return {"cuts_per_minute": 0.0, "avg_shot_duration": 0.0}

        durations = [c.duration_sec for c in self.clips]
        total = self.timeline_duration or sum(durations)
        cpm = (len(self.clips) / total) * 60.0 if total > 0 else 0.0

        return {
            "num_clips": len(self.clips),
            "total_duration_sec": round(total, 2),
            "cuts_per_minute": round(cpm, 2),
            "avg_shot_duration": round(sum(durations) / len(durations), 2),
            "min_shot_duration": round(min(durations), 2),
            "max_shot_duration": round(max(durations), 2),
        }

    # ── Export: EDL ─────────────────────────────────────────────────────

    def export_edl(self, filepath: str) -> None:
        """Export timeline to CMX 3600 EDL format."""
        lines = ["TITLE: AIPROD Timeline", "FCM: NON-DROP FRAME", ""]
        fps = self.config.video_fps

        for i, clip in enumerate(self.clips):
            event = i + 1
            reel = clip.video_source[:8].ljust(8)
            track = "V" if not clip.audio_sources else "B"
            trans = "C" if clip.transition_in == TransitionType.CUT else "D"
            trans_dur = (
                f" {int(clip.transition_duration * fps):03d}"
                if trans == "D" else ""
            )

            src_in = self._tc(0.0, fps)
            src_out = self._tc(clip.duration_sec, fps)
            rec_in = self._tc(clip.start_sec, fps)
            rec_out = self._tc(clip.start_sec + clip.duration_sec, fps)

            lines.append(
                f"{event:03d}  {reel}  {track}    {trans}{trans_dur}   "
                f"{src_in} {src_out} {rec_in} {rec_out}"
            )
            lines.append(f"* FROM CLIP NAME: {clip.name}")
            lines.append("")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    # ── Export: FCPXML ──────────────────────────────────────────────────

    def export_fcpxml(self, filepath: str) -> None:
        """Export timeline to FCPXML v1.11 format."""
        fps = self.config.video_fps
        total_frames = int(self.timeline_duration * fps)

        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<!DOCTYPE fcpxml>',
            f'<fcpxml version="1.11">',
            '  <resources>',
        ]

        # Declare media resources
        for i, clip in enumerate(self.clips):
            dur_frames = int(clip.duration_sec * fps)
            lines.append(
                f'    <asset id="r{i+1}" name="{clip.video_source}" '
                f'duration="{dur_frames}/{fps}s" '
                f'hasVideo="1" hasAudio="1"/>'
            )

        lines += [
            '  </resources>',
            '  <library>',
            '    <event name="AIPROD Export">',
            f'      <project name="Timeline">',
            f'        <sequence duration="{total_frames}/{fps}s" '
            f'format="r0">',
            '          <spine>',
        ]

        for i, clip in enumerate(self.clips):
            offset_frames = int(clip.start_sec * fps)
            dur_frames = int(clip.duration_sec * fps)
            lines.append(
                f'            <asset-clip ref="r{i+1}" '
                f'offset="{offset_frames}/{fps}s" '
                f'duration="{dur_frames}/{fps}s" '
                f'name="{clip.name}"/>'
            )

        lines += [
            '          </spine>',
            '        </sequence>',
            '      </project>',
            '    </event>',
            '  </library>',
            '</fcpxml>',
        ]

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    # ── Export: AAF (simplified text representation) ────────────────────

    def export_aaf(self, filepath: str) -> None:
        """Export timeline to a simplified AAF-like text format.

        Full binary AAF requires the ``pyaaf2`` library; here we write a
        human-readable representation that captures the essential data.
        """
        lines = [
            "# AIPROD AAF Export (text representation)",
            f"# Total duration: {self.timeline_duration:.2f}s",
            f"# FPS: {self.config.video_fps}",
            "",
        ]
        for i, clip in enumerate(self.clips):
            lines.append(f"[Segment {i}]")
            lines.append(f"  Source   = {clip.video_source}")
            lines.append(f"  Start   = {clip.start_sec:.3f}s")
            lines.append(f"  Duration= {clip.duration_sec:.3f}s")
            lines.append(f"  Trans   = {clip.transition_in.value}")
            lines.append(f"  TransDur= {clip.transition_duration:.3f}s")
            lines.append("")

        with open(filepath, "w") as f:
            f.write("\n".join(lines))

    # ── Timecode helper ────────────────────────────────────────────────

    @staticmethod
    def _tc(seconds: float, fps: int) -> str:
        """Convert seconds → HH:MM:SS:FF timecode string."""
        total_frames = int(round(seconds * fps))
        ff = total_frames % fps
        s = (total_frames // fps) % 60
        m = (total_frames // (fps * 60)) % 60
        h = total_frames // (fps * 3600)
        return f"{h:02d}:{m:02d}:{s:02d}:{ff:02d}"
