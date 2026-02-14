"""
AIPROD Camera Control
======================

Advanced cinematic camera system for diffusion video generation:

- **ControlNet camera conditioning:**  Injects camera parameters as
  conditioning signals into the transformer backbone (pan, tilt, zoom,
  dolly, crane, steadicam).
- **Parametric trajectories:**  Bézier curves for smooth camera movements
  with keyframe interpolation.
- **Cinematic templates:**  Pre-built camera presets (tracking shot,
  dolly zoom / Vertigo, orbit, establishing shot, champ-contrechamp).
- **Camera shake simulation:**  Handheld, action cam, stabilised modes.
- **Integration point:**  Outputs ``CameraConditioningSignal`` consumed
  by the transformer's cross-attention layers.

Requires: numpy (trajectory math). torch optional for conditioning tensors.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# 6-DOF camera state
# ---------------------------------------------------------------------------


@dataclass
class CameraState:
    """
    6-DOF camera state at a single point in time.

    Position (x, y, z) and rotation (yaw, pitch, roll) in world space.
    Plus lens parameters (fov, focus_distance).
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0  # degrees, horizontal rotation
    pitch: float = 0.0  # degrees, vertical rotation
    roll: float = 0.0  # degrees, roll/tilt
    fov: float = 50.0  # degrees, field of view
    focus_distance: float = 5.0  # metres
    timestamp: float = 0.0  # normalised 0-1 within clip

    def to_vector(self) -> List[float]:
        """8-dimensional vector for conditioning."""
        return [self.x, self.y, self.z, self.yaw, self.pitch, self.roll, self.fov, self.focus_distance]

    @classmethod
    def lerp(cls, a: "CameraState", b: "CameraState", t: float) -> "CameraState":
        """Linear interpolation between two states."""
        inv = 1.0 - t
        return cls(
            x=a.x * inv + b.x * t,
            y=a.y * inv + b.y * t,
            z=a.z * inv + b.z * t,
            yaw=a.yaw * inv + b.yaw * t,
            pitch=a.pitch * inv + b.pitch * t,
            roll=a.roll * inv + b.roll * t,
            fov=a.fov * inv + b.fov * t,
            focus_distance=a.focus_distance * inv + b.focus_distance * t,
            timestamp=t,
        )


# ---------------------------------------------------------------------------
# Bézier trajectory
# ---------------------------------------------------------------------------


@dataclass
class BezierControlPoint:
    """A control point for a Bézier curve."""

    state: CameraState
    weight: float = 1.0  # for rational Bézier


class BezierTrajectory:
    """
    Parametric Bézier camera trajectory.

    Supports cubic (4 control points) and higher-order curves.
    Evaluates position + rotation at any t ∈ [0, 1].
    """

    def __init__(self, control_points: Optional[List[BezierControlPoint]] = None):
        self._points = control_points or []

    def add_point(self, state: CameraState, weight: float = 1.0) -> None:
        self._points.append(BezierControlPoint(state=state, weight=weight))

    @property
    def order(self) -> int:
        return max(0, len(self._points) - 1)

    def evaluate(self, t: float) -> CameraState:
        """
        Evaluate the Bézier curve at parameter t ∈ [0, 1].

        Uses De Casteljau's algorithm for numerical stability.
        """
        n = len(self._points)
        if n == 0:
            return CameraState(timestamp=t)
        if n == 1:
            s = self._points[0].state
            return CameraState(**{**s.__dict__, "timestamp": t})

        # De Casteljau
        vectors = [p.state.to_vector() for p in self._points]
        while len(vectors) > 1:
            new_vectors = []
            for i in range(len(vectors) - 1):
                interp = [a * (1 - t) + b * t for a, b in zip(vectors[i], vectors[i + 1])]
                new_vectors.append(interp)
            vectors = new_vectors

        v = vectors[0]
        return CameraState(
            x=v[0], y=v[1], z=v[2],
            yaw=v[3], pitch=v[4], roll=v[5],
            fov=v[6], focus_distance=v[7],
            timestamp=t,
        )

    def sample(self, num_frames: int) -> List[CameraState]:
        """Sample the trajectory at evenly spaced intervals."""
        if num_frames <= 1:
            return [self.evaluate(0.0)]
        return [self.evaluate(i / (num_frames - 1)) for i in range(num_frames)]


# ---------------------------------------------------------------------------
# Camera shake simulation
# ---------------------------------------------------------------------------


class ShakeMode(str, Enum):
    NONE = "none"
    HANDHELD = "handheld"
    ACTION_CAM = "action_cam"
    STABILISED = "stabilised"


@dataclass
class ShakeConfig:
    mode: ShakeMode = ShakeMode.NONE
    intensity: float = 1.0  # 0-3 multiplier
    frequency_hz: float = 8.0  # oscillation frequency


class CameraShake:
    """
    Procedural camera shake generator.

    Adds noise to yaw/pitch/roll based on mode and intensity.
    Uses sum-of-sines for natural-looking handheld motion.
    """

    def __init__(self, config: Optional[ShakeConfig] = None):
        self._config = config or ShakeConfig()

    def apply(self, states: List[CameraState], fps: float = 24.0) -> List[CameraState]:
        """Apply shake to a list of camera states."""
        if self._config.mode == ShakeMode.NONE:
            return states

        result: List[CameraState] = []
        for i, s in enumerate(states):
            t = i / fps
            dy, dp, dr = self._shake_at(t)
            result.append(CameraState(
                x=s.x,
                y=s.y,
                z=s.z,
                yaw=s.yaw + dy,
                pitch=s.pitch + dp,
                roll=s.roll + dr,
                fov=s.fov,
                focus_distance=s.focus_distance,
                timestamp=s.timestamp,
            ))
        return result

    def _shake_at(self, t: float) -> Tuple[float, float, float]:
        """Returns (yaw_delta, pitch_delta, roll_delta) at time t."""
        f = self._config.frequency_hz
        intensity = self._config.intensity

        if self._config.mode == ShakeMode.HANDHELD:
            yaw = intensity * 0.5 * (
                math.sin(2 * math.pi * f * t)
                + 0.5 * math.sin(2 * math.pi * f * 1.7 * t + 0.3)
                + 0.25 * math.sin(2 * math.pi * f * 3.1 * t + 1.2)
            )
            pitch = intensity * 0.3 * (
                math.sin(2 * math.pi * f * 0.9 * t + 0.7)
                + 0.4 * math.sin(2 * math.pi * f * 2.3 * t + 2.1)
            )
            roll = intensity * 0.1 * math.sin(2 * math.pi * f * 0.5 * t + 1.5)
        elif self._config.mode == ShakeMode.ACTION_CAM:
            yaw = intensity * 1.0 * math.sin(2 * math.pi * f * t) + intensity * 0.8 * math.sin(
                2 * math.pi * f * 2.5 * t + 0.5
            )
            pitch = intensity * 0.7 * math.sin(2 * math.pi * f * 1.3 * t + 1.0)
            roll = intensity * 0.4 * math.sin(2 * math.pi * f * 0.7 * t + 2.0)
        elif self._config.mode == ShakeMode.STABILISED:
            yaw = intensity * 0.05 * math.sin(2 * math.pi * f * 0.3 * t)
            pitch = intensity * 0.03 * math.sin(2 * math.pi * f * 0.4 * t + 0.5)
            roll = intensity * 0.01 * math.sin(2 * math.pi * f * 0.2 * t + 1.0)
        else:
            yaw = pitch = roll = 0.0

        return yaw, pitch, roll


# ---------------------------------------------------------------------------
# Cinematic templates
# ---------------------------------------------------------------------------


class CinematicTemplate(str, Enum):
    """Pre-built camera movement templates."""

    ESTABLISHING = "establishing"  # wide → slow dolly in
    TRACKING_LATERAL = "tracking_lateral"  # side tracking (steadicam)
    DOLLY_ZOOM = "dolly_zoom"  # Vertigo effect (dolly out + zoom in)
    ORBIT = "orbit"  # 360° orbit around subject
    CRANE_REVEAL = "crane_reveal"  # crane up to reveal landscape
    PUSH_IN = "push_in"  # slow dolly in to subject
    PULL_OUT = "pull_out"  # dolly out from subject
    SHOT_REVERSE_SHOT = "shot_reverse_shot"  # champ-contrechamp
    STATIC = "static"  # locked tripod
    HANDHELD_WALK = "handheld_walk"  # following character


def build_template(
    template: CinematicTemplate,
    num_frames: int = 72,
    duration_sec: float = 3.0,
    **kwargs: Any,
) -> List[CameraState]:
    """
    Generate a camera trajectory from a cinematic template.

    Returns a list of CameraState for each frame.
    """
    trajectory = BezierTrajectory()

    if template == CinematicTemplate.ESTABLISHING:
        trajectory.add_point(CameraState(x=0, y=2, z=-10, fov=70, pitch=-5))
        trajectory.add_point(CameraState(x=0, y=1.8, z=-7, fov=60, pitch=-3))
        trajectory.add_point(CameraState(x=0, y=1.6, z=-5, fov=50, pitch=0))

    elif template == CinematicTemplate.TRACKING_LATERAL:
        distance = kwargs.get("distance", 5.0)
        trajectory.add_point(CameraState(x=-distance, y=1.5, z=0, yaw=20))
        trajectory.add_point(CameraState(x=-distance / 2, y=1.5, z=0, yaw=10))
        trajectory.add_point(CameraState(x=distance / 2, y=1.5, z=0, yaw=-10))
        trajectory.add_point(CameraState(x=distance, y=1.5, z=0, yaw=-20))

    elif template == CinematicTemplate.DOLLY_ZOOM:
        # Dolly out while zooming in (Vertigo effect)
        trajectory.add_point(CameraState(z=-3, fov=35))
        trajectory.add_point(CameraState(z=-4, fov=45))
        trajectory.add_point(CameraState(z=-6, fov=60))
        trajectory.add_point(CameraState(z=-8, fov=75))

    elif template == CinematicTemplate.ORBIT:
        radius = kwargs.get("radius", 5.0)
        n_points = 8
        for i in range(n_points + 1):
            angle = 2 * math.pi * i / n_points
            trajectory.add_point(CameraState(
                x=radius * math.sin(angle),
                z=-radius * math.cos(angle),
                y=1.5,
                yaw=math.degrees(angle),
            ))

    elif template == CinematicTemplate.CRANE_REVEAL:
        trajectory.add_point(CameraState(y=0.5, z=-2, pitch=30, fov=45))
        trajectory.add_point(CameraState(y=3.0, z=-3, pitch=10, fov=55))
        trajectory.add_point(CameraState(y=8.0, z=-5, pitch=-10, fov=70))

    elif template == CinematicTemplate.PUSH_IN:
        trajectory.add_point(CameraState(z=-8, fov=50))
        trajectory.add_point(CameraState(z=-5, fov=45))
        trajectory.add_point(CameraState(z=-3, fov=40))

    elif template == CinematicTemplate.PULL_OUT:
        trajectory.add_point(CameraState(z=-2, fov=35))
        trajectory.add_point(CameraState(z=-5, fov=50))
        trajectory.add_point(CameraState(z=-10, fov=65))

    elif template == CinematicTemplate.SHOT_REVERSE_SHOT:
        # Two alternating angles
        half = num_frames // 2
        states: List[CameraState] = []
        for i in range(half):
            t = i / max(half - 1, 1)
            states.append(CameraState(x=-1, z=-3, yaw=15, pitch=0, fov=50, timestamp=t * 0.5))
        for i in range(num_frames - half):
            t = i / max(num_frames - half - 1, 1)
            states.append(CameraState(x=1, z=-3, yaw=-15, pitch=0, fov=50, timestamp=0.5 + t * 0.5))
        return states

    elif template == CinematicTemplate.HANDHELD_WALK:
        trajectory.add_point(CameraState(x=0, y=1.6, z=0, fov=50))
        trajectory.add_point(CameraState(x=0, y=1.6, z=3, fov=50))
        trajectory.add_point(CameraState(x=0, y=1.6, z=6, fov=50))
        trajectory.add_point(CameraState(x=0, y=1.6, z=9, fov=50))
        states = trajectory.sample(num_frames)
        shake = CameraShake(ShakeConfig(mode=ShakeMode.HANDHELD, intensity=0.8))
        return shake.apply(states, fps=num_frames / duration_sec)

    else:  # STATIC
        return [CameraState(y=1.5, z=-5, fov=50, timestamp=i / max(num_frames - 1, 1))
                for i in range(num_frames)]

    return trajectory.sample(num_frames)


# ---------------------------------------------------------------------------
# Camera conditioning signal (for transformer injection)
# ---------------------------------------------------------------------------


@dataclass
class CameraConditioningSignal:
    """
    Camera conditioning signal consumed by the diffusion transformer.

    Contains per-frame 8-D vectors (x, y, z, yaw, pitch, roll, fov, focus)
    optionally converted to torch tensors for cross-attention injection.
    """

    signal_id: str = ""
    states: List[CameraState] = field(default_factory=list)
    template: Optional[CinematicTemplate] = None
    shake_mode: ShakeMode = ShakeMode.NONE

    def __post_init__(self):
        if not self.signal_id:
            self.signal_id = str(uuid.uuid4())[:8]

    @property
    def num_frames(self) -> int:
        return len(self.states)

    def to_matrix(self) -> List[List[float]]:
        """Convert to (num_frames, 8) matrix."""
        return [s.to_vector() for s in self.states]

    def to_tensor(self) -> Any:
        """Convert to torch tensor (num_frames, 8) if available."""
        mat = self.to_matrix()
        if HAS_TORCH:
            return torch.tensor(mat, dtype=torch.float32)
        if HAS_NUMPY:
            return np.array(mat, dtype=np.float32)
        return mat


# ---------------------------------------------------------------------------
# Camera Controller (orchestrator)
# ---------------------------------------------------------------------------


class CameraController:
    """
    High-level camera control API.

    Generates ``CameraConditioningSignal`` from templates, custom
    trajectories, or keyframes.  Applies optional camera shake.

    Usage:
        ctrl = CameraController()
        signal = ctrl.from_template(CinematicTemplate.DOLLY_ZOOM, num_frames=72)
        tensor = signal.to_tensor()  # inject into transformer
    """

    def from_template(
        self,
        template: CinematicTemplate,
        num_frames: int = 72,
        duration_sec: float = 3.0,
        shake: Optional[ShakeConfig] = None,
        **kwargs: Any,
    ) -> CameraConditioningSignal:
        """Generate conditioning from a cinematic template."""
        states = build_template(template, num_frames, duration_sec, **kwargs)

        if shake and shake.mode != ShakeMode.NONE:
            shaker = CameraShake(shake)
            states = shaker.apply(states, fps=num_frames / duration_sec)

        return CameraConditioningSignal(
            states=states,
            template=template,
            shake_mode=shake.mode if shake else ShakeMode.NONE,
        )

    def from_keyframes(
        self,
        keyframes: List[CameraState],
        num_frames: int = 72,
        shake: Optional[ShakeConfig] = None,
    ) -> CameraConditioningSignal:
        """Generate conditioning from user keyframes (Bézier interpolated)."""
        traj = BezierTrajectory()
        for kf in keyframes:
            traj.add_point(kf)
        states = traj.sample(num_frames)

        if shake and shake.mode != ShakeMode.NONE:
            shaker = CameraShake(shake)
            states = shaker.apply(states, fps=24.0)

        return CameraConditioningSignal(states=states)

    def from_scenarist_shot(
        self,
        shot_type: str,
        camera_move: str,
        duration_sec: float = 3.0,
        fps: float = 24.0,
        shake: Optional[ShakeConfig] = None,
    ) -> CameraConditioningSignal:
        """
        Map scenarist shot description to camera conditioning.

        Bridges the AI Scenarist output to camera control signals.
        """
        num_frames = max(1, int(duration_sec * fps))

        # Map camera_move string to template
        move_map: Dict[str, CinematicTemplate] = {
            "static": CinematicTemplate.STATIC,
            "pan_left": CinematicTemplate.TRACKING_LATERAL,
            "pan_right": CinematicTemplate.TRACKING_LATERAL,
            "dolly_in": CinematicTemplate.PUSH_IN,
            "dolly_out": CinematicTemplate.PULL_OUT,
            "tracking": CinematicTemplate.TRACKING_LATERAL,
            "crane_up": CinematicTemplate.CRANE_REVEAL,
            "crane_down": CinematicTemplate.CRANE_REVEAL,
            "steadicam": CinematicTemplate.HANDHELD_WALK,
            "handheld": CinematicTemplate.HANDHELD_WALK,
            "zoom_in": CinematicTemplate.PUSH_IN,
            "zoom_out": CinematicTemplate.PULL_OUT,
        }

        template = move_map.get(camera_move, CinematicTemplate.STATIC)

        # Default shake for handheld
        if camera_move in ("handheld", "steadicam") and shake is None:
            mode = ShakeMode.HANDHELD if camera_move == "handheld" else ShakeMode.STABILISED
            shake = ShakeConfig(mode=mode, intensity=0.6)

        return self.from_template(template, num_frames, duration_sec, shake)

    @staticmethod
    def list_templates() -> List[str]:
        return [t.value for t in CinematicTemplate]
