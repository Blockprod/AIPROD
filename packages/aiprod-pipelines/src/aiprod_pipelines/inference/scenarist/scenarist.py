"""
AIPROD AI Scenarist
====================

Local LLM-based cinematic scene planner — replaces Gemini API dependency.

Capabilities:
- **Scene decomposition:**  Prompt → structured JSON scene list
  (shots, camera, lighting, dialogue, duration, emotion)
- **Local LLM backend:**  Llama / Mistral via HuggingFace transformers
  (no external API call needed)
- **Creative control:**  Style, genre, target audience, tone, pacing
- **Storyboard generation:**  Timeline with transitions, audio cues
- **Fallback:**  Rule-based decomposer when no GPU / model available

Requires: transformers, accelerate (optional for GPU inference)
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Scene data model
# ---------------------------------------------------------------------------


class ShotType(str, Enum):
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE_UP = "extreme_close_up"
    OVER_SHOULDER = "over_shoulder"
    POV = "pov"
    AERIAL = "aerial"
    LOW_ANGLE = "low_angle"
    HIGH_ANGLE = "high_angle"
    DUTCH_ANGLE = "dutch_angle"


class CameraMove(str, Enum):
    STATIC = "static"
    PAN_LEFT = "pan_left"
    PAN_RIGHT = "pan_right"
    TILT_UP = "tilt_up"
    TILT_DOWN = "tilt_down"
    DOLLY_IN = "dolly_in"
    DOLLY_OUT = "dolly_out"
    TRACKING = "tracking"
    CRANE_UP = "crane_up"
    CRANE_DOWN = "crane_down"
    STEADICAM = "steadicam"
    HANDHELD = "handheld"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"


class Transition(str, Enum):
    CUT = "cut"
    DISSOLVE = "dissolve"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    WIPE = "wipe"
    MATCH_CUT = "match_cut"
    SMASH_CUT = "smash_cut"
    J_CUT = "j_cut"
    L_CUT = "l_cut"


@dataclass
class SceneShot:
    """A single shot within a scene."""

    shot_id: str = ""
    description: str = ""
    shot_type: ShotType = ShotType.MEDIUM
    camera_move: CameraMove = CameraMove.STATIC
    duration_sec: float = 3.0
    dialogue: str = ""
    emotion: str = ""  # e.g. "tension", "joy", "melancholy"
    lighting: str = "natural"  # e.g. "golden_hour", "neon", "chiaroscuro"
    color_palette: str = ""  # e.g. "warm", "desaturated", "teal_orange"
    audio_cue: str = ""  # e.g. "ambient_rain", "dramatic_strings"
    transition_out: Transition = Transition.CUT
    prompt_override: str = ""  # optional direct prompt for this shot

    def __post_init__(self):
        if not self.shot_id:
            self.shot_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "description": self.description,
            "shot_type": self.shot_type.value,
            "camera_move": self.camera_move.value,
            "duration_sec": self.duration_sec,
            "dialogue": self.dialogue,
            "emotion": self.emotion,
            "lighting": self.lighting,
            "color_palette": self.color_palette,
            "audio_cue": self.audio_cue,
            "transition_out": self.transition_out.value,
            "prompt_override": self.prompt_override,
        }


@dataclass
class Scene:
    """A scene composed of multiple shots."""

    scene_id: str = ""
    title: str = ""
    location: str = ""
    time_of_day: str = "day"
    weather: str = ""
    shots: List[SceneShot] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        if not self.scene_id:
            self.scene_id = str(uuid.uuid4())[:8]

    @property
    def total_duration(self) -> float:
        return sum(s.duration_sec for s in self.shots)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "title": self.title,
            "location": self.location,
            "time_of_day": self.time_of_day,
            "weather": self.weather,
            "total_duration_sec": self.total_duration,
            "shots": [s.to_dict() for s in self.shots],
            "notes": self.notes,
        }


@dataclass
class Storyboard:
    """Complete storyboard — the output of the scenarist."""

    storyboard_id: str = ""
    title: str = ""
    genre: str = ""
    style: str = ""
    target_audience: str = ""
    tone: str = ""
    total_duration_sec: float = 0.0
    scenes: List[Scene] = field(default_factory=list)
    created_at: float = 0.0

    def __post_init__(self):
        if not self.storyboard_id:
            self.storyboard_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()

    def recalculate_duration(self) -> float:
        self.total_duration_sec = sum(s.total_duration for s in self.scenes)
        return self.total_duration_sec

    def to_dict(self) -> Dict[str, Any]:
        self.recalculate_duration()
        return {
            "storyboard_id": self.storyboard_id,
            "title": self.title,
            "genre": self.genre,
            "style": self.style,
            "target_audience": self.target_audience,
            "tone": self.tone,
            "total_duration_sec": self.total_duration_sec,
            "scenes": [s.to_dict() for s in self.scenes],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Creative direction config
# ---------------------------------------------------------------------------


@dataclass
class CreativeConfig:
    """Creative parameters for the scenarist."""

    genre: str = "cinematic"  # cinematic, documentary, commercial, music_video, narrative
    style: str = "realistic"  # realistic, stylized, noir, anime, fantasy
    tone: str = "neutral"  # dramatic, comedic, suspenseful, romantic, melancholic
    pacing: str = "moderate"  # fast, moderate, slow, variable
    target_audience: str = "general"
    target_duration_sec: float = 30.0
    max_scenes: int = 10
    preferred_shot_types: List[str] = field(default_factory=list)
    color_direction: str = ""  # e.g. "warm earth tones", "cold blue"
    music_style: str = ""  # e.g. "orchestral", "electronic", "ambient"


# ---------------------------------------------------------------------------
# Rule-based decomposer (fallback)
# ---------------------------------------------------------------------------


class RuleBasedDecomposer:
    """
    Simple rule-based scene decomposer.

    Splits prompt into scenes based on sentence structure,
    assigns shot types and camera moves heuristically.
    Used as fallback when no LLM is available.
    """

    # Mapping keywords → shot type
    SHOT_HINTS = {
        "landscape": ShotType.WIDE,
        "panorama": ShotType.WIDE,
        "city": ShotType.WIDE,
        "skyline": ShotType.WIDE,
        "face": ShotType.CLOSE_UP,
        "eye": ShotType.EXTREME_CLOSE_UP,
        "detail": ShotType.CLOSE_UP,
        "person": ShotType.MEDIUM,
        "walking": ShotType.MEDIUM,
        "running": ShotType.MEDIUM,
        "aerial": ShotType.AERIAL,
        "drone": ShotType.AERIAL,
        "bird": ShotType.AERIAL,
    }

    CAMERA_HINTS = {
        "follow": CameraMove.TRACKING,
        "chase": CameraMove.TRACKING,
        "reveal": CameraMove.DOLLY_OUT,
        "approach": CameraMove.DOLLY_IN,
        "look up": CameraMove.TILT_UP,
        "look down": CameraMove.TILT_DOWN,
        "fly": CameraMove.CRANE_UP,
        "sweep": CameraMove.PAN_RIGHT,
    }

    def decompose(self, prompt: str, config: Optional[CreativeConfig] = None) -> Storyboard:
        config = config or CreativeConfig()

        # Split into sentences
        sentences = re.split(r'[.!?;]+', prompt)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            sentences = [prompt]

        # Target duration per shot
        n_shots = max(1, min(len(sentences), config.max_scenes * 3))
        shot_dur = config.target_duration_sec / n_shots

        scenes: List[Scene] = []
        shots_in_scene: List[SceneShot] = []

        for i, sent in enumerate(sentences):
            shot_type = self._detect_shot_type(sent)
            camera = self._detect_camera(sent)

            shot = SceneShot(
                description=sent,
                shot_type=shot_type,
                camera_move=camera,
                duration_sec=round(max(2.0, min(shot_dur, 8.0)), 1),
                lighting=self._detect_lighting(sent),
                emotion=self._detect_emotion(sent, config.tone),
            )
            shots_in_scene.append(shot)

            # Group every 2-3 shots into a scene
            if len(shots_in_scene) >= 3 or i == len(sentences) - 1:
                scene = Scene(
                    title=f"Scene {len(scenes) + 1}",
                    shots=list(shots_in_scene),
                )
                scenes.append(scene)
                shots_in_scene.clear()

        return Storyboard(
            title=prompt[:60],
            genre=config.genre,
            style=config.style,
            tone=config.tone,
            target_audience=config.target_audience,
            scenes=scenes,
        )

    def _detect_shot_type(self, text: str) -> ShotType:
        lower = text.lower()
        for keyword, shot in self.SHOT_HINTS.items():
            if keyword in lower:
                return shot
        return ShotType.MEDIUM

    def _detect_camera(self, text: str) -> CameraMove:
        lower = text.lower()
        for keyword, cam in self.CAMERA_HINTS.items():
            if keyword in lower:
                return cam
        return CameraMove.STATIC

    def _detect_lighting(self, text: str) -> str:
        lower = text.lower()
        for kw, light in [
            ("sunset", "golden_hour"), ("sunrise", "golden_hour"),
            ("night", "low_key"), ("dark", "low_key"),
            ("neon", "neon"), ("rain", "overcast"),
            ("studio", "studio_three_point"),
        ]:
            if kw in lower:
                return light
        return "natural"

    def _detect_emotion(self, text: str, default_tone: str) -> str:
        lower = text.lower()
        for kw, emo in [
            ("happy", "joy"), ("sad", "melancholy"), ("angry", "anger"),
            ("fear", "tension"), ("love", "romance"), ("fight", "action"),
            ("peaceful", "serenity"), ("mysterious", "mystery"),
        ]:
            if kw in lower:
                return emo
        return default_tone


# ---------------------------------------------------------------------------
# LLM-based scenarist
# ---------------------------------------------------------------------------


class LLMScenarist:
    """
    Local LLM-powered scene planner.

    Uses HuggingFace transformers (Llama / Mistral) to convert
    a free-form prompt into a structured storyboard JSON.

    Falls back to RuleBasedDecomposer when no model is available.
    """

    SYSTEM_PROMPT = """You are AIPROD Scenarist, an expert film director AI.
Given a creative brief, produce a structured JSON storyboard.

Output format (JSON only, no markdown):
{
  "title": "...",
  "scenes": [
    {
      "title": "Scene N",
      "location": "...",
      "time_of_day": "day|night|dawn|dusk",
      "shots": [
        {
          "description": "Visual description for video generation",
          "shot_type": "wide|medium|close_up|aerial|...",
          "camera_move": "static|pan_left|dolly_in|tracking|...",
          "duration_sec": 3.0,
          "dialogue": "",
          "emotion": "...",
          "lighting": "...",
          "audio_cue": "..."
        }
      ]
    }
  ]
}

Rules:
- Each shot description must be vivid and specific for AI video generation
- Total duration should match the requested length
- Use cinematic vocabulary for shot types and camera moves
- Consider pacing, rhythm, and emotional arc
"""

    def __init__(
        self,
        model_name: str = "models/scenarist/mistral-7b",
        device: str = "auto",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        self._model_name = model_name
        self._device = device
        self._max_tokens = max_new_tokens
        self._temperature = temperature
        self._model: Any = None
        self._tokenizer: Any = None
        self._available = False
        self._fallback = RuleBasedDecomposer()

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]
            self._AutoModel = AutoModelForCausalLM
            self._AutoTokenizer = AutoTokenizer
            self._available = True
        except (ImportError, AttributeError, Exception):
            # AttributeError: triton stub modules on Windows
            # Any other error during transformers import
            pass

    @property
    def available(self) -> bool:
        return self._available

    def load_model(self) -> bool:
        """Load the LLM into memory. Returns True if successful."""
        if not self._available:
            return False
        if self._model is not None:
            return True
        try:
            self._tokenizer = self._AutoTokenizer.from_pretrained(
                self._model_name,
                local_files_only=True,
            )
            self._model = self._AutoModel.from_pretrained(
                self._model_name, local_files_only=True,
                torch_dtype="auto",
                device_map=self._device,
            )
            return True
        except Exception:
            return False

    def generate_storyboard(
        self, prompt: str, config: Optional[CreativeConfig] = None
    ) -> Storyboard:
        """
        Generate a storyboard from a creative prompt.

        Falls back to rule-based decomposer if LLM is not available.
        """
        config = config or CreativeConfig()

        if not self._available or self._model is None:
            return self._fallback.decompose(prompt, config)

        try:
            raw_json = self._call_llm(prompt, config)
            return self._parse_storyboard(raw_json, config)
        except Exception:
            return self._fallback.decompose(prompt, config)

    def _call_llm(self, prompt: str, config: CreativeConfig) -> str:
        """Call the local LLM and return raw JSON string."""
        user_msg = (
            f"Creative brief: {prompt}\n\n"
            f"Genre: {config.genre}\n"
            f"Style: {config.style}\n"
            f"Tone: {config.tone}\n"
            f"Pacing: {config.pacing}\n"
            f"Target duration: {config.target_duration_sec} seconds\n"
            f"Target audience: {config.target_audience}\n"
            f"Color direction: {config.color_direction or 'filmmaker choice'}\n"
            f"Music style: {config.music_style or 'filmmaker choice'}\n\n"
            "Produce the storyboard JSON now."
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Apply chat template
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        import torch

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_tokens,
                temperature=self._temperature,
                do_sample=True,
                top_p=0.9,
            )

        decoded = self._tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded

    def _parse_storyboard(self, raw: str, config: CreativeConfig) -> Storyboard:
        """Parse LLM output JSON into Storyboard."""
        # Extract JSON from potential markdown wrapping
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in LLM output")
        data = json.loads(json_match.group())

        scenes: List[Scene] = []
        for s_data in data.get("scenes", []):
            shots: List[SceneShot] = []
            for sh in s_data.get("shots", []):
                shots.append(SceneShot(
                    description=sh.get("description", ""),
                    shot_type=ShotType(sh.get("shot_type", "medium")),
                    camera_move=CameraMove(sh.get("camera_move", "static")),
                    duration_sec=float(sh.get("duration_sec", 3.0)),
                    dialogue=sh.get("dialogue", ""),
                    emotion=sh.get("emotion", ""),
                    lighting=sh.get("lighting", "natural"),
                    audio_cue=sh.get("audio_cue", ""),
                ))
            scenes.append(Scene(
                title=s_data.get("title", f"Scene {len(scenes)+1}"),
                location=s_data.get("location", ""),
                time_of_day=s_data.get("time_of_day", "day"),
                shots=shots,
            ))

        sb = Storyboard(
            title=data.get("title", config.genre),
            genre=config.genre,
            style=config.style,
            tone=config.tone,
            target_audience=config.target_audience,
            scenes=scenes,
        )
        sb.recalculate_duration()
        return sb

    def unload_model(self) -> None:
        """Free GPU memory."""
        self._model = None
        self._tokenizer = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
