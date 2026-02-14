"""
Visual Translator Adapter - Manifest to Shot Specifications
===========================================================

Converts high-level scene descriptions to detailed shot specifications
with technical parameters, seeds, and consistency references.

PHASE 1 implementation (Week 3 Days 4-5 + Week 4 Days 1-2).
"""

from typing import Dict, Any, List
import hashlib
from .base import BaseAdapter
from ..schema.schemas import Context


class VisualTranslatorAdapter(BaseAdapter):
    """
    Translates production manifest → detailed shot specifications.
    
    Features:
    - Scene → shot breakdown with technical details
    - Deterministic seed generation for reproducibility
    - Consistency reference generation
    - Visual parameter extraction
    - Streaming preview hints
    """
    
    async def execute(self, ctx: Context) -> Context:
        """
        Convert production manifest to shot list.
        
        Args:
            ctx: Context with production_manifest and consistency_markers
            
        Returns:
            Context with shot_list
        """
        # Validate context
        if not self.validate_context(ctx, ["production_manifest"]):
            raise ValueError("Missing production_manifest in context")
        
        manifest = ctx["memory"]["production_manifest"]
        consistency_markers = ctx["memory"].get("consistency_markers", {})
        
        # Convert scenes to shots
        shot_list: List[Dict[str, Any]] = []
        
        scenes = manifest.get("scenes", [])
        for scene_idx, scene in enumerate(scenes):
            # Split scene into shots if long
            shots = self._split_scene_into_shots(scene, scene_idx)
            shot_list.extend(shots)
        
        # Add consistency references to each shot
        shot_list = self._add_consistency_references(shot_list, consistency_markers)
        
        ctx["memory"]["shot_list"] = shot_list
        ctx["memory"]["translation_summary"] = {
            "total_scenes": len(scenes),
            "total_shots": len(shot_list),
            "total_duration_sec": sum(s.get("duration_sec", 0) for s in shot_list),
            "avg_shot_duration": sum(s.get("duration_sec", 0) for s in shot_list) / len(shot_list) if shot_list else 0
        }
        
        self.log("info", "Visual translation complete", 
                 scenes=len(scenes), shots=len(shot_list),
                 duration=ctx["memory"]["translation_summary"]["total_duration_sec"])
        
        return ctx
    
    def _split_scene_into_shots(self, scene: Dict[str, Any], scene_idx: int) -> List[Dict[str, Any]]:
        """
        Split a scene into individual shots based on duration and natural breaks.
        
        Args:
            scene: Scene from production manifest
            scene_idx: Index of scene
            
        Returns:
            List of shot specifications
        """
        duration = scene.get("duration_sec", 10)
        description = scene.get("description", "")
        
        # Determine number of shots (split long scenes)
        if duration > 60:
            num_shots = 4  # 60s+ → 4 shots of ~15s each
        elif duration > 40:
            num_shots = 3  # 40-60s → 3 shots
        elif duration > 20:
            num_shots = 2  # 20-40s → 2 shots
        else:
            num_shots = 1  # < 20s → 1 shot
        
        shots = []
        shot_duration = duration / num_shots
        
        for shot_num in range(num_shots):
            # Generate shot-specific prompt
            shot_prompt = self._generate_shot_prompt(
                description, shot_num, num_shots, duration
            )
            
            # Generate deterministic seed
            seed = self._generate_seed(
                shot_prompt, scene.get("scene_id", f"scene_{scene_idx}"), shot_num
            )
            
            shot: Dict[str, Any] = {
                "shot_id": f"shot_s{scene_idx + 1:02d}_s{shot_num + 1:02d}",
                "scene_id": scene.get("scene_id", f"scene_{scene_idx}"),
                "scene_number": scene_idx + 1,
                "shot_number": shot_num + 1,
                "total_shots_in_scene": num_shots,
                
                # Video specifications
                "prompt": shot_prompt,
                "negative_prompt": self._generate_negative_prompt(scene),
                "duration_sec": shot_duration,
                "seed": seed,
                
                # Technical parameters
                "technical_params": {
                    "resolution": "1080p",
                    "aspect_ratio": "16:9",
                    "fps": 30,
                    "codec": "h264",
                    "bitrate_mbps": 6
                },
                
                # Visual parameters from scene
                "visual_params": {
                    "camera_movement": scene.get("camera_movement", "static"),
                    "lighting_style": scene.get("lighting_style", "natural"),
                    "mood": scene.get("mood", "neutral"),
                    "time_of_day": scene.get("time_of_day", "day"),
                    "weather": scene.get("weather", "clear")
                },
                
                # Subject and environment
                "subjects": {
                    "characters": scene.get("characters", []),
                    "props": scene.get("props", [])
                },
                "environment": {
                    "location": scene.get("location", ""),
                    "setting": self._describe_setting(scene)
                }
            }
            
            shots.append(shot)
        
        return shots
    
    def _generate_shot_prompt(
        self, 
        description: str, 
        shot_num: int, 
        total_shots: int,
        duration: float
    ) -> str:
        """
        Generate shot-specific prompt with timing information.
        
        Args:
            description: Scene description
            shot_num: Current shot number (0-based)
            total_shots: Total shots in scene
            duration: Total duration
            
        Returns:
            Detailed shot prompt
        """
        # Determine shot timing
        if total_shots == 1:
            timing = "complete"
        elif shot_num == 0:
            timing = "opening"
        elif shot_num == total_shots - 1:
            timing = "closing"
        else:
            timing = "middle"
        
        # Build prompt
        shot_duration = duration / total_shots
        
        prompt = f"{description}. This is the {timing} shot ({shot_num + 1}/{total_shots}), "
        prompt += f"lasting {shot_duration:.1f} seconds. "
        
        # Add transition hints
        if shot_num < total_shots - 1:
            prompt += "Natural transition to next scene expected. "
        
        if shot_num > 0:
            prompt += "Maintain visual continuity with previous shot. "
        
        prompt += "High production quality, cinematic lighting, detailed composition."
        
        return prompt
    
    def _generate_negative_prompt(self, scene: Dict[str, Any]) -> str:
        """Generate negative prompt (what to avoid)."""
        base_negatives = [
            "low quality", "blur", "artifacts", "watermark",
            "text overlay", "unnatural lighting", "graininess",
            "dropped frames", "glitches", "distorted faces"
        ]
        
        # Add scene-specific negatives
        if scene.get("characters"):
            # For scenes with characters
            base_negatives.extend([
                "deformed humans", "wrong number of limbs",
                "inconsistent appearance", "floating characters"
            ])
        
        return ", ".join(base_negatives)
    
    def _generate_seed(self, prompt: str, scene_id: str, shot_num: int) -> int:
        """
        Generate deterministic but unique seed for reproducibility.
        
        Args:
            prompt: Shot prompt
            scene_id: Scene identifier
            shot_num: Shot number
            
        Returns:
            Deterministic integer seed
        """
        # Combine inputs for deterministic hash
        combined = f"{scene_id}_{shot_num}_{prompt}"
        hash_val = hashlib.sha256(combined.encode()).hexdigest()
        
        # Convert hex to integer in valid seed range
        seed = int(hash_val, 16) % (2 ** 32)
        return seed
    
    def _describe_setting(self, scene: Dict[str, Any]) -> str:
        """Describe the setting/environment."""
        parts = []
        
        if scene.get("location"):
            parts.append(f"Location: {scene['location']}")
        
        if scene.get("time_of_day"):
            parts.append(f"Time: {scene['time_of_day']}")
        
        if scene.get("weather"):
            parts.append(f"Weather: {scene['weather']}")
        
        if scene.get("visual_style"):
            style = scene["visual_style"]
            if style.get("tone"):
                parts.append(f"Tone: {style['tone']}")
        
        return ", ".join(parts) if parts else "Default environment"
    
    def _add_consistency_references(
        self, 
        shot_list: List[Dict[str, Any]], 
        consistency_markers: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Add consistency reference information to each shot.
        
        Args:
            shot_list: List of shots
            consistency_markers: Consistency markers from creative director
            
        Returns:
            Enhanced shot list
        """
        for idx, shot in enumerate(shot_list):
            # Reference to previous shot for continuity
            if idx > 0:
                shot["continuity_reference"] = shot_list[idx - 1]["shot_id"]
            
            # Add consistency markers
            shot["consistency_markers"] = {
                "visual_style": consistency_markers.get("visual_style", {}),
                "character_continuity": consistency_markers.get("character_continuity", {}),
                "narrative_elements": consistency_markers.get("narrative_elements", {})
            }
        
        return shot_list
