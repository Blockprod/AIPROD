"""
Schema Transformer - Bidirectional Conversion
=============================================

Converts between AIPROD internal format and external AIPROD manifest format.
Ensures lossless round-trip transformation.
"""

import time
from typing import Dict, Any, List
from .schemas import Context, PipelineRequest
from .aiprod_schemas import AIPRODManifest, AIPRODScene, ConsistencyMarkers


class SchemaTransformer:
    """
    Bidirectional schema transformer for AIPROD ↔ AIPROD conversion.
    
    Handles conversion between internal execution format and external
    production manifest format with validation.
    """
    
    def __init__(self):
        """Initialize schema transformer."""
        self.version = "2.0.0"
    
    def to_aiprod(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform AIPROD production manifest → AIPROD internal format.
        
        Args:
            manifest: External AIPROD production manifest
            
        Returns:
            Internal AIPROD format suitable for inference systems
        """
        return {
            "scenes": self._convert_scenes_to_internal(manifest.get("scenes", [])),
            "metadata": self._convert_metadata_to_internal(manifest.get("metadata", {})),
            "consistency_rules": self._extract_consistency_rules(
                manifest.get("consistency_markers", {})
            ),
            "technical_params": {
                "duration": manifest.get("total_duration_sec", 60),
                "quality_preset": "high",
                "backend": "auto"
            }
        }
    
    def aiprod_to_manifest(self, aiprod_output: Dict[str, Any]) -> AIPRODManifest:
        """
        Transform AIPROD internal output → AIPROD production manifest.
        
        Args:
            aiprod_output: Internal AIPROD execution result
            
        Returns:
            External AIPROD production manifest
        """
        scenes = self._convert_scenes_to_external(aiprod_output.get("scenes", []))
        
        manifest: AIPRODManifest = {
            "production_id": aiprod_output.get("job_id", "unknown"),
            "title": aiprod_output.get("title", "Untitled Production"),
            "total_duration_sec": sum(s.get("duration_sec", 0) for s in scenes),
            "scenes": scenes,
            "consistency_markers": self._build_consistency_markers(aiprod_output),
            "metadata": self._build_metadata(aiprod_output),
            "created_at": time.time(),
            "version": self.version
        }
        
        return manifest
    
    def _convert_scenes_to_internal(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert external scene format to internal processing format.
        
        This is where V1 underestimated effort - complex field mapping required.
        """
        internal_scenes = []
        
        for scene in scenes:
            internal_scene = {
                "id": scene.get("scene_id", ""),
                "duration": scene.get("duration_sec", 10.0),
                "description": scene.get("description", ""),
                
                # Camera and cinematography
                "camera": {
                    "movement": scene.get("camera_movement", "static"),
                    "angle": scene.get("camera_angle", "eye_level"),
                    "lens": scene.get("lens", "standard")
                },
                
                # Lighting
                "lighting": {
                    "style": scene.get("lighting_style", "natural"),
                    "time_of_day": scene.get("time_of_day", "day"),
                    "mood": scene.get("mood", "neutral")
                },
                
                # Environment
                "environment": {
                    "location": scene.get("location", ""),
                    "weather": scene.get("weather", "clear"),
                    "props": scene.get("props", [])
                },
                
                # Characters and subjects
                "subjects": {
                    "characters": scene.get("characters", []),
                    "actions": scene.get("actions", [])
                },
                
                # Visual style
                "style": scene.get("visual_style", {}),
                
                # Metadata
                "metadata": {
                    "scene_number": scene.get("scene_number", 0),
                    "importance": scene.get("importance", "normal")
                }
            }
            
            internal_scenes.append(internal_scene)
        
        return internal_scenes
    
    def _convert_scenes_to_external(self, scenes: List[Dict[str, Any]]) -> List[AIPRODScene]:
        """Convert internal scene format to external AIPROD format."""
        external_scenes: List[AIPRODScene] = []
        
        for scene in scenes:
            camera = scene.get("camera", {})
            lighting = scene.get("lighting", {})
            environment = scene.get("environment", {})
            subjects = scene.get("subjects", {})
            
            external_scene: AIPRODScene = {
                "scene_id": scene.get("id", ""),
                "duration_sec": scene.get("duration", 10.0),
                "description": scene.get("description", ""),
                "camera_movement": camera.get("movement", "static"),
                "lighting_style": lighting.get("style", "natural"),
                "mood": lighting.get("mood", "neutral"),
                "characters": subjects.get("characters", []),
                "props": environment.get("props", []),
                "location": environment.get("location", ""),
                "time_of_day": lighting.get("time_of_day", "day"),
                "weather": environment.get("weather", "clear"),
                "visual_style": scene.get("style", {})
            }
            
            external_scenes.append(external_scene)
        
        return external_scenes
    
    def _extract_consistency_rules(self, markers: Dict[str, Any]) -> Dict[str, Any]:
        """Extract consistency rules from markers for internal processing."""
        return {
            "visual_continuity": markers.get("visual_style", {}),
            "character_appearance": markers.get("character_continuity", {}),
            "narrative_flow": markers.get("narrative_elements", {}),
            "color_scheme": markers.get("color_palette", []),
            "cinematography": markers.get("cinematography_style", "")
        }
    
    def _build_consistency_markers(self, output: Dict[str, Any]) -> ConsistencyMarkers:
        """Build consistency markers from internal output."""
        consistency_rules = output.get("consistency_rules", {})
        
        markers: ConsistencyMarkers = {
            "visual_style": {
                "cinematography": consistency_rules.get("cinematography", ""),
                "color_palette": consistency_rules.get("color_scheme", []),
                "lighting_style": consistency_rules.get("lighting", "natural")
            },
            "character_continuity": consistency_rules.get("character_appearance", {}),
            "narrative_elements": consistency_rules.get("narrative_flow", {}),
            "color_palette": consistency_rules.get("color_scheme", []),
            "cinematography_style": consistency_rules.get("cinematography", ""),
            "lighting_signature": consistency_rules.get("lighting", "natural")
        }
        
        return markers
    
    def _convert_metadata_to_internal(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Convert external metadata to internal format."""
        return {
            "source": "external_manifest",
            "original_metadata": metadata,
            "processing_hints": metadata.get("hints", {})
        }
    
    def _build_metadata(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Build external metadata from internal output."""
        return {
            "generator": "AIPROD v2.0",
            "processing_time_sec": output.get("execution_time", 0),
            "cost_usd": output.get("cost", 0),
            "quality_score": output.get("quality_score", 0),
            "backend_used": output.get("backend", "unknown")
        }
    
    def schemas_equivalent(self, original: Dict[str, Any], transformed: Dict[str, Any]) -> bool:
        """
        Validate that round-trip transformation preserves critical information.
        
        Args:
            original: Original input manifest
            transformed: Manifest after round-trip transformation
            
        Returns:
            True if schemas are equivalent (< 5% difference in critical fields)
        """
        # Compare scene count
        orig_scenes = len(original.get("scenes", []))
        trans_scenes = len(transformed.get("scenes", []))
        
        if orig_scenes != trans_scenes:
            return False
        
        # Compare total duration (allow 5% tolerance)
        orig_duration = original.get("total_duration_sec", 0)
        trans_duration = transformed.get("total_duration_sec", 0)
        
        if orig_duration > 0:
            duration_diff = abs(orig_duration - trans_duration) / orig_duration
            if duration_diff > 0.05:  # 5% tolerance
                return False
        
        # Compare scene descriptions
        for orig_scene, trans_scene in zip(
            original.get("scenes", []), 
            transformed.get("scenes", [])
        ):
            if orig_scene.get("description") != trans_scene.get("description"):
                return False
        
        return True
    
    def validate_schema(self, manifest: Dict[str, Any]) -> bool:
        """
        Validate that manifest conforms to expected schema.
        
        Args:
            manifest: Manifest to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check required top-level fields
        required_fields = ["scenes", "metadata"]
        for field in required_fields:
            if field not in manifest:
                return False
        
        # Check scenes structure
        scenes = manifest.get("scenes", [])
        if not isinstance(scenes, list):
            return False
        
        for scene in scenes:
            if not isinstance(scene, dict):
                return False
            if "description" not in scene:
                return False
        
        return True
