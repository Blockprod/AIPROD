"""
Creative Director Adapter - Gemini + Distilled Pipeline Integration
===================================================================

Bridges to distilled.py and Gemini API to generate production manifests
with consistency markers.

PHASE 1 implementation (Week 3 Days 1-3 in execution plan).
"""

from typing import Dict, Any, List
import time
import hashlib
import json
from .base import BaseAdapter
from ..schema.schemas import Context


class CreativeDirectorAdapter(BaseAdapter):
    """
    Maps AIPROD schema → Gemini → distilled.py → AIPROD schema.
    
    Generates detailed production manifest with scenes and consistency markers.
    Includes caching for repeated prompts and consistency validation.
    
    Features:
    - Gemini-based scene generation
    - Consistency marker extraction
    - Smart caching with TTL
    - Detailed scene breakdown
    """
    
    # Cache configuration
    CACHE_TTL_HOURS = 168  # 1 week
    CACHE_MAX_SIZE = 5000
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize creative director adapter.
        
        Args:
            config: Configuration including Gemini client, schema transformer
        """
        super().__init__(config)
        
        # Components (will be injected in PHASE 1)
        self.gemini_client = config.get("gemini_client") if config else None
        self.distilled_pipeline = config.get("distilled_pipeline") if config else None
        self.schema_transformer = config.get("schema_transformer") if config else None
        
        # Caching
        self.consistency_cache = {}  # {hash(prompt): manifest}
        self.cache_metadata = {}     # {hash(prompt): {timestamp, budget, etc}}
    
    async def execute(self, ctx: Context) -> Context:
        """
        Generate production manifest from sanitized input.
        
        Args:
            ctx: Context with sanitized_input in memory
            
        Returns:
            Context with production_manifest and consistency_markers
        """
        # Validate context
        if not self.validate_context(ctx, ["sanitized_input"]):
            raise ValueError("Missing sanitized_input in context")
        
        sanitized_input = ctx["memory"]["sanitized_input"]
        prompt = sanitized_input.get("prompt", "")
        duration = sanitized_input.get("duration_sec", 60)
        
        # Generate cache key
        cache_key = self._generate_cache_key(prompt)
        
        # Check cache
        cached_manifest = await self._get_cached_manifest(cache_key, ctx["memory"])
        
        if cached_manifest:
            self.log("info", "Using cached manifest", prompt_hash=cache_key[:8])
            ctx["memory"]["cache_hit"] = True
            ctx["memory"]["production_manifest"] = cached_manifest["manifest"]
            ctx["memory"]["consistency_markers"] = cached_manifest["markers"]
            return ctx
        
        ctx["memory"]["cache_hit"] = False
        
        # Generate new manifest
        if self.gemini_client:
            # Real implementation with Gemini
            manifest, markers = await self._generate_with_gemini(
                prompt, duration, ctx["memory"].get("preferences", {})
            )
        else:
            # Fallback: Simple generation
            manifest, markers = self._generate_simple_manifest(
                prompt, duration
            )
        
        # Cache result
        await self._cache_manifest(cache_key, manifest, markers, ctx["memory"])
        
        # Transform to AIPROD schema if transformer available
        if self.schema_transformer:
            manifest = self.schema_transformer.to_aiprod(manifest)
        
        ctx["memory"]["production_manifest"] = manifest
        ctx["memory"]["consistency_markers"] = markers
        ctx["memory"]["generation_time"] = time.time()
        
        self.log("info", "Created production manifest", 
                 scenes=len(manifest.get("scenes", [])),
                 cache_hit=ctx["memory"]["cache_hit"])
        
        return ctx
    
    async def _generate_with_gemini(
        self, 
        prompt: str, 
        duration: int,
        preferences: Dict[str, Any]
    ) -> tuple:
        """
        Generate manifest using Gemini API.
        
        Args:
            prompt: User prompt
            duration: Video duration in seconds
            preferences: User preferences
            
        Returns:
            Tuple of (manifest, consistency_markers)
        """
        # Prepare Gemini prompt
        gemini_prompt = self._prepare_gemini_prompt(prompt, duration, preferences)
        
        try:
            # Call Gemini API
            response = await self.gemini_client.generate_content(
                gemini_prompt,
                model="gemini-1.5-pro",
                temperature=0.7,
                max_tokens=8000,
                timeout=60
            )
            
            # Parse response
            manifest_data = json.loads(response.text)
            
            # Extract and structure manifest
            manifest = self._structure_manifest(manifest_data, duration)
            
            # Extract consistency markers
            markers = self._extract_consistency_markers(manifest_data, manifest)
            
            return manifest, markers
            
        except Exception as e:
            self.log("error", "Gemini generation failed", error=str(e))
            # Fallback to simple generation on error
            return self._generate_simple_manifest(prompt, duration)
    
    def _generate_simple_manifest(self, prompt: str, duration: int) -> tuple:
        """
        Simple fallback manifest generation without Gemini.
        
        Args:
            prompt: User prompt
            duration: Video duration
            
        Returns:
            Tuple of (manifest, consistency_markers)
        """
        # Estimate number of scenes (simple heuristic)
        if duration > 120:
            num_scenes = 4
        elif duration > 60:
            num_scenes = 3
        elif duration > 30:
            num_scenes = 2
        else:
            num_scenes = 1
        
        scene_duration = duration / num_scenes
        
        # Create scenes
        scenes = []
        for i in range(num_scenes):
            if num_scenes == 1:
                scene_desc = prompt
            elif i == 0:
                scene_desc = f"Opening: {prompt}"
            elif i == num_scenes - 1:
                scene_desc = f"Closing: {prompt}"
            else:
                scene_desc = f"Middle section: {prompt}"
            
            scene = {
                "scene_id": f"scene_{i + 1}",
                "duration_sec": scene_duration,
                "description": scene_desc,
                "camera_movement": self._infer_camera_movement(scene_desc),
                "lighting_style": self._infer_lighting(scene_desc),
                "mood": "neutral",
                "characters": [],
                "props": [],
                "location": "unspecified",
                "time_of_day": "day",
                "weather": "clear",
                "visual_style": {}
            }
            scenes.append(scene)
        
        manifest = {
            "production_id": hashlib.md5(prompt.encode()).hexdigest()[:12],
            "title": "Generated Production",
            "total_duration_sec": duration,
            "scenes": scenes,
            "metadata": {
                "generator": "creative_director_adapter",
                "generation_mode": "simple"
            }
        }
        
        markers = {
            "visual_style": {
                "cinematography": "standard",
                "color_palette": ["natural"],
                "lighting_style": "natural"
            },
            "character_continuity": {},
            "narrative_elements": {
                "pacing": "moderate"
            }
        }
        
        return manifest, markers
    
    def _prepare_gemini_prompt(
        self, 
        user_prompt: str, 
        duration: int,
        preferences: Dict[str, Any]
    ) -> str:
        """Prepare detailed prompt for Gemini."""
        prompt = f"""Generate a detailed production manifest for a {duration}-second video with this description:
        
"{user_prompt}"

User preferences: {json.dumps(preferences, indent=2)}

Return a JSON structure with:
- scenes: array of {{
    - scene_id: unique id
    - duration_sec: duration for this scene
    - description: detailed scene description
    - camera_movement: static/pan/tracking/etc
    - lighting_style: natural/dramatic/warm/cool/etc
    - mood: emotional tone
    - characters: list of character descriptions
    - props: list of props in scene
    - location: where the scene takes place
    - time_of_day: day/night/sunrise/etc
    - weather: weather conditions
    - visual_style: {{tone: bright/dark/cinematic, color_grade: description}}
}}
- consistency_markers: {{
    - visual_style: {{}},
    - character_continuity: describe consistent character traits,
    - narrative_elements: describe story flow
}}

Ensure visual continuity between scenes."""
        
        return prompt
    
    def _structure_manifest(self, data: Dict[str, Any], duration: int) -> Dict[str, Any]:
        """Structure generated data into manifest format."""
        manifest = {
            "production_id": hashlib.md5(json.dumps(data).encode()).hexdigest()[:12],
            "title": data.get("title", "Generated Production"),
            "total_duration_sec": duration,
            "scenes": data.get("scenes", []),
            "metadata": {
                "generator": "gemini",
                "generation_mode": "llm"
            }
        }
        return manifest
    
    def _extract_consistency_markers(
        self, 
        raw_data: Dict[str, Any], 
        manifest: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract and structure consistency markers."""
        raw_markers = raw_data.get("consistency_markers", {})
        
        markers = {
            "visual_style": {
                "cinematography": raw_markers.get("visual_style", {}).get("cinematography", ""),
                "color_palette": self._extract_color_palette(manifest),
                "lighting_style": self._extract_dominant_lighting(manifest)
            },
            "character_continuity": raw_markers.get("character_continuity", {}),
            "narrative_elements": raw_markers.get("narrative_elements", {})
        }
        
        return markers
    
    def _extract_color_palette(self, manifest: Dict[str, Any]) -> List[str]:
        """Extract dominant colors from visual styles."""
        colors = set()
        
        for scene in manifest.get("scenes", []):
            style = scene.get("visual_style", {})
            if "color_grade" in style:
                # Simple heuristic: extract color keywords
                grade_desc = style["color_grade"].lower()
                if "warm" in grade_desc:
                    colors.add("warm")
                if "cool" in grade_desc:
                    colors.add("cool")
                if "dark" in grade_desc:
                    colors.add("dark")
                if "bright" in grade_desc:
                    colors.add("bright")
        
        return list(colors) if colors else ["natural"]
    
    def _extract_dominant_lighting(self, manifest: Dict[str, Any]) -> str:
        """Extract dominant lighting style from scenes."""
        lighting_styles = [s.get("lighting_style", "natural") for s in manifest.get("scenes", [])]
        
        if not lighting_styles:
            return "natural"
        
        # Return most common style
        from collections import Counter
        return Counter(lighting_styles).most_common(1)[0][0]
    
    def _infer_camera_movement(self, description: str) -> str:
        """Infer camera movement from description."""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ["flying", "soaring", "aerial", "drone"]):
            return "aerial"
        elif any(word in desc_lower for word in ["panning", "sweeping", "crossing"]):
            return "pan"
        elif any(word in desc_lower for word in ["tracking", "following", "moving towards"]):
            return "tracking"
        elif any(word in desc_lower for word in ["rotate", "rotating", "spinning"]):
            return "rotate"
        else:
            return "static"
    
    def _infer_lighting(self, description: str) -> str:
        """Infer lighting style from description."""
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ["dramatic", "moody", "dark", "shadow"]):
            return "dramatic"
        elif any(word in desc_lower for word in ["golden", "sunset", "warm"]):
            return "warm"
        elif any(word in desc_lower for word in ["cool", "blue", "cold", "night"]):
            return "cool"
        elif any(word in desc_lower for word in ["studio", "professional", "bright"]):
            return "studio"
        else:
            return "natural"
    
    async def _get_cached_manifest(
        self, 
        cache_key: str, 
        memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Retrieve cached manifest if valid.
        
        Args:
            cache_key: Cache key
            memory: Execution memory
            
        Returns:
            Cached manifest or None if invalid/expired
        """
        if cache_key not in self.consistency_cache:
            return None
        
        metadata = self.cache_metadata.get(cache_key, {})
        timestamp = metadata.get("timestamp", 0)
        
        # Check TTL
        age_hours = (time.time() - timestamp) / 3600
        if age_hours > self.CACHE_TTL_HOURS:
            # Expired
            del self.consistency_cache[cache_key]
            del self.cache_metadata[cache_key]
            return None
        
        # Check budget consistency
        cached_budget = metadata.get("budget", memory.get("budget", 1.0))
        current_budget = memory.get("budget", 1.0)
        
        if abs(cached_budget - current_budget) / current_budget > 0.2:  # 20% tolerance
            # Budget changed significantly, don't use cache
            return None
        
        return self.consistency_cache[cache_key]
    
    async def _cache_manifest(
        self,
        cache_key: str,
        manifest: Dict[str, Any],
        markers: Dict[str, Any],
        memory: Dict[str, Any]
    ) -> None:
        """Cache manifest with metadata."""
        # Manage cache size
        if len(self.consistency_cache) >= self.CACHE_MAX_SIZE:
            # Remove oldest entry
            oldest_key = min(self.cache_metadata, key=lambda k: self.cache_metadata[k]["timestamp"])
            del self.consistency_cache[oldest_key]
            del self.cache_metadata[oldest_key]
        
        # Store manifest
        self.consistency_cache[cache_key] = {
            "manifest": manifest,
            "markers": markers
        }
        
        # Store metadata
        self.cache_metadata[cache_key] = {
            "timestamp": time.time(),
            "budget": memory.get("budget", 1.0),
            "duration": memory.get("duration_sec", 60)
        }
    
    def _generate_cache_key(self, prompt: str) -> str:
        """Generate cache key from prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()
