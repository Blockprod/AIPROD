"""
Input Sanitizer Adapter - Input Validation and Normalization
============================================================

Validates and normalizes pipeline input before processing.
Ensures data consistency, validates ranges, and applies transformations.

PHASE 1 implementation (Week 3 Days 4-5).
"""

from typing import Dict, Any
from .base import BaseAdapter
from ..schema.schemas import Context


class InputSanitizerAdapter(BaseAdapter):
    """
    Validates and sanitizes pipeline input.
    
    Features:
    - Prompt validation (min/max length)
    - Duration validation (10-300 seconds)
    - Budget validation (0.1-10.0 USD)
    - Complexity score normalization (0-1)
    - Preference schema validation
    - Consistency checks
    """
    
    # Validation constraints
    PROMPT_MIN_LENGTH = 10
    PROMPT_MAX_LENGTH = 2000
    DURATION_MIN = 10
    DURATION_MAX = 300
    BUDGET_MIN = 0.1
    BUDGET_MAX = 10.0
    COMPLEXITY_MIN = 0.0
    COMPLEXITY_MAX = 1.0
    
    async def execute(self, ctx: Context) -> Context:
        """
        Validate and sanitize pipeline input.
        
        Args:
            ctx: Context with raw input in memory
            
        Returns:
            Context with sanitized_input
            
        Raises:
            ValueError: If validation fails
        """
        # Extract input
        memory = ctx["memory"]
        
        # Validate required fields
        required = ["prompt", "duration_sec", "budget"]
        for field in required:
            if field not in memory:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate and normalize each field
        sanitized = {
            "prompt": self._validate_prompt(memory["prompt"]),
            "duration_sec": self._validate_duration(memory["duration_sec"]),
            "budget": self._validate_budget(memory["budget"]),
            "complexity": self._normalize_complexity(memory.get("complexity", 0.5)),
            "preferences": self._validate_preferences(memory.get("preferences", {}))
        }
        
        # Additional validations
        self._validate_consistency(sanitized)
        
        # Add to context
        ctx["memory"]["sanitized_input"] = sanitized
        ctx["memory"]["input_validation"] = {
            "prompt_length": len(sanitized["prompt"]),
            "duration": sanitized["duration_sec"],
            "budget_usd": sanitized["budget"],
            "complexity": sanitized["complexity"],
            "preferences_count": len(sanitized["preferences"])
        }
        
        self.log("info", "Input sanitized successfully", 
                 prompt_len=len(sanitized["prompt"]),
                 duration=sanitized["duration_sec"],
                 budget=sanitized["budget"])
        
        return ctx
    
    def _validate_prompt(self, prompt: str) -> str:
        """
        Validate prompt text.
        
        Args:
            prompt: Raw prompt text
            
        Returns:
            Validated and normalized prompt
            
        Raises:
            ValueError: If prompt is invalid
        """
        if not isinstance(prompt, str):
            raise ValueError(f"Prompt must be string, got {type(prompt)}")
        
        # Strip whitespace
        prompt = prompt.strip()
        
        # Check length
        if len(prompt) < self.PROMPT_MIN_LENGTH:
            raise ValueError(
                f"Prompt too short: {len(prompt)} < {self.PROMPT_MIN_LENGTH}"
            )
        
        if len(prompt) > self.PROMPT_MAX_LENGTH:
            raise ValueError(
                f"Prompt too long: {len(prompt)} > {self.PROMPT_MAX_LENGTH}"
            )
        
        # Check for forbidden characters
        forbidden = ["<script", "javascript:", "eval("]
        for forbidden_str in forbidden:
            if forbidden_str.lower() in prompt.lower():
                raise ValueError(f"Forbidden content detected: {forbidden_str}")
        
        return prompt
    
    def _validate_duration(self, duration: int) -> int:
        """Validate video duration in seconds."""
        if not isinstance(duration, (int, float)):
            raise ValueError(f"Duration must be number, got {type(duration)}")
        
        duration = int(duration)
        
        if duration < self.DURATION_MIN:
            raise ValueError(
                f"Duration too short: {duration} < {self.DURATION_MIN}"
            )
        
        if duration > self.DURATION_MAX:
            raise ValueError(
                f"Duration too long: {duration} > {self.DURATION_MAX}"
            )
        
        return duration
    
    def _validate_budget(self, budget: float) -> float:
        """Validate budget in USD."""
        if not isinstance(budget, (int, float)):
            raise ValueError(f"Budget must be number, got {type(budget)}")
        
        budget = float(budget)
        
        if budget < self.BUDGET_MIN:
            raise ValueError(
                f"Budget too low: ${budget} < ${self.BUDGET_MIN}"
            )
        
        if budget > self.BUDGET_MAX:
            raise ValueError(
                f"Budget too high: ${budget} > ${self.BUDGET_MAX}"
            )
        
        return budget
    
    def _normalize_complexity(self, complexity: float) -> float:
        """Normalize complexity score to 0-1 range."""
        if not isinstance(complexity, (int, float)):
            return 0.5  # Default if invalid
        
        complexity = float(complexity)
        
        # Clamp to [0, 1]
        return max(self.COMPLEXITY_MIN, min(self.COMPLEXITY_MAX, complexity))
    
    def _validate_preferences(self, prefs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate user preferences.
        
        Args:
            prefs: User preferences dict
            
        Returns:
            Validated preferences
        """
        if not isinstance(prefs, dict):
            return {}
        
        validated = {}
        
        # List of valid preference keys
        valid_keys = {
            "style": str,           # cinematic, documentary, animated, etc
            "mood": str,            # cheerful, serious, calm, energetic, etc
            "camera_style": str,    # static, panning, tracking, etc
            "color_palette": list,  # list of colors
            "aspect_ratio": str,    # 16:9, 1:1, 9:16, etc
            "fps": int             # 24, 30, 60
        }
        
        for key, expected_type in valid_keys.items():
            if key in prefs:
                value = prefs[key]
                
                # Type check
                if isinstance(value, expected_type):
                    validated[key] = value
                else:
                    self.log("warning", f"Invalid type for preference {key}", 
                             expected=expected_type.__name__, got=type(value).__name__)
        
        return validated
    
    def _validate_consistency(self, sanitized: Dict[str, Any]) -> None:
        """
        Validate logical consistency between fields.
        
        Args:
            sanitized: Sanitized input dict
            
        Raises:
            ValueError: If consistency check fails
        """
        # High complexity should have reasonable budget
        if sanitized["complexity"] > 0.8 and sanitized["budget"] < 0.5:
            self.log("warning", "High complexity with low budget may not meet expectations")
        
        # Short duration should have low complexity
        if sanitized["duration_sec"] < 20 and sanitized["complexity"] > 0.8:
            self.log("warning", "Complex scenes in very short duration may be challenging")
