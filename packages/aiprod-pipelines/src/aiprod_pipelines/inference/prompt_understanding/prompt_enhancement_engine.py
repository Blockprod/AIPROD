"""
Prompt Enhancement Engine

Automatically enhances prompts with better descriptions, style terms, technical details.
Handles structured enhancement and iterative improvement.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class EnhancementStrategy(Enum):
    """Strategies for prompt enhancement."""
    MINIMAL = "minimal"  # Small targeted improvements
    MODERATE = "moderate"  # Balanced enhancements
    AGGRESSIVE = "aggressive"  # Significant rewrites
    CREATIVE = "creative"  # Creative interpretations


@dataclass
class PromptEnhancement:
    """Single enhancement to a prompt."""
    original_text: str
    enhanced_text: str
    enhancement_type: str  # "style", "detail", "clarity", "constraint"
    rationale: str
    confidence: float = 0.8
    

@dataclass
class EnhancedPromptResult:
    """Result of prompt enhancement."""
    original_prompt: str
    enhanced_prompt: str
    enhancements: List[PromptEnhancement]
    enhancement_level: EnhancementStrategy
    estimated_quality_improvement: float  # 0-1
    suggested_alternatives: List[str]


class PromptEnhancementEngine:
    """Enhances prompts to improve generation quality."""
    
    def __init__(self):
        self.enhancement_templates = {
            "style_enhancement": [
                "photorealistic",
                "cinematic",
                "high quality",
                "professional",
                "detailed",
                "intricate",
            ],
            "detail_enhancement": [
                "with intricate details",
                "sharp and clear",
                "well-lit",
                "in focus",
                "high resolution",
            ],
            "clarity_enhancement": [
                "showing",
                "displaying",
                "featuring",
                "with",
                "and",
            ],
        }
    
    def enhance(
        self,
        prompt: str,
        strategy: EnhancementStrategy = EnhancementStrategy.MODERATE,
    ) -> EnhancedPromptResult:
        """Enhance prompt based on strategy."""
        enhancements = []
        enhanced = prompt
        
        if strategy == EnhancementStrategy.MINIMAL:
            enhancements = self._enhance_minimal(prompt)
        elif strategy == EnhancementStrategy.MODERATE:
            enhancements = self._enhance_moderate(prompt)
        elif strategy == EnhancementStrategy.AGGRESSIVE:
            enhancements = self._enhance_aggressive(prompt)
        elif strategy == EnhancementStrategy.CREATIVE:
            enhancements = self._enhance_creative(prompt)
        
        # Apply enhancements
        for enhancement in enhancements:
            enhanced = enhanced.replace(
                enhancement.original_text,
                enhancement.enhanced_text,
            )
        
        # Generate alternatives
        alternatives = self._generate_alternatives(prompt, enhancements)
        
        # Estimate improvement
        improvement = min(1.0, len(enhancements) * 0.2)
        
        return EnhancedPromptResult(
            original_prompt=prompt,
            enhanced_prompt=enhanced,
            enhancements=enhancements,
            enhancement_level=strategy,
            estimated_quality_improvement=improvement,
            suggested_alternatives=alternatives,
        )
    
    def _enhance_minimal(self, prompt: str) -> List[PromptEnhancement]:
        """Apply minimal enhancements."""
        enhancements = []
        
        # Add quality descriptors if missing
        if "high quality" not in prompt.lower():
            enh = PromptEnhancement(
                original_text=prompt,
                enhanced_text=f"high quality {prompt}",
                enhancement_type="quality",
                rationale="Adding quality descriptor improves model output",
                confidence=0.9,
            )
            enhancements.append(enh)
        
        return enhancements
    
    def _enhance_moderate(self, prompt: str) -> List[PromptEnhancement]:
        """Apply moderate enhancements."""
        enhancements = self._enhance_minimal(prompt)
        
        # Add style if missing
        if "cinematic" not in prompt.lower() and "realistic" not in prompt.lower():
            enh = PromptEnhancement(
                original_text=prompt,
                enhanced_text=f"cinematic {prompt}",
                enhancement_type="style",
                rationale="Adding cinematic style enhances visual appeal",
                confidence=0.85,
            )
            enhancements.append(enh)
        
        # Add detail level if missing
        if "detailed" not in prompt.lower():
            enh = PromptEnhancement(
                original_text=prompt,
                enhanced_text=f"{prompt}, detailed",
                enhancement_type="detail",
                rationale="Adding detail descriptor improves texture quality",
                confidence=0.8,
            )
            enhancements.append(enh)
        
        return enhancements
    
    def _enhance_aggressive(self, prompt: str) -> List[PromptEnhancement]:
        """Apply aggressive enhancements."""
        enhancements = self._enhance_moderate(prompt)
        
        # Add technical parameters
        enh = PromptEnhancement(
            original_text=prompt,
            enhanced_text=f"{prompt}, 8k, hdr, professional lighting, sharp focus",
            enhancement_type="technical",
            rationale="Adding technical parameters optimizes model performance",
            confidence=0.8,
        )
        enhancements.append(enh)
        
        return enhancements
    
    def _enhance_creative(self, prompt: str) -> List[PromptEnhancement]:
        """Apply creative enhancements."""
        enhancements = []
        
        # Creative rewriting
        creative_versions = [
            f"artistic interpretation of: {prompt}",
            f"imaginative rendering of: {prompt}",
            f"dreamlike version of: {prompt}",
        ]
        
        for creative_version in creative_versions:
            enh = PromptEnhancement(
                original_text=prompt,
                enhanced_text=creative_version,
                enhancement_type="creative",
                rationale="Creative rewriting encourages novel interpretations",
                confidence=0.75,
            )
            enhancements.append(enh)
        
        return [enhancements[0]] if enhancements else []
    
    def _generate_alternatives(
        self,
        prompt: str,
        enhancements: List[PromptEnhancement],
    ) -> List[str]:
        """Generate alternative prompt versions."""
        alternatives = []
        
        # Alternative 1: Add different style
        alternatives.append(f"photorealistic {prompt}")
        
        # Alternative 2: Add different style
        alternatives.append(f"artistic render of {prompt}")
        
        # Alternative 3: Extended details
        alternatives.append(f"{prompt}, ultra detailed, sharp focus, professional quality")
        
        return alternatives


class IterativePromptOptimizer:
    """Iteratively optimizes prompts through feedback loop."""
    
    def __init__(self):
        self.enhancement_engine = PromptEnhancementEngine()
        self.history: List[Tuple[str, float]] = []  # (prompt, score)
    
    def optimize(
        self,
        initial_prompt: str,
        quality_evaluator: callable,
        max_iterations: int = 5,
    ) -> str:
        """Iteratively optimize prompt by evaluating generations."""
        current_prompt = initial_prompt
        best_prompt = initial_prompt
        best_score = 0.0
        
        for iteration in range(max_iterations):
            # Evaluate current
            score = quality_evaluator(current_prompt)
            self.history.append((current_prompt, score))
            
            if score > best_score:
                best_score = score
                best_prompt = current_prompt
            
            # Enhance for next iteration
            strategy = EnhancementStrategy.MODERATE
            result = self.enhancement_engine.enhance(current_prompt, strategy)
            current_prompt = result.enhanced_prompt
        
        return best_prompt
