"""Core prompt analysis and semantic understanding.

Provides:
- Prompt parsing and tokenization
- Syntax analysis
- Language detection
- Structure preservation for semantic understanding
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class PromptToken:
    """Individual token in parsed prompt."""
    
    text: str                   # Original text
    token_id: int              # Position in sequence
    token_type: str            # "noun", "verb", "adjective", "object", "action", etc.
    embedding: Optional[Any] = None  # Optional embedding
    weight: float = 1.0        # Importance weight (0-2)
    parent_id: Optional[int] = None   # Parent token for dependency
    
    def __repr__(self) -> str:
        return f"Token({self.text}, type={self.token_type}, weight={self.weight:.2f})"


@dataclass
class PromptAnalysisResult:
    """Result of comprehensive prompt analysis."""
    
    original_prompt: str        # Raw input
    tokens: List[PromptToken]   # Parsed tokens
    
    # Structural information
    main_subjects: List[str]    # Primary subjects
    main_actions: List[str]     # Primary actions/verbs
    descriptors: List[str]      # Adjectives and descriptive phrases
    objects: List[str]          # Objects and targets
    
    # Metadata
    language: str = "en"        # Detected language
    complexity: float = 0.5     # 0-1 scale
    has_temporal: bool = False  # Time-related content
    has_spatial: bool = False   # Location/position content
    has_style: bool = False     # Style/aesthetic descriptors
    
    # Relationships
    token_weights: Dict[str, float] = field(default_factory=dict)  # Token importance
    
    def __repr__(self) -> str:
        return (
            f"PromptAnalysis(subjects={self.main_subjects}, "
            f"actions={self.main_actions}, complexity={self.complexity:.2f})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["tokens"] = [asdict(t) for t in self.tokens]
        return result


class PromptAnalyzer:
    """Analyzes and parses video generation prompts.
    
    Handles:
    - Subject extraction
    - Action/verb identification
    - Descriptor parsing
    - Temporal/spatial markers
    - Style indicators
    - Complexity estimation
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize prompt analyzer.
        
        Args:
            device: Computation device
        """
        self.device = device
        
        # Token type patterns
        self.action_verbs = {
            "walking", "running", "jumping", "flying", "swimming", "dancing",
            "falling", "rising", "moving", "rolling", "sliding", "spinning",
            "turning", "diving", "floating", "gliding", "crawling", "climbing",
            "sitting", "standing", "lying", "resting", "eating", "drinking",
        }
        
        self.style_words = {
            "cinematic", "professional", "artistic", "painterly", "photorealistic",
            "highly detailed", "sharp", "clear", "vivid", "colorful", "monochrome",
            "noir", "steampunk", "cyberpunk", "surreal", "abstract", "minimalist",
            "maximalist", "elegant", "dramatic", "serene", "ethereal", "gritty",
        }
        
        self.temporal_words = {
            "slowly", "quickly", "fast", "smooth", "suddenly", "gradually",
            "throughout", "during", "while", "when", "as", "before", "after",
        }
        
        self.spatial_words = {
            "above", "below", "left", "right", "center", "forward", "backward",
            "inside", "outside", "around", "through", "across", "near", "far",
            "high", "low", "top", "bottom", "side", "corner", "edge",
        }
    
    def analyze(self, prompt: str) -> PromptAnalysisResult:
        """
        Analyze a prompt comprehensively.
        
        Args:
            prompt: Video generation prompt text
            
        Returns:
            PromptAnalysisResult with parsed structure
        """
        # Clean and normalize
        prompt = prompt.strip()
        
        # Tokenize basic
        tokens = self._tokenize(prompt)
        
        # Enhanced tokenization with types
        enhanced_tokens = self._classify_tokens(tokens)
        
        # Extract structural elements
        subjects = self._extract_subjects(enhanced_tokens)
        actions = self._extract_actions(enhanced_tokens)
        descriptors = self._extract_descriptors(enhanced_tokens)
        objects = self._extract_objects(enhanced_tokens)
        
        # Detect special characteristics
        has_temporal = any(word in prompt.lower() for word in self.temporal_words)
        has_spatial = any(word in prompt.lower() for word in self.spatial_words)
        has_style = any(word in prompt.lower() for word in self.style_words)
        
        # Calculate complexity
        complexity = self._estimate_complexity(
            len(subjects),
            len(actions),
            len(descriptors),
            has_temporal,
            has_spatial,
            has_style,
        )
        
        # Compute token weights
        token_weights = self._compute_token_weights(enhanced_tokens, subjects, actions)
        
        result = PromptAnalysisResult(
            original_prompt=prompt,
            tokens=enhanced_tokens,
            main_subjects=subjects,
            main_actions=actions,
            descriptors=descriptors,
            objects=objects,
            language="en",
            complexity=complexity,
            has_temporal=has_temporal,
            has_spatial=has_spatial,
            has_style=has_style,
            token_weights=token_weights,
        )
        
        logger.info(
            f"Analyzed prompt: subjects={subjects}, "
            f"actions={actions}, complexity={complexity:.2f}"
        )
        
        return result
    
    def _tokenize(self, text: str) -> List[str]:
        """Basic tokenization."""
        # Split on whitespace and punctuation
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        tokens = text.split()
        return [t for t in tokens if t.strip()]
    
    def _classify_tokens(self, tokens: List[str]) -> List[PromptToken]:
        """Classify tokens by type."""
        classified = []
        
        for idx, token in enumerate(tokens):
            token_lower = token.lower().strip('.,!?;:')
            
            # Determine type
            token_type = "unknown"
            weight = 1.0
            
            if token_lower in self.action_verbs:
                token_type = "action"
                weight = 1.8
            elif token_lower in self.style_words:
                token_type = "style"
                weight = 1.5
            elif token_lower in self.temporal_words:
                token_type = "temporal"
                weight = 1.3
            elif token_lower in self.spatial_words:
                token_type = "spatial"
                weight = 1.3
            elif len(token_lower) > 2 and token_lower[0].isupper():
                token_type = "noun"
                weight = 1.6
            elif token_lower in {"very", "extremely", "incredibly", "amazingly"}:
                token_type = "intensifier"
                weight = 1.2
            else:
                token_type = "modifier"
                weight = 1.0
            
            prompt_token = PromptToken(
                text=token,
                token_id=idx,
                token_type=token_type,
                weight=weight,
            )
            classified.append(prompt_token)
        
        return classified
    
    def _extract_subjects(self, tokens: List[PromptToken]) -> List[str]:
        """Extract main subjects (typically nouns)."""
        subjects = []
        
        for token in tokens:
            # Primary nouns with high weight
            if token.token_type == "noun":
                subjects.append(token.text.lower())
        
        # Limit to top subjects
        return subjects[:5] if subjects else ["scene"]
    
    def _extract_actions(self, tokens: List[PromptToken]) -> List[str]:
        """Extract main actions (verbs)."""
        actions = []
        
        for token in tokens:
            if token.token_type == "action":
                actions.append(token.text.lower())
        
        return actions[:5] if actions else ["static"]
    
    def _extract_descriptors(self, tokens: List[PromptToken]) -> List[str]:
        """Extract descriptive phrases."""
        descriptors = []
        
        for token in tokens:
            if token.token_type in {"style", "modifier"}:
                descriptors.append(token.text.lower())
        
        return descriptors[:8]
    
    def _extract_objects(self, tokens: List[PromptToken]) -> List[str]:
        """Extract object references."""
        # Objects typically come after nouns or actions
        return []  # Simplified for now
    
    def _estimate_complexity(
        self,
        num_subjects: int,
        num_actions: int,
        num_descriptors: int,
        has_temporal: bool,
        has_spatial: bool,
        has_style: bool,
    ) -> float:
        """Estimate prompt complexity on 0-1 scale.
        
        More subjects, actions, descriptors = higher complexity.
        Special characteristics increase complexity.
        """
        base_complexity = 0.3
        
        # Component contributions
        subject_contrib = min(0.2, num_subjects * 0.05)
        action_contrib = min(0.2, num_actions * 0.08)
        descriptor_contrib = min(0.15, num_descriptors * 0.03)
        
        # Characteristic bonuses
        char_contrib = sum([
            0.1 if has_temporal else 0,
            0.1 if has_spatial else 0,
            0.1 if has_style else 0,
        ])
        
        complexity = min(1.0, base_complexity + subject_contrib + action_contrib + 
                         descriptor_contrib + char_contrib)
        
        return complexity
    
    def _compute_token_weights(
        self,
        tokens: List[PromptToken],
        subjects: List[str],
        actions: List[str],
    ) -> Dict[str, float]:
        """Compute relative importance of each token."""
        weights = {}
        
        for token in tokens:
            weight = token.weight
            
            # Boost if in subjects or actions
            if token.text.lower() in subjects:
                weight *= 1.5
            if token.text.lower() in actions:
                weight *= 1.4
            
            weights[token.text.lower()] = weight
        
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def analyze_batch(self, prompts: List[str]) -> List[PromptAnalysisResult]:
        """Analyze multiple prompts.
        
        Args:
            prompts: List of prompt strings
            
        Returns:
            List of analysis results
        """
        results = []
        for prompt in prompts:
            result = self.analyze(prompt)
            results.append(result)
        
        return results
    
    def get_keyword_emphasis(self, analysis: PromptAnalysisResult) -> Dict[str, float]:
        """Get keyword emphasis scores from analysis.
        
        Args:
            analysis: Prompt analysis result
            
        Returns:
            Dictionary mapping keywords to emphasis (0-1)
        """
        emphasis = {}
        
        # Subjects high emphasis
        for subject in analysis.main_subjects:
            emphasis[subject] = min(1.0, 0.7 + 0.1 * len(analysis.main_subjects))
        
        # Actions moderate-high emphasis
        for action in analysis.main_actions:
            emphasis[action] = min(1.0, 0.5 + 0.1 * len(analysis.main_actions))
        
        # Descriptors lower emphasis
        for desc in analysis.descriptors:
            emphasis[desc] = 0.4
        
        return emphasis
