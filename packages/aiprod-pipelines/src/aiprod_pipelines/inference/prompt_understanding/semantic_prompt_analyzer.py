"""
Semantic Prompt Analysis

Advanced semantic analysis of prompts to extract intent, concepts, and relationships.
Enables auto-enhancement and better conditional guidance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import re


class SemanticIntent(Enum):
    """Semantic intent classification."""
    DESCRIPTION = "description"  # Object/scene description
    ACTION = "action"  # Dynamic action/motion
    STYLE = "style"  # Aesthetic style
    EMOTION = "emotion"  # Emotional tone
    COMPOSITION = "composition"  # Layout/framing
    LIGHTING = "lighting"  # Light/shadow
    COLOR = "color"  # Color palette
    TIMELINE = "timeline"  # Time-related
    CONSTRAINT = "constraint"  # Negative/constraint
    MODIFICATION = "modification"  # Editing operation


@dataclass
class SemanticToken:
    """Token with semantic properties."""
    text: str
    token_id: int
    embedding: Optional[List[float]] = None
    intent: Optional[SemanticIntent] = None
    confidence: float = 1.0
    importance: float = 1.0
    semantic_category: Optional[str] = None


@dataclass
class ConceptCluster:
    """Cluster of related semantic concepts."""
    primary_concept: str
    related_concepts: List[str] = field(default_factory=list)
    semantic_distance: float = 0.0
    is_primary: bool = True
    category: Optional[str] = None
    

@dataclass
class SemanticAnalysisResult:
    """Result of semantic prompt analysis."""
    original_prompt: str
    semantic_tokens: List[SemanticToken]
    concepts: List[ConceptCluster]
    dominant_intents: Dict[SemanticIntent, float]  # intent -> confidence
    primary_subject: Optional[str]
    secondary_subjects: List[str] = field(default_factory=list)
    style_descriptors: List[str] = field(default_factory=list)
    action_descriptors: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    temporal_info: Optional[str] = None
    lighting_info: Optional[str] = None
    camera_info: Optional[str] = None
    

class SemanticPromptAnalyzer:
    """Analyzes prompts at semantic level."""
    
    def __init__(self):
        self.intent_keywords = {
            SemanticIntent.ACTION: [
                "moving", "jumping", "flying", "running", "walking", "spinning",
                "dancing", "falling", "rising", "rotating", "transforming"
            ],
            SemanticIntent.STYLE: [
                "realistic", "cinematic", "cartoon", "animated", "modern", "vintage",
                "oil painting", "watercolor", "sketch", "3d", "illustration"
            ],
            SemanticIntent.EMOTION: [
                "joyful", "sad", "peaceful", "chaotic", "mysterious", "dramatic",
                "energetic", "calm", "tense", "beautiful"
            ],
            SemanticIntent.LIGHTING: [
                "bright", "dark", "dimly lit", "spotlight", "shadows", "golden hour",
                "neon", "moonlight", "sunrise", "sunset", "backlit"
            ],
            SemanticIntent.COLOR: [
                "colorful", "vibrant", "muted", "grayscale", "blue", "red", "green",
                "warm", "cool", "pastel", "saturated"
            ],
        }
        
        self.constraint_prefixes = ["not", "without", "no", "avoid", "exclude"]
    
    def analyze(self, prompt: str) -> SemanticAnalysisResult:
        """Perform semantic analysis on prompt."""
        tokens = self._tokenize(prompt)
        semantic_tokens = self._classify_tokens(tokens)
        concepts = self._extract_concepts(prompt, semantic_tokens)
        intents = self._infer_intents(semantic_tokens)
        subjects = self._extract_subjects(prompt, concepts)
        styles = self._extract_styles(prompt, semantic_tokens)
        actions = self._extract_actions(prompt, semantic_tokens)
        constraints = self._extract_constraints(prompt)
        
        return SemanticAnalysisResult(
            original_prompt=prompt,
            semantic_tokens=semantic_tokens,
            concepts=concepts,
            dominant_intents=intents,
            primary_subject=subjects[0] if subjects else None,
            secondary_subjects=subjects[1:],
            style_descriptors=styles,
            action_descriptors=actions,
            constraints=constraints,
        )
    
    def _tokenize(self, prompt: str) -> List[str]:
        """Tokenize prompt into words."""
        return prompt.lower().split()
    
    def _classify_tokens(self, tokens: List[str]) -> List[SemanticToken]:
        """Classify tokens by intent."""
        classified = []
        for i, token in enumerate(tokens):
            cleaned = re.sub(r'[^\w]', '', token)
            if not cleaned:
                continue
            
            semantic_token = SemanticToken(
                text=cleaned,
                token_id=i,
            )
            
            # Infer intent
            for intent, keywords in self.intent_keywords.items():
                if cleaned in keywords:
                    semantic_token.intent = intent
                    break
            
            classified.append(semantic_token)
        
        return classified
    
    def _extract_concepts(self, prompt: str, tokens: List[SemanticToken]) -> List[ConceptCluster]:
        """Extract semantic concepts."""
        concepts = []
        seen = set()
        
        for token in tokens:
            if token.text not in seen:
                cluster = ConceptCluster(
                    primary_concept=token.text,
                    category=token.intent.value if token.intent else None,
                )
                concepts.append(cluster)
                seen.add(token.text)
        
        return concepts
    
    def _infer_intents(self, tokens: List[SemanticToken]) -> Dict[SemanticIntent, float]:
        """Infer dominant intents."""
        intent_scores = {}
        
        for token in tokens:
            if token.intent:
                intent_scores[token.intent] = intent_scores.get(token.intent, 0) + token.confidence
        
        # Normalize
        total = sum(intent_scores.values())
        if total > 0:
            intent_scores = {k: v/total for k, v in intent_scores.items()}
        
        return intent_scores
    
    def _extract_subjects(self, prompt: str, concepts: List[ConceptCluster]) -> List[str]:
        """Extract primary and secondary subjects."""
        # Simple heuristic: nouns and objects mentioned
        subjects = [c.primary_concept for c in concepts[:3]]
        return subjects
    
    def _extract_styles(self, prompt: str, tokens: List[SemanticToken]) -> List[str]:
        """Extract style descriptors."""
        styles = []
        for token in tokens:
            if token.intent == SemanticIntent.STYLE:
                styles.append(token.text)
        return styles
    
    def _extract_actions(self, prompt: str, tokens: List[SemanticToken]) -> List[str]:
        """Extract action descriptors."""
        actions = []
        for token in tokens:
            if token.intent == SemanticIntent.ACTION:
                actions.append(token.text)
        return actions
    
    def _extract_constraints(self, prompt: str) -> List[str]:
        """Extract constraint/negative prompts."""
        constraints = []
        tokens = prompt.lower().split()
        
        for i, token in enumerate(tokens):
            if token in self.constraint_prefixes and i + 1 < len(tokens):
                constraint = tokens[i + 1]
                constraints.append(constraint)
        
        return constraints
    
    def suggest_enhancements(self, analysis: SemanticAnalysisResult) -> List[str]:
        """Suggest prompt enhancements."""
        suggestions = []
        
        # If no style mentioned, suggest adding one
        if not analysis.style_descriptors:
            suggestions.append("Consider specifying an art style (e.g., 'cinematic', 'animated', 'oil painting')")
        
        # If dominant action but no lighting, suggest adding
        if SemanticIntent.ACTION in analysis.dominant_intents and not analysis.lighting_info:
            suggestions.append("Consider adding lighting details for better visual quality")
        
        # If multiple objects but no composition hint
        if len(analysis.secondary_subjects) > 2 and not analysis.camera_info:
            suggestions.append("Consider specifying composition or camera angle")
        
        return suggestions


class SemanticSimilarityMatcher:
    """Matches semantically similar prompts."""
    
    def __init__(self):
        self.prompt_cache: Dict[str, SemanticAnalysisResult] = {}
    
    def compute_similarity(
        self,
        prompt1: SemanticAnalysisResult,
        prompt2: SemanticAnalysisResult,
    ) -> float:
        """Compute semantic similarity (0-1)."""
        # Compare intents
        intents1 = set(prompt1.dominant_intents.keys())
        intents2 = set(prompt2.dominant_intents.keys())
        intent_overlap = len(intents1 & intents2) / len(intents1 | intents2) if intents1 or intents2 else 0
        
        # Compare subjects
        subjects1 = {prompt1.primary_subject} if prompt1.primary_subject else set()
        subjects2 = {prompt2.primary_subject} if prompt2.primary_subject else set()
        subject_overlap = len(subjects1 & subjects2) / max(len(subjects1 | subjects2), 1)
        
        # Weighted similarity
        similarity = 0.6 * intent_overlap + 0.4 * subject_overlap
        return min(1.0, max(0.0, similarity))
    
    def find_similar_prompts(
        self,
        query_prompt: SemanticAnalysisResult,
        threshold: float = 0.7,
    ) -> List[Tuple[str, float]]:
        """Find similar prompts from cache."""
        similar = []
        
        for original_prompt, cached_analysis in self.prompt_cache.items():
            similarity = self.compute_similarity(query_prompt, cached_analysis)
            if similarity >= threshold:
                similar.append((original_prompt, similarity))
        
        # Sort by similarity descending
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar
