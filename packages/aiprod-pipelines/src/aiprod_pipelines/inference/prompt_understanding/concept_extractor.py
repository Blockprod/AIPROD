"""Concept extraction and semantic relationship discovery.

Provides:
- Named entity extraction
- Concept identification
- Relationship extraction
- Concept hierarchy building
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of named entities."""
    
    PERSON = "PERSON"
    OBJECT = "OBJECT"
    LOCATION = "LOCATION"
    ACTION = "ACTION"
    ATTRIBUTE = "ATTRIBUTE"
    QUALITY = "QUALITY"
    MATERIAL = "MATERIAL"
    STYLE = "STYLE"
    COUNT = "COUNT"


@dataclass
class Concept:
    """Extracted concept from prompt."""
    
    text: str                       # Concept text
    entity_type: EntityType        # Classification
    confidence: float = 0.9        # Confidence 0-1
    parent: Optional[str] = None   # Parent concept
    children: List[str] = None     # Child concepts
    frequency: int = 1             # Occurrence count
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def __repr__(self) -> str:
        return f"Concept({self.text}, type={self.entity_type.value}, conf={self.confidence:.2f})"


@dataclass
class ConceptRelationship:
    """Relationship between two concepts."""
    
    concept1: str              # First concept
    concept2: str              # Second concept
    relation_type: str         # "describes", "contains", "similar", etc.
    strength: float = 0.8      # 0-1 relationship strength


class ConceptExtractor:
    """Extracts and organizes concepts from prompts.
    
    Identifies:
    - Named entities (people, objects, locations)
    - Object attributes
    - Visual qualities
    - Materials and textures
    - Artistic styles
    """
    
    def __init__(self):
        """Initialize concept extractor."""
        
        # Known entities by type
        self.person_names = {
            "man", "woman", "boy", "girl", "person", "character",
            "gentleman", "lady", "child", "adult", "elderly", "young",
        }
        
        self.object_names = {
            "cat", "dog", "bird", "fish", "horse", "elephant",
            "car", "bicycle", "house", "tree", "flower", "rock",
            "book", "chair", "table", "lamp", "window", "door",
        }
        
        self.location_names = {
            "forest", "beach", "mountain", "city", "village", "room",
            "garden", "park", "desert", "ocean", "river", "lake",
            "street", "house", "building", "temple", "castle", "palace",
        }
        
        self.action_verbs = {
            "walking", "running", "jumping", "flying", "swimming",
            "dancing", "sitting", "standing", "falling", "rising",
            "moving", "turning", "climbing", "diving", "floating",
        }
        
        self.quality_words = {
            # Visual
            "bright", "dark", "clear", "blurry", "sharp", "soft",
            "colorful", "monochrome", "vibrant", "muted", "vivid",
            # Emotional
            "peaceful", "chaotic", "serene", "dramatic", "calm",
            "energetic", "melancholic", "joyful", "sad", "angry",
            # Physical
            "smooth", "rough", "wet", "dry", "hot", "cold",
            "hard", "soft", "heavy", "light", "big", "small",
        }
        
        self.style_terms = {
            "cinematic", "photorealistic", "painterly", "cartoon",
            "anime", "watercolor", "oil", "sketch", "digital",
            "surreal", "abstract", "minimalist", "detailed",
            "steampunk", "cyberpunk", "fantasy", "sci-fi",
        }
        
        self.materials = {
            "wood", "metal", "stone", "glass", "plastic", "paper",
            "fabric", "leather", "clay", "marble", "brick", "concrete",
        }
    
    def extract_concepts(self, tokens: List[str]) -> Dict[EntityType, List[Concept]]:
        """
        Extract all concepts from tokenized prompt.
        
        Args:
            tokens: List of token strings
            
        Returns:
            Dictionary mapping entity types to concepts
        """
        concepts_by_type = {et: [] for et in EntityType}
        
        for token in tokens:
            token_lower = token.lower().strip('.,!?;:')
            
            entity_type = self._classify_entity(token_lower)
            if entity_type:
                # Check for existing concept (handle duplicates)
                existing = next(
                    (c for c in concepts_by_type[entity_type] 
                     if c.text == token_lower),
                    None
                )
                
                if existing:
                    existing.frequency += 1
                else:
                    concept = Concept(
                        text=token_lower,
                        entity_type=entity_type,
                        confidence=0.9,
                    )
                    concepts_by_type[entity_type].append(concept)
        
        logger.info(f"Extracted concepts: {sum(len(c) for c in concepts_by_type.values())}")
        
        return concepts_by_type
    
    def _classify_entity(self, token: str) -> Optional[EntityType]:
        """Classify token as entity type."""
        if token in self.person_names:
            return EntityType.PERSON
        elif token in self.object_names:
            return EntityType.OBJECT
        elif token in self.location_names:
            return EntityType.LOCATION
        elif token in self.action_verbs:
            return EntityType.ACTION
        elif token in self.quality_words:
            return EntityType.QUALITY
        elif token in self.style_terms:
            return EntityType.STYLE
        elif token in self.materials:
            return EntityType.MATERIAL
        else:
            return None
    
    def extract_named_entities(self, prompt: str) -> List[Concept]:
        """
        Extract named entities from prompt.
        
        Args:
            prompt: Raw prompt string
            
        Returns:
            List of extracted entities
        """
        entities = []
        tokens = prompt.lower().split()
        
        for token in tokens:
            clean_token = token.strip('.,!?;:')
            entity_type = self._classify_entity(clean_token)
            
            if entity_type:
                concept = Concept(
                    text=clean_token,
                    entity_type=entity_type,
                    confidence=self._confidence_for_entity(clean_token, entity_type),
                )
                entities.append(concept)
        
        return entities
    
    def _confidence_for_entity(self, token: str, entity_type: EntityType) -> float:
        """Estimate confidence in entity classification."""
        # Higher confidence for unambiguous terms
        high_confidence = {EntityType.LOCATION, EntityType.OBJECT}
        
        if entity_type in high_confidence:
            return 0.95
        else:
            return 0.85
    
    def build_concept_hierarchy(
        self,
        concepts_by_type: Dict[EntityType, List[Concept]],
    ) -> Dict[str, Concept]:
        """
        Build hierarchy of concepts.
        
        Args:
            concepts_by_type: Concepts organized by type
            
        Returns:
            Hierarchical concept dictionary
        """
        hierarchy = {}
        
        # Build initial dictionary
        for entity_type, concepts in concepts_by_type.items():
            for concept in concepts:
                hierarchy[concept.text] = concept
        
        # Establish parent-child relationships
        for concept_text, concept in hierarchy.items():
            # Object qualities have object as parent
            if concept.entity_type == EntityType.QUALITY:
                for obj_text, obj_concept in hierarchy.items():
                    if obj_concept.entity_type == EntityType.OBJECT:
                        # Simple heuristic: could be parent
                        concept.parent = obj_text
                        obj_concept.children.append(concept_text)
                        break
            
            # Actions might have objects as their targets
            elif concept.entity_type == EntityType.ACTION:
                for obj_text, obj_concept in hierarchy.items():
                    if obj_concept.entity_type == EntityType.OBJECT:
                        # Could be performed on object
                        pass
        
        return hierarchy
    
    def extract_relationships(
        self,
        concepts: Dict[str, Concept],
    ) -> List[ConceptRelationship]:
        """
        Extract relationships between concepts.
        
        Args:
            concepts: Concept hierarchy
            
        Returns:
            List of relationships
        """
        relationships = []
        concept_list = list(concepts.values())
        
        for i, concept1 in enumerate(concept_list):
            for concept2 in concept_list[i+1:]:
                rel = self._infer_relationship(concept1, concept2)
                if rel:
                    relationships.append(rel)
        
        return relationships
    
    def _infer_relationship(
        self,
        concept1: Concept,
        concept2: Concept,
    ) -> Optional[ConceptRelationship]:
        """Infer relationship between two concepts."""
        
        # Quality describes Object
        if concept1.entity_type == EntityType.QUALITY and concept2.entity_type == EntityType.OBJECT:
            return ConceptRelationship(
                concept1=concept1.text,
                concept2=concept2.text,
                relation_type="describes",
                strength=0.8,
            )
        
        # Action involves Object
        if concept1.entity_type == EntityType.ACTION and concept2.entity_type == EntityType.OBJECT:
            return ConceptRelationship(
                concept1=concept1.text,
                concept2=concept2.text,
                relation_type="acts_on",
                strength=0.8,
            )
        
        # Location context
        if concept1.entity_type == EntityType.LOCATION and concept2.entity_type == EntityType.OBJECT:
            return ConceptRelationship(
                concept1=concept2.text,
                concept2=concept1.text,
                relation_type="located_in",
                strength=0.7,
            )
        
        return None
    
    def get_concept_importance(
        self,
        concepts: Dict[str, Concept],
    ) -> Dict[str, float]:
        """
        Compute importance scores for concepts.
        
        Args:
            concepts: Concept hierarchy
            
        Returns:
            Mapping of concept text to importance (0-1)
        """
        importance = {}
        
        for concept_text, concept in concepts.items():
            # Base importance by type
            type_importance = {
                EntityType.OBJECT: 0.9,
                EntityType.ACTION: 0.85,
                EntityType.LOCATION: 0.7,
                EntityType.QUALITY: 0.6,
                EntityType.STYLE: 0.5,
                EntityType.MATERIAL: 0.4,
                EntityType.ATTRIBUTE: 0.5,
                EntityType.COUNT: 0.3,
                EntityType.PERSON: 0.85,
            }
            
            base_imp = type_importance.get(concept.entity_type, 0.5)
            
            # Boost by frequency
            freq_boost = min(0.2, concept.frequency * 0.05)
            
            # Adjust by confidence
            conf_factor = concept.confidence
            
            final_importance = min(1.0, base_imp + freq_boost) * conf_factor
            importance[concept_text] = final_importance
        
        return importance
    
    def get_semantic_summary(
        self,
        concepts: Dict[str, Concept],
        relationships: List[ConceptRelationship],
    ) -> Dict[str, Any]:
        """
        Get semantic summary of concepts.
        
        Args:
            concepts: Extracted concepts
            relationships: Concept relationships
            
        Returns:
            Summary dictionary
        """
        importance = self.get_concept_importance(concepts)
        
        # Sort by importance
        sorted_concepts = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return {
            "concepts": concepts,
            "relationships": relationships,
            "importance": importance,
            "top_concepts": [c[0] for c in sorted_concepts[:10]],
            "concept_count": len(concepts),
            "relationship_count": len(relationships),
        }
