"""
Entity Recognition and Extraction from Prompts

Identifies named entities, objects, actors, and scene elements in prompts.
Enables structured understanding of prompt content.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum


class EntityType(Enum):
    """Types of entities in prompts."""
    OBJECT = "object"  # Physical objects
    SUBJECT = "subject"  # Main subject (person, animal, etc.)
    SCENE = "scene"  # Location/environment
    ATTRIBUTE = "attribute"  # Visual attribute
    ACTION = "action"  # Verb/action
    STYLE = "style"  # Artistic/visual style
    MATERIAL = "material"  # Material/texture
    QUANTITY = "quantity"  # Numbers/quantities


@dataclass
class Entity:
    """Named entity in prompt."""
    text: str
    entity_type: EntityType
    start_position: int
    end_position: int
    confidence: float = 1.0
    relationships: List[str] = None
    attributes: Dict[str, str] = None
    
    def __post_init__(self):
        if self.relationships is None:
            self.relationships = []
        if self.attributes is None:
            self.attributes = {}


@dataclass
class EntityCluster:
    """Cluster of related entities."""
    entities: List[Entity]
    cluster_type: EntityType
    representative_entity: Entity
    cohesion_score: float = 0.8


class EntityRecognizer:
    """Recognizes entities in prompts."""
    
    def __init__(self):
        self.entity_patterns = {
            EntityType.SUBJECT: [
                "person", "man", "woman", "girl", "boy", "child", "baby",
                "cat", "dog", "bird", "horse", "elephant", "alien", "robot",
                "character", "figure", "creature"
            ],
            EntityType.SCENE: [
                "forest", "mountain", "ocean", "beach", "city", "room", "office",
                "street", "park", "desert", "cave", "space", "void", "cloud"
            ],
            EntityType.MATERIAL: [
                "metal", "wood", "glass", "stone", "fabric", "plastic", "crystal",
                "ice", "fire", "water", "smoke", "light"
            ],
            EntityType.ATTRIBUTE: [
                "bright", "dark", "colorful", "soft", "sharp", "smooth", "rough",
                "transparent", "opaque", "glowing"
            ],
            EntityType.STYLE: [
                "photorealistic", "abstract", "minimalist", "surreal", "cyberpunk",
                "steampunk", "fantasy", "scifi", "noir", "vintage"
            ],
        }
        
        self.builtin_entities: Dict[str, EntityType] = {}
        self._build_entity_index()
    
    def _build_entity_index(self) -> None:
        """Build searchable index of entity patterns."""
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                self.builtin_entities[pattern.lower()] = entity_type
    
    def recognize_entities(self, prompt: str) -> List[Entity]:
        """Extract entities from prompt."""
        entities = []
        words = prompt.lower().split()
        
        for i, word in enumerate(words):
            cleaned_word = word.strip('.,!?;:')
            
            if cleaned_word in self.builtin_entities:
                entity_type = self.builtin_entities[cleaned_word]
                entity = Entity(
                    text=cleaned_word,
                    entity_type=entity_type,
                    start_position=i,
                    end_position=i + 1,
                    confidence=0.9,
                )
                entities.append(entity)
        
        # Deduplicate
        seen = set()
        unique_entities = []
        for e in entities:
            if e.text not in seen:
                unique_entities.append(e)
                seen.add(e.text)
        
        return unique_entities
    
    def cluster_entities(self, entities: List[Entity]) -> List[EntityCluster]:
        """Cluster related entities."""
        clusters_dict: Dict[EntityType, List[Entity]] = {}
        
        for entity in entities:
            if entity.entity_type not in clusters_dict:
                clusters_dict[entity.entity_type] = []
            clusters_dict[entity.entity_type].append(entity)
        
        clusters = []
        for entity_type, entity_list in clusters_dict.items():
            if entity_list:
                cluster = EntityCluster(
                    entities=entity_list,
                    cluster_type=entity_type,
                    representative_entity=entity_list[0],
                    cohesion_score=0.85,
                )
                clusters.append(cluster)
        
        return clusters
    
    def enrich_entities(self, entities: List[Entity], context: str = "") -> List[Entity]:
        """Enrich entities with additional attributes."""
        for entity in entities:
            # Infer attributes from context
            if "big" in context.lower() or "large" in context.lower():
                entity.attributes["size"] = "large"
            elif "small" in context.lower() or "tiny" in context.lower():
                entity.attributes["size"] = "small"
            
            if "fast" in context.lower() or "quick" in context.lower():
                entity.attributes["speed"] = "fast"
        
        return entities


class ObjectRelationshipExtractor:
    """Extracts relationships between objects in prompts."""
    
    def __init__(self):
        self.spatial_relations = [
            "above", "below", "left", "right", "in", "on", "next to",
            "behind", "in front of", "around", "near", "far", "between"
        ]
        
        self.action_relations = [
            "holding", "wearing", "carrying", "pushing", "pulling", "hitting",
            "attacking", "defending", "interacting with", "looking at"
        ]
    
    def extract_relationships(self, prompt: str, entities: List[Entity]) -> List[Tuple[Entity, str, Entity]]:
        """Extract relationships between entities."""
        relationships = []
        tokens = prompt.lower().split()
        
        # Simple pattern matching for relationships
        for i, token in enumerate(tokens):
            if token in self.spatial_relations or token in self.action_relations:
                # Try to find entities before and after
                if i > 0 and i < len(tokens) - 1:
                    # Find matching entities
                    for e1 in entities:
                        for e2 in entities:
                            if e1.text != e2.text:
                                relationships.append((e1, token, e2))
        
        # Deduplicate
        unique_rels = list(set(relationships))
        return unique_rels


class SceneUnderstanding:
    """High-level scene understanding from entities."""
    
    @staticmethod
    def infer_scene_type(entities: List[Entity]) -> str:
        """Infer overall scene type from entities."""
        scene_count = sum(1 for e in entities if e.entity_type == EntityType.SCENE)
        subject_count = sum(1 for e in entities if e.entity_type == EntityType.SUBJECT)
        
        if scene_count > 0:
            return "outdoor" if any("forest" in e.text or "mountain" in e.text for e in entities if e.entity_type == EntityType.SCENE) else "indoor"
        elif subject_count > 0:
            return "portrait" if subject_count == 1 else "group"
        else:
            return "abstract"
    
    @staticmethod
    def estimate_visual_complexity(entities: List[Entity]) -> float:
        """Estimate visual complexity (0-1)."""
        total_entities = len(entities)
        unique_types = len(set(e.entity_type for e in entities))
        
        # More entities and types = higher complexity
        complexity = min(1.0, (total_entities * unique_types) / 20.0)
        return complexity
    
    @staticmethod
    def infer_required_style_enhancements(entities: List[Entity]) -> List[str]:
        """Infer style enhancements needed."""
        suggestions = []
        
        has_style = any(e.entity_type == EntityType.STYLE for e in entities)
        if not has_style:
            suggestions.append("Add artistic style (e.g., cinematic, realistic)")
        
        subject_count = sum(1 for e in entities if e.entity_type == EntityType.SUBJECT)
        if subject_count > 2:
            suggestions.append("Consider adding composition hints for multi-subject arrangement")
        
        return suggestions
