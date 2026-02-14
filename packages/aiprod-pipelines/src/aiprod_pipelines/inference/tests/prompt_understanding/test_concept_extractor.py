"""Test suite for concept extraction.

Tests:
- Named entity recognition
- Concept classification
- Entity type detection
- Concept hierarchy building
- Relationship extraction
"""

import pytest
from aiprod_pipelines.inference.prompt_understanding.concept_extractor import (
    EntityType,
    Concept,
    ConceptRelationship,
    ConceptExtractor,
)


class TestEntityType:
    """Test EntityType enum."""
    
    def test_entity_types(self):
        """Test available entity types."""
        assert EntityType.PERSON.value == "PERSON"
        assert EntityType.OBJECT.value == "OBJECT"
        assert EntityType.LOCATION.value == "LOCATION"
        assert EntityType.ACTION.value == "ACTION"


class TestConcept:
    """Test Concept dataclass."""
    
    def test_concept_creation(self):
        """Test creating concept."""
        concept = Concept(
            text="cat",
            entity_type=EntityType.OBJECT,
            confidence=0.95,
        )
        
        assert concept.text == "cat"
        assert concept.entity_type == EntityType.OBJECT
        assert concept.confidence == 0.95
    
    def test_concept_with_children(self):
        """Test concept with child concepts."""
        concept = Concept(
            text="cat",
            entity_type=EntityType.OBJECT,
            children=["fluffy", "orange"],
        )
        
        assert len(concept.children) == 2


class TestConceptRelationship:
    """Test ConceptRelationship dataclass."""
    
    def test_relationship_creation(self):
        """Test creating concept relationship."""
        rel = ConceptRelationship(
            concept1="cat",
            concept2="walking",
            relation_type="performs",
            strength=0.9,
        )
        
        assert rel.concept1 == "cat"
        assert rel.concept2 == "walking"


class TestConceptExtractor:
    """Test ConceptExtractor functionality."""
    
    @pytest.fixture
    def extractor(self):
        """Create extractor."""
        return ConceptExtractor()
    
    def test_extractor_initialization(self, extractor):
        """Test extractor setup."""
        assert extractor is not None
        assert len(extractor.person_names) > 0
        assert len(extractor.object_names) > 0
        assert len(extractor.location_names) > 0
    
    def test_entity_classification(self, extractor):
        """Test entity type classification."""
        # Object
        entity_type = extractor._classify_entity("cat")
        assert entity_type == EntityType.OBJECT
        
        # Location
        entity_type = extractor._classify_entity("forest")
        assert entity_type == EntityType.LOCATION
        
        # Action
        entity_type = extractor._classify_entity("walking")
        assert entity_type == EntityType.ACTION
        
        # Quality
        entity_type = extractor._classify_entity("bright")
        assert entity_type == EntityType.QUALITY
    
    def test_extract_concepts(self, extractor):
        """Test concept extraction."""
        tokens = ["cat", "walking", "in", "forest", "bright"]
        
        concepts = extractor.extract_concepts(tokens)
        
        # Should have extracted multiple types
        assert any(concepts[et] for et in EntityType)
    
    def test_named_entity_extraction(self, extractor):
        """Test named entity extraction."""
        prompt = "a cat walking in the forest"
        entities = extractor.extract_named_entities(prompt)
        
        assert len(entities) > 0
        # Should find cat and forest
        texts = {e.text for e in entities}
        assert "cat" in texts
        assert "forest" in texts
    
    def test_concept_hierarchy(self, extractor):
        """Test hierarchy building."""
        concepts_by_type = {
            EntityType.OBJECT: [
                Concept("cat", EntityType.OBJECT),
                Concept("dog", EntityType.OBJECT),
            ],
            EntityType.QUALITY: [
                Concept("bright", EntityType.QUALITY),
            ],
            EntityType.LOCATION: [
                Concept("forest", EntityType.LOCATION),
            ],
            EntityType.ACTION: [
                Concept("walking", EntityType.ACTION),
            ],
            EntityType.ATTRIBUTE: [],
            EntityType.MATERIAL: [],
            EntityType.STYLE: [],
            EntityType.COUNT: [],
            EntityType.PERSON: [],
        }
        
        hierarchy = extractor.build_concept_hierarchy(concepts_by_type)
        
        assert len(hierarchy) > 0
        assert "cat" in hierarchy
    
    def test_relationship_extraction(self, extractor):
        """Test relationship extraction."""
        concepts = {
            "cat": Concept("cat", EntityType.OBJECT),
            "walking": Concept("walking", EntityType.ACTION),
            "bright": Concept("bright", EntityType.QUALITY),
        }
        
        relationships = extractor.extract_relationships(concepts)
        
        # Should find relationships between concepts
        assert isinstance(relationships, list)
    
    def test_concept_importance(self, extractor):
        """Test concept importance scoring."""
        concepts = {
            "cat": Concept("cat", EntityType.OBJECT, frequency=3),
            "walking": Concept("walking", EntityType.ACTION, frequency=1),
            "bright": Concept("bright", EntityType.QUALITY, frequency=1),
        }
        
        importance = extractor.get_concept_importance(concepts)
        
        # Objects should have high importance
        assert importance["cat"] > 0.7
        # Walking should be second
        assert importance["walking"] > importance["bright"]
    
    def test_semantic_summary(self, extractor):
        """Test semantic summary generation."""
        concepts = {
            "cat": Concept("cat", EntityType.OBJECT),
            "forest": Concept("forest", EntityType.LOCATION),
            "walking": Concept("walking", EntityType.ACTION),
        }
        
        relationships = [
            ConceptRelationship("cat", "walking", "performs"),
        ]
        
        summary = extractor.get_semantic_summary(concepts, relationships)
        
        assert "concepts" in summary
        assert "relationships" in summary
        assert "importance" in summary
        assert "top_concepts" in summary
        assert len(summary["top_concepts"]) > 0


class TestConceptIntegration:
    """Integration tests for concept extraction."""
    
    def test_full_extraction_workflow(self):
        """Test complete extraction workflow."""
        extractor = ConceptExtractor()
        
        prompt = "a bright cat walking in the dark forest"
        entities = extractor.extract_named_entities(prompt)
        
        assert len(entities) > 0
        
        # Build hierarchy
        concepts_by_type = extractor.extract_concepts(prompt.split())
        hierarchy = extractor.build_concept_hierarchy(concepts_by_type)
        
        assert len(hierarchy) > 0
    
    def test_multiple_object_extraction(self):
        """Test extracting multiple objects."""
        extractor = ConceptExtractor()
        
        prompt = "a cat and dog playing"
        entities = extractor.extract_named_entities(prompt)
        
        texts = {e.text for e in entities}
        assert "cat" in texts
        assert "dog" in texts
    
    def test_quality_extraction(self):
        """Test quality descriptor extraction."""
        extractor = ConceptExtractor()
        
        tokens = ["bright", "dark", "smooth", "cat"]
        concepts = extractor.extract_concepts(tokens)
        
        qualities = concepts[EntityType.QUALITY]
        assert any(c.text in ["bright", "dark", "smooth"] for c in qualities)
