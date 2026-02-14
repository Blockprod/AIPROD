"""Integration tests for prompt understanding system.

Tests end-to-end workflows:
- Prompt parsing through graph construction
- Multi-component interactions
- Complex scene understanding
- Real-world prompt analysis
"""

import pytest
from aiprod_pipelines.inference.prompt_understanding.prompt_analyzer import PromptAnalyzer
from aiprod_pipelines.inference.prompt_understanding.semantic_tokenizer import SemanticTokenizer
from aiprod_pipelines.inference.prompt_understanding.concept_extractor import ConceptExtractor, EntityType
from aiprod_pipelines.inference.prompt_understanding.semantic_graph import SemanticGraph


class TestEndToEndWorkflow:
    """Test complete prompt understanding workflow."""
    
    def test_simple_prompt_full_pipeline(self):
        """Test analysis of simple prompt through all steps."""
        prompt = "a cat walking in a forest"
        
        # Step 1: Analyze
        analyzer = PromptAnalyzer()
        analysis = analyzer.analyze(prompt)
        
        assert len(analysis.tokens) > 0
        assert "cat" in analysis.main_subjects
        assert "walking" in analysis.main_actions
        
        # Step 2: Tokenize semantically
        tokenizer = SemanticTokenizer()
        semantic_tokens, relationships = tokenizer.tokenize_semantic(
            prompt,
            [t.text for t in analysis.tokens],
        )
        
        assert len(semantic_tokens) > 0
        
        # Step 3: Extract concepts
        extractor = ConceptExtractor()
        concepts = extractor.extract_named_entities(prompt)
        
        assert any(c.text == "cat" for c in concepts)
        assert any(c.text == "forest" for c in concepts)
        
        # Step 4: Build graph
        graph = SemanticGraph()
        
        for concept in concepts:
            graph.add_node(
                concept.text,
                concept.entity_type.value,
                importance=concept.confidence,
            )
        
        # Add extracted relationships
        concept_rels = extractor.extract_relationships({c.text: c for c in concepts})
        for rel in concept_rels:
            graph.add_edge(rel.concept1, rel.concept2, rel.relation_type)
        
        # Verify graph
        assert len(graph.nodes) >= 2
    
    def test_complex_scene_analysis(self):
        """Test analyzing complex scene prompt."""
        prompt = (
            "A majestic phoenix rising from blue flames at sunrise, "
            "surrounded by mountains, cinematic lighting, "
            "highly detailed, photorealistic"
        )
        
        analyzer = PromptAnalyzer()
        analysis = analyzer.analyze(prompt)
        
        # Should detect complexity
        assert analysis.complexity > 0.5
        assert analysis.has_style is True
        assert analysis.has_spatial is True
        
        # Extract concepts
        extractor = ConceptExtractor()
        concepts = extractor.extract_named_entities(prompt)
        
        # Should find key concepts
        concept_texts = {c.text for c in concepts}
        assert "phoenix" in concept_texts or "flames" in concept_texts
    
    def test_multi_subject_scene(self):
        """Test scene with multiple subjects."""
        prompt = "A cat and dog playing together in a sunny garden"
        
        analyzer = PromptAnalyzer()
        analysis = analyzer.analyze(prompt)
        
        # Should identify multiple subjects
        assert len(analysis.main_subjects) >= 2
        
        # Extract and build graph
        extractor = ConceptExtractor()
        entities = extractor.extract_named_entities(prompt)
        
        graph = SemanticGraph()
        for entity in entities:
            graph.add_node(
                entity.text,
                entity.entity_type.value,
                importance=entity.confidence
            )
        
        # Queen should have multiple subjects
        assert len(graph.nodes) >= 3  # cat, dog, garden
    
    def test_style_and_attributes(self):
        """Test extracting style attributes."""
        prompt = (
            "surreal watercolor painting of a dreaming astronaut, "
            "soft pastel colors, ethereal mood, abstract style"
        )
        
        analyzer = PromptAnalyzer()
        analysis = analyzer.analyze(prompt)
        
        # Should detect style
        assert analysis.has_style is True
        
        # Should extract descriptors
        assert len(analysis.descriptors) > 0
        
        # Get emphasis
        emphasis = analyzer.get_keyword_emphasis(analysis)
        assert len(emphasis) > 0


class TestPromptComponentInteraction:
    """Test interactions between components."""
    
    def test_analyzer_to_tokenizer(self):
        """Test flow: analyzer -> tokenizer."""
        prompt = "a running dog"
        
        # Analyze
        analyzer = PromptAnalyzer()
        analysis = analyzer.analyze(prompt)
        
        # Tokenize the tokens
        tokenizer = SemanticTokenizer()
        semantic_tokens, _ = tokenizer.tokenize_semantic(
            prompt,
            [t.text for t in analysis.tokens],
        )
        
        # Roles should match analysis
        assert any(t.role.value == "protagonist" for t in semantic_tokens)
        assert any(t.role.value == "action" for t in semantic_tokens)
    
    def test_analyzer_to_graph(self):
        """Test flow: analyzer -> extractor -> graph."""
        prompt = "bright cat in dark forest"
        
        # Analyze
        analyzer = PromptAnalyzer()
        analysis = analyzer.analyze(prompt)
        
        # Extract concepts
        extractor = ConceptExtractor()
        concepts = extractor.extract_named_entities(prompt)
        hierarchy = extractor.build_concept_hierarchy({
            et: [c for c in concepts if c.entity_type == et]
            for et in EntityType
        })
        
        # Build graph
        graph = SemanticGraph()
        for text, concept in hierarchy.items():
            graph.add_node(
                text,
                concept.entity_type.value,
                importance=concept.confidence,
            )
        
        # Verify connectivity
        assert len(graph.nodes) > 0
    
    def test_concept_importance_to_graph(self):
        """Test using concept importance in graph."""
        concepts = {
            "cat": "OBJECT",
            "walking": "ACTION",
            "bright": "QUALITY",
        }
        
        extractor = ConceptExtractor()
        
        # Create concept objects
        from aiprod_pipelines.inference.prompt_understanding.concept_extractor import Concept
        concept_objs = {
            text: Concept(text, EntityType[entity_type])
            for text, entity_type in concepts.items()
        }
        
        # Get importance
        importance = extractor.get_concept_importance(concept_objs)
        
        # Build graph with importance
        graph = SemanticGraph()
        for text, entity_type in concepts.items():
            graph.add_node(
                text,
                entity_type,
                importance=importance.get(text, 0.5)
            )
        
        # Get central concepts
        central = graph.get_central_concepts(top_k=3)
        
        # Object should be most central
        assert central[0][0] == "cat"


class TestRealWorldPrompts:
    """Test with realistic video generation prompts."""
    
    def test_film_noir_prompt(self):
        """Test film noir scene prompt."""
        prompt = (
            "film noir style: a mysterious figure in a fedora hat, "
            "standing in shadows with dramatic lighting, "
            "rain-slicked street, 1940s atmosphere, "
            "high contrast black and white, moody cinematic"
        )
        
        analyzer = PromptAnalyzer()
        analysis = analyzer.analyze(prompt)
        
        # Should detect high complexity
        assert analysis.complexity > 0.6
        # Should detect style
        assert analysis.has_style is True
        # Should detect temporal (1940s)
        assert analysis.has_temporal is True
    
    def test_nature_documentary_prompt(self):
        """Test nature documentary style prompt."""
        prompt = (
            "documentary style nature footage: "
            "majestic elephant herd walking across vast savanna, "
            "golden hour sunlight, dust particles in air, "
            "distant acacia trees, shallow depth of field, "
            "cinematic color grading, 4K ultra detailed"
        )
        
        analyzer = PromptAnalyzer()
        analysis = analyzer.analyze(prompt)
        
        # Analyze
        assert len(analysis.main_subjects) > 0
        assert len(analysis.descriptors) > 5
        
        # Extract entities
        extractor = ConceptExtractor()
        entities = extractor.extract_named_entities(prompt)
        
        entity_types = {e.entity_type for e in entities}
        assert EntityType.OBJECT in entity_types
        assert EntityType.LOCATION in entity_types
    
    def test_fantasy_prompt(self):
        """Test fantasy worldbuilding prompt."""
        prompt = (
            "high fantasy magical realm: ancient dragon sleeping on hoard of gold, "
            "surrounded by glowing runes, crystal cave formations, "
            "ethereal blue aurora, floating islands in background, "
            "surreal dreamlike atmosphere, highly detailed, 8K"
        )
        
        analyzer = PromptAnalyzer()
        analysis = analyzer.analyze(prompt)
        
        # Complex fantasy scene
        assert analysis.complexity > 0.5
        assert analysis.has_style is True
        assert analysis.has_spatial is True
        
        # Build complete graph
        extractor = ConceptExtractor()
        entities = extractor.extract_named_entities(prompt)
        
        # Create hierarchy
        concepts_by_type = {
            et: [e for e in entities if e.entity_type == et]
            for et in EntityType
        }
        hierarchy = extractor.build_concept_hierarchy(concepts_by_type)
        
        # Build graph
        graph = SemanticGraph()
        for text, concept in hierarchy.items():
            graph.add_node(text, concept.entity_type.value)
        
        # Find central elements
        central = graph.get_central_concepts(top_k=3)
        
        # Should identify key elements
        assert len(central) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_prompt(self):
        """Test handling empty prompt."""
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("")
        
        # Should handle gracefully
        assert result.original_prompt == ""
        assert result.complexity < 0.5
    
    def test_single_word_prompt(self):
        """Test single-word prompt."""
        analyzer = PromptAnalyzer()
        result = analyzer.analyze("cat")
        
        assert len(result.tokens) == 1
        assert result.complexity < 0.3
    
    def test_special_characters(self):
        """Test prompt with special characters."""
        analyzer = PromptAnalyzer()
        prompt = "cat! running? jumping..."
        result = analyzer.analyze(prompt)
        
        # Should handle punctuation
        assert len(result.tokens) > 0
    
    def test_unknown_language_mixed(self):
        """Test mixed language content."""
        analyzer = PromptAnalyzer()
        prompt = "a cat in the jardín (garden)"
        result = analyzer.analyze(prompt)
        
        # Should still process
        assert len(result.tokens) > 0
