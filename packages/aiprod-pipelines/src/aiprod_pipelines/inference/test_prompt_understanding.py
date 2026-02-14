"""
Comprehensive Test Suite for Enhanced Prompt Understanding

Tests all components of the prompt understanding infrastructure.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_understanding import (
    # Phase 5 New Components
    SemanticIntent, SemanticToken, ConceptCluster,
    SemanticAnalysisResult, SemanticPromptAnalyzer, SemanticSimilarityMatcher,
    EntityType, Entity, EntityCluster,
    EntityRecognizer, ObjectRelationshipExtractor, SceneUnderstanding,
    EnhancementStrategy, PromptEnhancement, EnhancedPromptResult,
    PromptEnhancementEngine, IterativePromptOptimizer,
    # Phase 2 Components
    PromptAnalyzer, SemanticTokenizer, ConceptExtractor, SemanticGraph
)


class TestSemanticIntentAndTokens(unittest.TestCase):
    """Test semantic intent classification and tokenization."""
    
    def test_semantic_intents_defined(self):
        """Test that semantic intents are properly defined."""
        self.assertTrue(hasattr(SemanticIntent, 'DESCRIPTION'))
        self.assertTrue(hasattr(SemanticIntent, 'ACTION'))
        self.assertTrue(hasattr(SemanticIntent, 'STYLE'))
    
    def test_semantic_token_creation(self):
        """Test semantic token creation."""
        token = SemanticToken(
            text="jumping",
            token_id=0,
            intent=SemanticIntent.ACTION,
            importance=0.9,
        )
        self.assertEqual(token.text, "jumping")
        self.assertEqual(token.intent, SemanticIntent.ACTION)
        self.assertEqual(token.importance, 0.9)
    
    def test_concept_cluster(self):
        """Test concept cluster creation."""
        cluster = ConceptCluster(
            primary_concept="person",
            related_concepts=["human", "character"],
            semantic_distance=0.2,
        )
        self.assertEqual(cluster.primary_concept, "person")
        self.assertEqual(len(cluster.related_concepts), 2)


class TestSemanticPromptAnalyzer(unittest.TestCase):
    """Test semantic prompt analysis."""
    
    def test_analyzer_initialization(self):
        """Test analyzer creation."""
        analyzer = SemanticPromptAnalyzer()
        self.assertIsNotNone(analyzer)
        self.assertTrue(hasattr(analyzer, 'intent_keywords'))
    
    def test_simple_prompt_analysis(self):
        """Test analyzing a simple prompt."""
        analyzer = SemanticPromptAnalyzer()
        prompt = "a cat walking"
        
        result = analyzer.analyze(prompt)
        self.assertIsNotNone(result)
        self.assertEqual(result.original_prompt, prompt)
        self.assertIsNotNone(result.semantic_tokens)
    
    def test_extract_style_descriptors(self):
        """Test style descriptor extraction."""
        analyzer = SemanticPromptAnalyzer()
        prompt = "a cinematic realistic person running"
        
        result = analyzer.analyze(prompt)
        self.assertGreater(len(result.style_descriptors), 0)
    
    def test_extract_action_descriptors(self):
        """Test action descriptor extraction."""
        analyzer = SemanticPromptAnalyzer()
        prompt = "person spinning and dancing"
        
        result = analyzer.analyze(prompt)
        self.assertGreater(len(result.action_descriptors), 0)
    
    def test_constraint_extraction(self):
        """Test constraint/negative prompt extraction."""
        analyzer = SemanticPromptAnalyzer()
        prompt = "a cat without whiskers, not realistic"
        
        result = analyzer.analyze(prompt)
        self.assertGreater(len(result.constraints), 0)
    
    def test_suggest_enhancements(self):
        """Test enhancement suggestions."""
        analyzer = SemanticPromptAnalyzer()
        prompt = "a person"
        
        result = analyzer.analyze(prompt)
        suggestions = analyzer.suggest_enhancements(result)
        self.assertIsInstance(suggestions, list)


class TestSemanticSimilarityMatcher(unittest.TestCase):
    """Test semantic similarity matching."""
    
    def test_matcher_initialization(self):
        """Test matcher creation."""
        matcher = SemanticSimilarityMatcher()
        self.assertIsNotNone(matcher)
    
    def test_similarity_computation(self):
        """Test computing similarity between prompts."""
        analyzer = SemanticPromptAnalyzer()
        matcher = SemanticSimilarityMatcher()
        
        analysis1 = analyzer.analyze("a dog running")
        analysis2 = analyzer.analyze("a dog walking")
        
        similarity = matcher.compute_similarity(analysis1, analysis2)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_find_similar_prompts(self):
        """Test finding similar prompts."""
        analyzer = SemanticPromptAnalyzer()
        matcher = SemanticSimilarityMatcher()
        
        # Populate cache
        prompt1 = analyzer.analyze("a dog running")
        prompt2 = analyzer.analyze("a cat jumping")
        
        matcher.prompt_cache["dog_prompt"] = prompt1
        matcher.prompt_cache["cat_prompt"] = prompt2
        
        # Find similar
        similar = matcher.find_similar_prompts(prompt1, threshold=0.5)
        self.assertIsInstance(similar, list)


class TestEntityRecognition(unittest.TestCase):
    """Test entity recognition from prompts."""
    
    def test_entity_types_defined(self):
        """Test entity types enumeration."""
        self.assertTrue(hasattr(EntityType, 'SUBJECT'))
        self.assertTrue(hasattr(EntityType, 'SCENE'))
        self.assertTrue(hasattr(EntityType, 'STYLE'))
    
    def test_entity_creation(self):
        """Test entity creation."""
        entity = Entity(
            text="person",
            entity_type=EntityType.SUBJECT,
            start_position=0,
            end_position=1,
        )
        self.assertEqual(entity.text, "person")
        self.assertEqual(entity.entity_type, EntityType.SUBJECT)
    
    def test_entity_recognizer(self):
        """Test entity recognition."""
        recognizer = EntityRecognizer()
        prompt = "a person in a forest"
        
        entities = recognizer.recognize_entities(prompt)
        self.assertGreater(len(entities), 0)
    
    def test_entity_clustering(self):
        """Test entity clustering."""
        recognizer = EntityRecognizer()
        prompt = "a person in a forest"
        
        entities = recognizer.recognize_entities(prompt)
        clusters = recognizer.cluster_entities(entities)
        
        self.assertIsInstance(clusters, list)
    
    def test_scene_type_inference(self):
        """Test scene type inference."""
        entities = [
            Entity("forest", EntityType.SCENE, 0, 1),
            Entity("person", EntityType.SUBJECT, 2, 3),
        ]
        
        scene_type = SceneUnderstanding.infer_scene_type(entities)
        self.assertIn(scene_type, ["outdoor", "indoor", "portrait", "group", "abstract"])
    
    def test_visual_complexity_estimation(self):
        """Test visual complexity estimation."""
        entities = [
            Entity("person", EntityType.SUBJECT, 0, 1),
            Entity("forest", EntityType.SCENE, 2, 3),
            Entity("cinematic", EntityType.STYLE, 4, 5),
        ]
        
        complexity = SceneUnderstanding.estimate_visual_complexity(entities)
        self.assertGreaterEqual(complexity, 0.0)
        self.assertLessEqual(complexity, 1.0)


class TestObjectRelationshipExtraction(unittest.TestCase):
    """Test relationship extraction between objects."""
    
    def test_relationship_extractor(self):
        """Test relationship extractor creation."""
        extractor = ObjectRelationshipExtractor()
        self.assertIsNotNone(extractor)
    
    def test_spatial_relationships(self):
        """Test spatial relationship extraction."""
        extractor = ObjectRelationshipExtractor()
        entities = [
            Entity("person", EntityType.SUBJECT, 0, 1),
            Entity("tree", EntityType.OBJECT, 2, 3),
        ]
        
        prompt = "a person standing next to a tree"
        relationships = extractor.extract_relationships(prompt, entities)
        
        self.assertIsInstance(relationships, list)


class TestPromptEnhancementEngine(unittest.TestCase):
    """Test prompt enhancement capabilities."""
    
    def test_engine_initialization(self):
        """Test enhancement engine creation."""
        engine = PromptEnhancementEngine()
        self.assertIsNotNone(engine)
    
    def test_minimal_enhancement(self):
        """Test minimal enhancement strategy."""
        engine = PromptEnhancementEngine()
        prompt = "a cat"
        
        result = engine.enhance(prompt, EnhancementStrategy.MINIMAL)
        self.assertEqual(result.original_prompt, prompt)
        self.assertIsNotNone(result.enhanced_prompt)
        self.assertIsNotNone(result.enhancements)
    
    def test_moderate_enhancement(self):
        """Test moderate enhancement strategy."""
        engine = PromptEnhancementEngine()
        prompt = "a person"
        
        result = engine.enhance(prompt, EnhancementStrategy.MODERATE)
        self.assertGreater(len(result.enhancements), 0)
        self.assertGreater(result.estimated_quality_improvement, 0)
    
    def test_aggressive_enhancement(self):
        """Test aggressive enhancement strategy."""
        engine = PromptEnhancementEngine()
        prompt = "a cat"
        
        result = engine.enhance(prompt, EnhancementStrategy.AGGRESSIVE)
        self.assertGreater(len(result.enhancements), 0)
    
    def test_creative_enhancement(self):
        """Test creative enhancement strategy."""
        engine = PromptEnhancementEngine()
        prompt = "a person running"
        
        result = engine.enhance(prompt, EnhancementStrategy.CREATIVE)
        self.assertIsNotNone(result.enhanced_prompt)
    
    def test_alternatives_generated(self):
        """Test alternative prompt generation."""
        engine = PromptEnhancementEngine()
        prompt = "a dog"
        
        result = engine.enhance(prompt, EnhancementStrategy.MODERATE)
        self.assertGreater(len(result.suggested_alternatives), 0)


class TestIterativePromptOptimization(unittest.TestCase):
    """Test iterative prompt optimization."""
    
    def test_optimizer_initialization(self):
        """Test optimizer creation."""
        optimizer = IterativePromptOptimizer()
        self.assertIsNotNone(optimizer)
    
    def test_optimize_with_mock_evaluator(self):
        """Test optimization with mock evaluator."""
        optimizer = IterativePromptOptimizer()
        
        call_count = [0]
        def mock_evaluator(prompt):
            call_count[0] += 1
            # Always return increasing score
            return 0.5 + (call_count[0] * 0.1)
        
        prompt = "a cat"
        optimized = optimizer.optimize(
            prompt,
            mock_evaluator,
            max_iterations=3
        )
        
        self.assertIsNotNone(optimized)
        self.assertEqual(len(optimizer.history), 3)


class TestIntegration(unittest.TestCase):
    """Integration tests for prompt understanding."""
    
    def test_full_pipeline_analysis_and_enhancement(self):
        """Test full pipeline: analysis -> enhancement -> similarity."""
        analyzer = SemanticPromptAnalyzer()
        engine = PromptEnhancementEngine()
        matcher = SemanticSimilarityMatcher()
        
        # Original prompt
        original = "a person"
        
        # Analyze
        analysis = analyzer.analyze(original)
        self.assertIsNotNone(analysis)
        
        # Enhance
        enhanced = engine.enhance(original)
        self.assertIsNotNone(enhanced.enhanced_prompt)
        
        # Analyze enhanced
        enhanced_analysis = analyzer.analyze(enhanced.enhanced_prompt)
        similarity = matcher.compute_similarity(analysis, enhanced_analysis)
        
        self.assertGreater(similarity, 0.0)
    
    def test_multi_step_enhancement_workflow(self):
        """Test multi-step enhancement workflow."""
        analyzer = SemanticPromptAnalyzer()
        engine = PromptEnhancementEngine()
        
        prompt = "a dog"
        
        # Step 1: Analyze
        analysis = analyzer.analyze(prompt)
        self.assertGreater(len(analysis.semantic_tokens), 0)
        
        # Step 2: Get suggestions
        suggestions = analyzer.suggest_enhancements(analysis)
        
        # Step 3: Enhance
        enhanced = engine.enhance(prompt, EnhancementStrategy.MODERATE)
        self.assertNotEqual(enhanced.enhanced_prompt, prompt)
        
        # Step 4: Re-analyze
        reanalysis = analyzer.analyze(enhanced.enhanced_prompt)
        self.assertIsNotNone(reanalysis)


if __name__ == '__main__':
    unittest.main()
