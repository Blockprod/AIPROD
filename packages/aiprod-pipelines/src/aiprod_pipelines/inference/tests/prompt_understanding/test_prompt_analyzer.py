"""Test suite for prompt analysis.

Tests:
- Prompt parsing and tokenization
- Subject/action/descriptor extraction
- Complexity estimation
- Language analysis
- Structural preservation
"""

import pytest
from aiprod_pipelines.inference.prompt_understanding.prompt_analyzer import (
    PromptToken,
    PromptAnalysisResult,
    PromptAnalyzer,
)


class TestPromptToken:
    """Test PromptToken dataclass."""
    
    def test_token_creation(self):
        """Test creating a token."""
        token = PromptToken(
            text="walking",
            token_id=0,
            token_type="action",
            weight=1.8,
        )
        
        assert token.text == "walking"
        assert token.token_type == "action"
        assert token.weight == 1.8
    
    def test_token_repr(self):
        """Test token string representation."""
        token = PromptToken(
            text="cat",
            token_id=1,
            token_type="noun",
            weight=1.6,
        )
        
        repr_str = repr(token)
        assert "cat" in repr_str
        assert "noun" in repr_str


class TestPromptAnalyzer:
    """Test PromptAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer."""
        return PromptAnalyzer()
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer setup."""
        assert analyzer is not None
        assert len(analyzer.action_verbs) > 0
        assert len(analyzer.style_words) > 0
    
    def test_simple_prompt_analysis(self, analyzer):
        """Test analyzing simple prompt."""
        prompt = "a cat walking in the forest"
        result = analyzer.analyze(prompt)
        
        assert result.original_prompt == prompt
        assert len(result.tokens) > 0
        assert len(result.main_subjects) > 0
        assert len(result.main_actions) > 0
    
    def test_action_extraction(self, analyzer):
        """Test action verb extraction."""
        prompt = "a dog running quickly through the park"
        result = analyzer.analyze(prompt)
        
        assert "running" in result.main_actions
    
    def test_descriptor_extraction(self, analyzer):
        """Test descriptor extraction."""
        prompt = "a cinematic scene with very bright lighting"
        result = analyzer.analyze(prompt)
        
        # Should find style descriptors
        descriptors_lower = [d.lower() for d in result.descriptors]
        assert any("bright" in d or "cinematic" in d for d in descriptors_lower)
    
    def test_temporal_detection(self, analyzer):
        """Test temporal marker detection."""
        prompt = "a cat walking slowly while birds fly around"
        result = analyzer.analyze(prompt)
        
        assert result.has_temporal is True
    
    def test_spatial_detection(self, analyzer):
        """Test spatial marker detection."""
        prompt = "a person standing above the clouds, looking below"
        result = analyzer.analyze(prompt)
        
        assert result.has_spatial is True
    
    def test_style_detection(self, analyzer):
        """Test artistic style detection."""
        prompt = "a cinematic photorealistic scene of a landscape"
        result = analyzer.analyze(prompt)
        
        assert result.has_style is True
    
    def test_complexity_simple(self, analyzer):
        """Test complexity scoring for simple prompt."""
        prompt = "red ball"
        result = analyzer.analyze(prompt)
        
        # Simple prompt should have low complexity
        assert result.complexity < 0.5
    
    def test_complexity_complex(self, analyzer):
        """Test complexity scoring for complex prompt."""
        prompt = "a majestic cinematic scene of a phoenix rising from flames above mountains while birds fly around it, very detailed, surreal style"
        result = analyzer.analyze(prompt)
        
        # Complex prompt should have higher complexity
        assert result.complexity > 0.5
    
    def test_tokenization(self, analyzer):
        """Test tokenization process."""
        prompt = "walking cat"
        result = analyzer.analyze(prompt)
        
        # Should have tokens for both words
        texts = [t.text for t in result.tokens]
        assert "walking" in texts
        assert "cat" in texts
    
    def test_batch_analysis(self, analyzer):
        """Test batch prompt analysis."""
        prompts = [
            "a glowing cat",
            "a dog running",
            "surreal landscape",
        ]
        
        results = analyzer.analyze_batch(prompts)
        
        assert len(results) == 3
        assert all(isinstance(r, PromptAnalysisResult) for r in results)
    
    def test_keyword_emphasis(self, analyzer):
        """Test keyword emphasis computation."""
        prompt = "a beautiful red cat running through green grass"
        result = analyzer.analyze(prompt)
        
        emphasis = analyzer.get_keyword_emphasis(result)
        
        # Subjects should have high emphasis
        assert "cat" in emphasis
        assert emphasis["cat"] > 0.5


class TestPromptAnalysisResult:
    """Test PromptAnalysisResult dataclass."""
    
    def test_result_creation(self):
        """Test creating analysis result."""
        result = PromptAnalysisResult(
            original_prompt="a cat",
            tokens=[PromptToken("cat", 0, "noun", weight=1.6)],
            main_subjects=["cat"],
            main_actions=[],
            descriptors=[],
            objects=[],
        )
        
        assert result.original_prompt == "a cat"
        assert len(result.tokens) == 1
    
    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = PromptAnalysisResult(
            original_prompt="test",
            tokens=[],
            main_subjects=["test"],
            main_actions=[],
            descriptors=[],
            objects=[],
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict["original_prompt"] == "test"


class TestPromptIntegration:
    """Integration tests for prompt analysis."""
    
    def test_full_analysis_workflow(self):
        """Test complete analysis workflow."""
        analyzer = PromptAnalyzer()
        
        prompt = "A majestic phoenix rising from flames, cinematic lighting, surreal style"
        result = analyzer.analyze(prompt)
        
        # Check all components extracted
        assert len(result.tokens) > 0
        assert result.complexity > 0.4
        assert result.language == "en"
        
        # Should detect style
        assert result.has_style is True
        
        # Get emphasis scores
        emphasis = analyzer.get_keyword_emphasis(result)
        assert len(emphasis) > 0
    
    def test_multiple_subjects(self):
        """Test handling multiple subjects."""
        analyzer = PromptAnalyzer()
        
        prompt = "a cat and dog playing together"
        result = analyzer.analyze(prompt)
        
        # Should extract multiple subjects
        assert len(result.main_subjects) >= 2
    
    def test_complex_scene_description(self):
        """Test complex scene with many elements."""
        analyzer = PromptAnalyzer()
        
        prompt = "surreal abstract landscape with floating islands, waterfalls of light, glowing crystals, ethereal atmosphere, cinematic depth of field"
        result = analyzer.analyze(prompt)
        
        assert result.has_style is True
        assert result.complexity > 0.6
        assert len(result.descriptors) > 3
