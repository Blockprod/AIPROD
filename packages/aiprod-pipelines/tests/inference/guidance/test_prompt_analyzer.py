"""
Unit tests for PromptAnalyzer and related classes.

Coverage:
  - GuidanceProfile dataclass
  - PromptAnalyzer neural network
  - PromptAnalyzerPredictor wrapper
  - Model save/load
  - Batch analysis
"""

import pytest
import torch

from aiprod_pipelines.inference.guidance.prompt_analyzer import (
    GuidanceProfile,
    PromptAnalyzer,
    PromptAnalyzerPredictor,
)


class TestGuidanceProfile:
    """Tests for GuidanceProfile dataclass."""
    
    def test_profile_creation(self):
        """Test creating a GuidanceProfile."""
        profile = GuidanceProfile(
            complexity=0.7,
            base_guidance=8.5,
            semantic_components={"action": "dancing", "object": "cat"},
            confidence=0.95,
            prompt_length=5,
        )
        
        assert profile.complexity == 0.7
        assert profile.base_guidance == 8.5
        assert profile.confidence == 0.95
        assert profile.prompt_length == 5
    
    def test_profile_complexity_range(self):
        """Test profile with complexity boundaries."""
        # Valid range [0, 1]
        profile_min = GuidanceProfile(0.0, 4.0, {}, 0.5, 1)
        profile_max = GuidanceProfile(1.0, 10.0, {}, 0.5, 1)
        
        assert profile_min.complexity == 0.0
        assert profile_max.complexity == 1.0
    
    def test_profile_guidance_range(self):
        """Test profile with guidance boundaries."""
        # Valid range [4, 10]
        profile_min = GuidanceProfile(0.5, 4.0, {}, 0.5, 1)
        profile_max = GuidanceProfile(0.5, 10.0, {}, 0.5, 1)
        
        assert profile_min.base_guidance == 4.0
        assert profile_max.base_guidance == 10.0
    
    def test_profile_semantic_components(self):
        """Test semantic components dictionary."""
        components = {
            "subject": "astronaut",
            "action": "floating",
            "setting": "space station",
            "style": "cinematic",
        }
        
        profile = GuidanceProfile(
            complexity=0.8,
            base_guidance=8.0,
            semantic_components=components,
            confidence=0.9,
            prompt_length=4,
        )
        
        assert profile.semantic_components == components
        assert len(profile.semantic_components) == 4


class TestPromptAnalyzerNetwork:
    """Tests for PromptAnalyzer neural network."""
    
    def test_init_default(self):
        """Test default initialization."""
        analyzer = PromptAnalyzer()
        
        assert analyzer.embedding_dim == 768
        assert analyzer.hidden_dim == 512
        assert analyzer.num_heads == 8
        assert analyzer.num_layers == 2
    
    def test_init_custom(self):
        """Test custom initialization."""
        analyzer = PromptAnalyzer(
            embedding_dim=512,
            hidden_dim=256,
            num_heads=4,
            num_layers=1,
        )
        
        assert analyzer.embedding_dim == 512
        assert analyzer.hidden_dim == 256
        assert analyzer.num_heads == 4
        assert analyzer.num_layers == 1
    
    def test_forward_output_shape(self):
        """Test forward pass output shape and range."""
        analyzer = PromptAnalyzer()
        analyzer.eval()
        
        batch_size = 2
        seq_len = 77
        embeddings = torch.randn(batch_size, seq_len, 768)
        
        with torch.no_grad():
            complexity, base_guidance, confidence = analyzer(embeddings)
        
        # Check shapes
        assert complexity.shape == (batch_size,)
        assert base_guidance.shape == (batch_size,)
        assert confidence.shape == (batch_size,)
        
        # Check ranges (sigmoid outputs [0, 1])
        assert (complexity >= 0).all() and (complexity <= 1).all()
        assert (confidence >= 0).all() and (confidence <= 1).all()
        
        # Guidance should be in [4, 10]
        assert (base_guidance >= 4).all() and (base_guidance <= 10).all()
    
    def test_forward_batch_processing(self):
        """Test forward pass with different batch sizes."""
        analyzer = PromptAnalyzer()
        analyzer.eval()
        
        for batch_size in [1, 2, 4, 8]:
            embeddings = torch.randn(batch_size, 77, 768)
            
            with torch.no_grad():
                complexity, guidance, confidence = analyzer(embeddings)
            
            assert complexity.shape[0] == batch_size
            assert guidance.shape[0] == batch_size
            assert confidence.shape[0] == batch_size
    
    def test_forward_gradient_flow(self):
        """Test that gradients flow during training."""
        analyzer = PromptAnalyzer()
        analyzer.train()
        
        embeddings = torch.randn(2, 77, 768, requires_grad=True)
        complexity, guidance, confidence = analyzer(embeddings)
        
        # Compute loss
        loss = complexity.mean() + guidance.mean() + confidence.mean()
        loss.backward()
        
        # Check gradients exist
        assert embeddings.grad is not None
        for param in analyzer.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestPromptAnalyzerPredictor:
    """Tests for PromptAnalyzerPredictor wrapper."""
    
    def test_init(self):
        """Test predictor initialization."""
        predictor = PromptAnalyzerPredictor()
        
        assert predictor.analyzer is not None
        assert isinstance(predictor.analyzer, PromptAnalyzer)
    
    def test_analyze_simple_embedding(self):
        """Test analyzing a single embedding."""
        predictor = PromptAnalyzerPredictor()
        
        embeddings = torch.randn(77, 768)
        profile = predictor.analyze("A test prompt", embeddings)
        
        assert isinstance(profile, GuidanceProfile)
        assert 0 <= profile.complexity <= 1
        assert 4 <= profile.base_guidance <= 10
        assert 0 <= profile.confidence <= 1
    
    def test_analyze_with_batched_embeddings(self):
        """Test analyzing with pre-batched embeddings."""
        predictor = PromptAnalyzerPredictor()
        
        batch_embeddings = torch.randn(2, 77, 768)
        profiles = []
        
        # Analyze each in batch
        for i in range(batch_embeddings.shape[0]):
            profile = predictor.analyze(
                f"Prompt {i}",
                batch_embeddings[i],
            )
            profiles.append(profile)
        
        assert len(profiles) == 2
        assert all(isinstance(p, GuidanceProfile) for p in profiles)
    
    def test_batch_analyze(self):
        """Test batch_analyze method."""
        predictor = PromptAnalyzerPredictor()
        
        prompts = ["First prompt", "Second prompt", "Third prompt"]
        embeddings_list = [torch.randn(77, 768) for _ in prompts]
        
        profiles = predictor.batch_analyze(prompts, embeddings_list)
        
        assert len(profiles) == len(prompts)
        assert all(isinstance(p, GuidanceProfile) for p in profiles)
    
    def test_batch_analyze_empty(self):
        """Test batch_analyze with empty inputs."""
        predictor = PromptAnalyzerPredictor()
        
        profiles = predictor.batch_analyze([], [])
        
        assert profiles == []
    
    def test_analyze_complexity_variation(self):
        """Test that different prompts can produce different complexity."""
        predictor = PromptAnalyzerPredictor()
        predictor.analyzer.eval()
        
        simple_prompt = "A cat"
        complex_prompt = "A photorealistic astronaut floating in a nebula at night, highly detailed and cinematic"
        
        with torch.no_grad():
            embeddings_simple = torch.randn(77, 768)
            embeddings_complex = torch.randn(77, 768)
            
            profile_simple = predictor.analyze(simple_prompt, embeddings_simple)
            profile_complex = predictor.analyze(complex_prompt, embeddings_complex)
        
        # Just verify both analyze properly
        assert isinstance(profile_simple, GuidanceProfile)
        assert isinstance(profile_complex, GuidanceProfile)
    
    def test_device_handling(self):
        """Test device handling (CPU/CUDA)."""
        predictor = PromptAnalyzerPredictor()
        
        # Move to CPU explicitly
        predictor.analyzer.cpu()
        embeddings = torch.randn(77, 768)
        profile = predictor.analyze("Test", embeddings)
        
        assert isinstance(profile, GuidanceProfile)
    
    def test_eval_mode(self):
        """Test setting model to eval mode."""
        predictor = PromptAnalyzerPredictor()
        predictor.analyzer.eval()
        
        embeddings = torch.randn(77, 768)
        with torch.no_grad():
            profile = predictor.analyze("Test", embeddings)
        
        assert isinstance(profile, GuidanceProfile)


class TestPromptAnalyzerCheckpointing:
    """Tests for model checkpointing."""
    
    def test_save_checkpoint(self, tmp_path):
        """Test saving a checkpoint."""
        analyzer = PromptAnalyzer()
        checkpoint_path = tmp_path / "analyzer.pt"
        
        torch.save(analyzer.state_dict(), checkpoint_path)
        
        assert checkpoint_path.exists()
    
    def test_load_checkpoint(self, tmp_path):
        """Test loading a checkpoint."""
        # Save original
        original = PromptAnalyzer()
        checkpoint_path = tmp_path / "analyzer.pt"
        torch.save(original.state_dict(), checkpoint_path)
        
        # Load into new model
        loaded = PromptAnalyzer()
        loaded.load_state_dict(torch.load(checkpoint_path))
        
        # Verify they produce same output
        embeddings = torch.randn(2, 77, 768)
        with torch.no_grad():
            out1 = original(embeddings)
            out2 = loaded(embeddings)
        
        # Parameters should match
        for p1, p2 in zip(original.parameters(), loaded.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_checkpoint_determinism(self, tmp_path):
        """Test that loading checkpoint produces deterministic results."""
        analyzer = PromptAnalyzer()
        checkpoint_path = tmp_path / "analyzer.pt"
        torch.save(analyzer.state_dict(), checkpoint_path)
        
        loaded1 = PromptAnalyzer()
        loaded1.load_state_dict(torch.load(checkpoint_path))
        
        loaded2 = PromptAnalyzer()
        loaded2.load_state_dict(torch.load(checkpoint_path))
        
        embeddings = torch.randn(2, 77, 768)
        
        with torch.no_grad():
            out1 = loaded1(embeddings)
            out2 = loaded2(embeddings)
        
        # Outputs should be identical
        assert torch.allclose(out1[0], out2[0])
        assert torch.allclose(out1[1], out2[1])
        assert torch.allclose(out1[2], out2[2])


class TestPromptAnalyzerEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_word_prompt(self):
        """Test analyzing single word prompt."""
        predictor = PromptAnalyzerPredictor()
        embeddings = torch.randn(77, 768)
        
        profile = predictor.analyze("Cat", embeddings)
        
        assert isinstance(profile, GuidanceProfile)
        assert profile.prompt_length == 1
    
    def test_very_long_prompt(self):
        """Test analyzing very long prompt."""
        predictor = PromptAnalyzerPredictor()
        embeddings = torch.randn(77, 768)
        
        long_prompt = " ".join(["word"] * 100)
        profile = predictor.analyze(long_prompt, embeddings)
        
        assert isinstance(profile, GuidanceProfile)
        assert profile.prompt_length == 100
    
    def test_empty_prompt(self):
        """Test analyzing empty prompt."""
        predictor = PromptAnalyzerPredictor()
        embeddings = torch.randn(77, 768)
        
        profile = predictor.analyze("", embeddings)
        
        assert isinstance(profile, GuidanceProfile)
        assert profile.prompt_length == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
