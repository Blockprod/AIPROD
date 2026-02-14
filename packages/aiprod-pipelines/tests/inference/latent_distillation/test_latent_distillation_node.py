"""
Unit tests for latent distillation nodes.

Coverage:
  - LatentDistillationNode initialization and modes
  - Compression mode execution
  - Decompression mode execution
  - Auto-detect mode
  - Metrics tracking and history
  - DistilledDenoiseNode interface
"""

import pytest
import torch

from aiprod_pipelines.inference.latent_distillation_node import (
    LatentDistillationNode,
    DistillationProfile,
    DistilledDenoiseNode,
)


class TestDistillationProfile:
    """Tests for DistillationProfile configuration."""
    
    def test_profile_defaults(self):
        """Test default profile."""
        profile = DistillationProfile()
        
        assert profile.enable_compression is True
        assert profile.enable_reconstruction_loss is True
        assert profile.codebook_size == 512
        assert profile.num_quantizers == 4
        assert profile.quality_target_percent == 95.0
    
    def test_profile_custom(self):
        """Test custom profile."""
        profile = DistillationProfile(
            enable_compression=False,
            codebook_size=256,
            num_quantizers=2,
            quality_target_percent=98.0,
        )
        
        assert profile.enable_compression is False
        assert profile.codebook_size == 256
        assert profile.num_quantizers == 2
        assert profile.quality_target_percent == 98.0


class TestLatentDistillationNodeInit:
    """Tests for LatentDistillationNode initialization."""
    
    def test_node_init_default(self):
        """Test node initialization with defaults."""
        node = LatentDistillationNode()
        
        assert node.profile is not None
        assert node.mode == "auto"
        assert node.engine is not None
        assert node.compression_history == []
    
    def test_node_init_compress_mode(self):
        """Test node in compress mode."""
        node = LatentDistillationNode(mode="compress")
        
        assert node.mode == "compress"
        assert "latents" in node.input_keys
        assert "compressed_codes" in node.output_keys
    
    def test_node_init_decompress_mode(self):
        """Test node in decompress mode."""
        node = LatentDistillationNode(mode="decompress")
        
        assert node.mode == "decompress"
        assert "compressed_codes" in node.input_keys
        assert "latents_decompressed" in node.output_keys
    
    def test_node_init_auto_mode(self):
        """Test node in auto mode."""
        node = LatentDistillationNode(mode="auto")
        
        assert node.mode == "auto"
        # Should accept either input type
        assert "latents" in node.input_keys or "compressed_codes" in node.input_keys
    
    def test_node_init_custom_profile(self):
        """Test node with custom profile."""
        profile = DistillationProfile(
            enable_compression=False,
            codebook_size=256,
        )
        node = LatentDistillationNode(profile=profile)
        
        assert node.profile.enable_compression is False
        assert node.profile.codebook_size == 256


class TestLatentDistillationNodeKeys:
    """Tests for node input/output keys."""
    
    def test_compress_mode_keys(self):
        """Test compress mode input/output keys."""
        node = LatentDistillationNode(mode="compress")
        
        assert node.input_keys == ["latents"]
        assert "compressed_codes" in node.output_keys
        assert "compression_metrics" in node.output_keys
    
    def test_decompress_mode_keys(self):
        """Test decompress mode input/output keys."""
        node = LatentDistillationNode(mode="decompress")
        
        assert node.input_keys == ["compressed_codes"]
        assert "latents_decompressed" in node.output_keys
        assert "reconstruction_metrics" in node.output_keys
    
    def test_auto_mode_keys(self):
        """Test auto mode accepts both input types."""
        node = LatentDistillationNode(mode="auto")
        
        # Should have both possible inputs
        input_keys = node.input_keys
        assert ("latents" in input_keys or "compressed_codes" in input_keys)


class TestLatentDistillationNodeExecution:
    """Tests for node execution."""
    
    def test_compress_execution(self, graph_context_with_latents):
        """Test compression execution."""
        node = LatentDistillationNode(mode="compress")
        
        result = node.execute(graph_context_with_latents)
        
        assert "compressed_codes" in result
        assert "compression_metrics" in result
        
        codes = result["compressed_codes"]
        assert codes.shape[0] == 4  # num_quantizers default
    
    def test_decompress_execution(self, graph_context_with_codes):
        """Test decompression execution."""
        node = LatentDistillationNode(mode="decompress")
        
        result = node.execute(graph_context_with_codes)
        
        assert "latents_decompressed" in result
        assert "reconstruction_metrics" in result
        
        latents = result["latents_decompressed"]
        assert latents.shape == (2, 4, 32, 32)
    
    def test_auto_mode_compression(self, graph_context_with_latents):
        """Test auto mode detects compression."""
        node = LatentDistillationNode(mode="auto")
        
        result = node.execute(graph_context_with_latents)
        
        # Should produce compression output
        assert "compressed_codes" in result
    
    def test_auto_mode_decompression(self, graph_context_with_codes):
        """Test auto mode detects decompression."""
        node = LatentDistillationNode(mode="auto")
        
        result = node.execute(graph_context_with_codes)
        
        # Should produce decompression output
        assert "latents_decompressed" in result
    
    def test_compression_disabled(self, graph_context_with_latents):
        """Test compression can be disabled."""
        profile = DistillationProfile(enable_compression=False)
        node = LatentDistillationNode(mode="compress", profile=profile)
        
        result = node.execute(graph_context_with_latents)
        
        # Should pass through uncompressed
        codes = result["compressed_codes"]
        latents = graph_context_with_latents["latents"]
        
        # Pass-through mode: codes should be same as input
        assert codes.shape == latents.shape


class TestLatentDistillationNodeMetrics:
    """Tests for metrics tracking."""
    
    def test_compress_metrics_recorded(self, graph_context_with_latents):
        """Test metrics are recorded during compression."""
        node = LatentDistillationNode(mode="compress")
        
        result = node.execute(graph_context_with_latents)
        
        metrics = result["compression_metrics"]
        assert metrics is not None
        assert metrics.compression_ratio > 1.0
    
    def test_compression_history(self, graph_context_with_latents):
        """Test compression history is maintained."""
        node = LatentDistillationNode(mode="compress")
        
        # Execute multiple times
        for _ in range(3):
            node.execute(graph_context_with_latents)
        
        assert len(node.compression_history) == 3
    
    def test_get_compression_summary(self, graph_context_with_latents):
        """Test getting compression summary."""
        node = LatentDistillationNode(mode="compress")
        
        # Execute multiple times
        for _ in range(5):
            node.execute(graph_context_with_latents)
        
        summary = node.get_compression_summary()
        
        assert "num_compressions" in summary
        assert "avg_compression_ratio" in summary
        assert "avg_quality_retention" in summary
        assert "total_memory_saved_mb" in summary
        
        assert summary["num_compressions"] == 5
    
    def test_reset_history(self, graph_context_with_latents):
        """Test resetting compression history."""
        node = LatentDistillationNode(mode="compress")
        
        # Execute and accumulate history
        for _ in range(3):
            node.execute(graph_context_with_latents)
        
        assert len(node.compression_history) == 3
        
        # Reset
        node.reset_history()
        
        assert len(node.compression_history) == 0
        assert node.last_metrics is None


class TestDistilledDenoiseNode:
    """Tests for DistilledDenoiseNode."""
    
    def test_node_init(self):
        """Test distilled denoise node initialization."""
        node = DistilledDenoiseNode(
            model=None,
            scheduler=None,
            num_inference_steps=30,
            guidance_scale=7.5,
        )
        
        assert node.num_inference_steps == 30
        assert node.guidance_scale == 7.5
    
    def test_node_input_keys(self):
        """Test input keys for distilled denoising."""
        node = DistilledDenoiseNode(model=None, scheduler=None)
        
        # Should expect compressed codes, not full latents
        assert "compressed_codes_input" in node.input_keys
        assert "embeddings" in node.input_keys
    
    def test_node_output_keys(self):
        """Test output keys for distilled denoising."""
        node = DistilledDenoiseNode(model=None, scheduler=None)
        
        # Should output compressed codes
        assert "compressed_codes_denoised" in node.output_keys
        assert "steps_completed" in node.output_keys
    
    def test_node_execution_placeholder(self):
        """Test distilled denoise execution (placeholder)."""
        class MockGraphContext(dict):
            pass
        
        context = MockGraphContext()
        context["compressed_codes_input"] = torch.randint(0, 256, (2, 4, 32, 32))
        context["embeddings"] = torch.randn(2, 77, 768)
        
        node = DistilledDenoiseNode(
            model=None,
            scheduler=None,
            num_inference_steps=20,
        )
        
        result = node.execute(context)
        
        assert "compressed_codes_denoised" in result
        assert "steps_completed" in result
        assert result["steps_completed"] == 20


class TestLatentDistillationNodeIntegration:
    """Integration tests."""
    
    def test_full_compression_decompression_pipeline(self, sample_latents):
        """Test full compress → denoise → decompress pipeline."""
        # Create contexts
        class MockGraphContext(dict):
            pass
        
        # Stage 1: Compression
        compress_context = MockGraphContext()
        compress_context["latents"] = sample_latents
        
        compress_node = LatentDistillationNode(mode="compress")
        compress_result = compress_node.execute(compress_context)
        
        codes = compress_result["compressed_codes"]
        
        # Stage 2: Decompression
        decompress_context = MockGraphContext()
        decompress_context["compressed_codes"] = codes
        
        decompress_node = LatentDistillationNode(mode="decompress")
        decompress_result = decompress_node.execute(decompress_context)
        
        recon = decompress_result["latents_decompressed"]
        
        # Reconstruction should match original shape and be reasonable
        assert recon.shape == sample_latents.shape
        
        # Compute quality
        mse = torch.nn.functional.mse_loss(sample_latents, recon)
        assert mse < 1.0
    
    def test_node_on_various_latent_types(self, sample_latents, high_variance_latents, structured_latents):
        """Test node on various latent types."""
        node = LatentDistillationNode(mode="compress")
        
        for latents in [sample_latents, high_variance_latents, structured_latents]:
            class MockGraphContext(dict):
                pass
            
            context = MockGraphContext()
            context["latents"] = latents
            
            result = node.execute(context)
            
            metrics = result["compression_metrics"]
            # Should compress to reasonable ratio
            assert metrics.compression_ratio >= 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
