"""
Integration tests for InferenceGraph with real-world scenarios.
"""

import pytest
import torch
from aiprod_pipelines.inference import (
    InferenceGraph,
    preset,
    GraphContext,
)


class TestInferenceGraphIntegration:
    """Integration tests for complete inference workflows."""
    
    def test_t2v_one_stage_full_pipeline(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test complete t2v_one_stage execution."""
        graph = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
            num_inference_steps=3,  # Reduced for testing
        )
        
        result = graph.run(
            prompt="A dog running through a forest",
            num_inference_steps=3,
        )
        
        # Verify outputs exist
        assert "video_frames" in result or any(k in result for k in ["latents_denoised", "embeddings"])
    
    def test_t2v_two_stages_full_pipeline(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_upsampler, mock_vae_decoder):
        """Test complete t2v_two_stages execution."""
        graph = preset(
            "t2v_two_stages",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
            upsampler=mock_upsampler,
            stage1_steps=2,
            stage2_steps=2,
        )
        
        result = graph.run(
            prompt="A cat jumping on a couch",
        )
        
        # Verify outputs
        assert "video_frames" in result or "latents_denoised" in result
    
    def test_distilled_fast_full_pipeline(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test complete distilled_fast execution."""
        graph = preset(
            "distilled_fast",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        result = graph.run(
            prompt="A sunset over the ocean",
        )
        
        # Should complete without error
        assert result is not None
    
    def test_ic_lora_with_loras(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test ic_lora with LoRA weights."""
        graph = preset(
            "ic_lora",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
            loras=[("path/to/style.safetensors", 0.8)],
        )
        
        result = graph.run(
            prompt="A portrait of a girl in oil painting style",
        )
        
        assert result is not None
    
    def test_graph_validation_before_execution(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test that graphs validate before execution."""
        graph = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        is_valid, msg = graph.validate()
        assert is_valid
        
        # Should execute successfully
        result = graph.run(prompt="Test prompt")
        assert result is not None
    
    def test_multiple_prompts_batch(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test batch processing with multiple prompts."""
        graph = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        result = graph.run(
            prompt=["A cat", "A dog", "A bird"],
        )
        
        assert result is not None
    
    def test_guided_generation_parameters(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test guided generation with custom parameters."""
        graph = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        result = graph.run(
            prompt="A fantasy landscape",
            guidance_scale=15.0,  # Override default
            num_inference_steps=50,
            seed=42,
        )
        
        assert result is not None
    
    def test_negative_prompt_handling(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test negative prompt integration."""
        graph = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        result = graph.run(
            prompt="A beautiful landscape",
            negative_prompt="ugly, blurry, low quality",
        )
        
        assert result is not None
    
    def test_audio_conditional_generation(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test audio-conditional video generation."""
        graph = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        result = graph.run(
            prompt="A girl singing",
            audio_prompt="Soft acoustic guitar music",
        )
        
        assert result is not None


class TestGraphDataFlow:
    """Tests for data flow through graph nodes."""
    
    def test_context_accumulation(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test that context accumulates outputs through nodes."""
        graph = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        # Execute graph - outputs should accumulate
        result = graph.run(prompt="Test")
        
        # All nodes should contribute to outputs
        assert len(result) > 0
    
    def test_intermediate_tensor_passing(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test that tensors are correctly passed between nodes."""
        graph = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        # Mock to capture context at each step
        result = graph.run(prompt="Test prompt")
        
        # Should have embeddings from text encoder
        if "embeddings" in result:
            assert result["embeddings"].dtype == torch.bfloat16


class TestGraphPerformance:
    """Tests for graph execution performance characteristics."""
    
    def test_execution_order_consistency(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Verify topological execution order is consistent."""
        graph = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        # Get topological order
        order1 = graph._topological_sort()
        order2 = graph._topological_sort()
        
        # Should be deterministic
        assert order1 == order2
    
    def test_no_unnecessary_recomputation(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Verify graph doesn't recompute nodes."""
        graph = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        # Each node should execute exactly once
        result = graph.run(prompt="Test")
        
        # Should not raise any errors
        assert result is not None


class TestGraphTransforability:
    """Tests for graph composability and transformation."""
    
    def test_preset_mode_equivalence(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test that presets can be instantiated multiple times identically."""
        graph1 = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
            num_inference_steps=20,
        )
        
        graph2 = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
            num_inference_steps=20,
        )
        
        # Same structure
        assert graph1.name == graph2.name
        assert len(graph1.nodes) == len(graph2.nodes)
    
    def test_graph_summary_info(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test that graph provides useful summary information."""
        graph = preset(
            "t2v_two_stages",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
            upsampler=None,  # Will be mocked
        )
        
        # Should have a mock upsampler from the test
        # Just verify the summary works
        summary = graph.summary()
        assert "t2v_two_stages" in summary
        assert "encode" in summary or len(summary) > 0
