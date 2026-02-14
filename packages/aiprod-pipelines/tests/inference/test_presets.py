"""
Tests for preset inference graph configurations.
"""

import pytest
from aiprod_pipelines.inference import (
    InferenceGraph,
    PresetFactory,
    preset,
)


class TestPresetFactory:
    """Tests for PresetFactory.
    
    Verifies that all preset configurations create valid graphs
    with correct node structure.
    """
    
    def test_t2v_one_stage_creation(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test t2v_one_stage preset creation."""
        graph = PresetFactory.t2v_one_stage(
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        assert isinstance(graph, InferenceGraph)
        assert graph.name == "t2v_one_stage"
        assert "encode" in graph.nodes
        assert "denoise" in graph.nodes
        assert "decode" in graph.nodes
        assert "cleanup" in graph.nodes
    
    def test_t2v_one_stage_connections(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test t2v_one_stage connection structure."""
        graph = PresetFactory.t2v_one_stage(
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        # Should be linear: encode → denoise → decode → cleanup
        assert "denoise" in graph.edges["encode"]
        assert "decode" in graph.edges["denoise"]
        assert "cleanup" in graph.edges["decode"]
    
    def test_t2v_one_stage_config_override(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test t2v_one_stage with config overrides."""
        graph = PresetFactory.t2v_one_stage(
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
            num_inference_steps=50,
            guidance_scale=10.0,
        )
        
        denoise_node = graph.nodes["denoise"]
        assert denoise_node.num_steps == 50
        assert denoise_node.guidance_scale == 10.0
    
    def test_t2v_two_stages_creation(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_upsampler, mock_vae_decoder):
        """Test t2v_two_stages preset creation."""
        graph = PresetFactory.t2v_two_stages(
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_upsampler,
            mock_vae_decoder,
        )
        
        assert isinstance(graph, InferenceGraph)
        assert graph.name == "t2v_two_stages"
        
        # Check all required nodes
        required_nodes = ["encode", "denoise_stage1", "decode_stage1", "upsample", "denoise_stage2", "decode_stage2", "cleanup"]
        for node_id in required_nodes:
            assert node_id in graph.nodes
    
    def test_t2v_two_stages_stage_configs(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_upsampler, mock_vae_decoder):
        """Test t2v_two_stages stage-specific configs."""
        graph = PresetFactory.t2v_two_stages(
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_upsampler,
            mock_vae_decoder,
            stage1_steps=10,
            stage2_steps=5,
        )
        
        assert graph.nodes["denoise_stage1"].num_steps == 10
        assert graph.nodes["denoise_stage2"].num_steps == 5
    
    def test_t2v_two_stages_reduced_stage2_guidance(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_upsampler, mock_vae_decoder):
        """Test that stage 2 has reduced guidance."""
        graph = PresetFactory.t2v_two_stages(
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_upsampler,
            mock_vae_decoder,
            guidance_scale=7.5,
        )
        
        # Stage 2 should have lower guidance (50% of stage 1)
        stage1_guidance = graph.nodes["denoise_stage1"].guidance_scale
        stage2_guidance = graph.nodes["denoise_stage2"].guidance_scale
        
        assert stage2_guidance < stage1_guidance
        assert abs(stage2_guidance - stage1_guidance * 0.5) < 0.01
    
    def test_distilled_fast_creation(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test distilled_fast preset creation."""
        graph = PresetFactory.distilled_fast(
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        assert isinstance(graph, InferenceGraph)
        assert graph.name == "distilled_fast"
        assert "encode" in graph.nodes
        assert "denoise" in graph.nodes
        assert "decode" in graph.nodes
    
    def test_distilled_fast_optimized_config(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test distilled_fast has speed optimizations."""
        graph = PresetFactory.distilled_fast(
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        denoise_node = graph.nodes["denoise"]
        # Fast distilled: very few steps and low guidance
        assert denoise_node.num_steps == 4
        assert denoise_node.guidance_scale == 1.0
    
    def test_ic_lora_creation(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test ic_lora preset creation."""
        graph = PresetFactory.ic_lora(
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        assert isinstance(graph, InferenceGraph)
        assert graph.name == "ic_lora"
        assert "encode" in graph.nodes
        assert "denoise" in graph.nodes
        assert "decode" in graph.nodes
    
    def test_ic_lora_with_loras(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test ic_lora with LoRA configuration."""
        loras = [
            ("path/to/lora1.safetensors", 1.0),
            ("path/to/lora2.safetensors", 0.5),
        ]
        
        graph = PresetFactory.ic_lora(
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
            loras=loras,
        )
        
        denoise_node = graph.nodes["denoise"]
        assert denoise_node.loras == loras
    
    def test_keyframe_interpolation_creation(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test keyframe_interpolation preset creation."""
        graph = PresetFactory.keyframe_interpolation(
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        assert isinstance(graph, InferenceGraph)
        assert graph.name == "keyframe_interpolation"
    
    def test_keyframe_interpolation_smooth_guidance(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test keyframe has reduced guidance for smooth transitions."""
        graph = PresetFactory.keyframe_interpolation(
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        # Lower guidance for smooth transitions
        assert graph.nodes["denoise"].guidance_scale == 5.0
    
    def test_custom_graph_creation(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test custom graph creation."""
        from aiprod_pipelines.inference import TextEncodeNode, DenoiseNode, DecodeVideoNode
        
        nodes = {
            "encode": TextEncodeNode(mock_text_encoder),
            "denoise": DenoiseNode(mock_denoising_model, mock_scheduler),
            "decode": DecodeVideoNode(mock_vae_decoder),
        }
        edges = [
            ("encode", "denoise"),
            ("denoise", "decode"),
        ]
        
        graph = PresetFactory.custom(nodes, edges, name="custom_pipeline")
        
        assert graph.name == "custom_pipeline"
        assert len(graph.nodes) == 3
        assert "decode" in graph.edges["denoise"]


class TestPresetFactory:
    """Tests for preset() factory function."""
    
    def test_preset_t2v_one_stage(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test preset function for t2v_one_stage."""
        graph = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        assert graph.name == "t2v_one_stage"
    
    def test_preset_t2v_two_stages(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_upsampler, mock_vae_decoder):
        """Test preset function for t2v_two_stages."""
        graph = preset(
            "t2v_two_stages",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
            upsampler=mock_upsampler,
        )
        
        assert graph.name == "t2v_two_stages"
    
    def test_preset_t2v_two_stages_missing_upsampler(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test that t2v_two_stages requires upsampler."""
        with pytest.raises(ValueError, match="upsampler required"):
            preset(
                "t2v_two_stages",
                mock_text_encoder,
                mock_denoising_model,
                mock_scheduler,
                mock_vae_decoder,
            )
    
    def test_preset_distilled_fast(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test preset function for distilled_fast."""
        graph = preset(
            "distilled_fast",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        assert graph.name == "distilled_fast"
    
    def test_preset_ic_lora(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test preset function for ic_lora."""
        graph = preset(
            "ic_lora",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        assert graph.name == "ic_lora"
    
    def test_preset_keyframe(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test preset function for keyframe."""
        graph = preset(
            "keyframe",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
        )
        
        assert graph.name == "keyframe_interpolation"
    
    def test_preset_unknown_mode(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test that unknown mode raises error."""
        with pytest.raises(ValueError, match="Unknown preset mode"):
            preset(
                "unknown_mode",
                mock_text_encoder,
                mock_denoising_model,
                mock_scheduler,
                mock_vae_decoder,
            )
    
    def test_preset_config_override(self, mock_text_encoder, mock_denoising_model, mock_scheduler, mock_vae_decoder):
        """Test configuration overrides through preset function."""
        graph = preset(
            "t2v_one_stage",
            mock_text_encoder,
            mock_denoising_model,
            mock_scheduler,
            mock_vae_decoder,
            num_inference_steps=60,
            guidance_scale=12.0,
        )
        
        denoise = graph.nodes["denoise"]
        assert denoise.num_steps == 60
        assert denoise.guidance_scale == 12.0
