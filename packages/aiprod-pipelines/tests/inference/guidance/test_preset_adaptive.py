"""
Integration tests for adaptive preset configurations.

Coverage:
  - All 5 adaptive preset modes
  - Node graph structure validation
  - Input/output connectivity
  - Configuration parameter passing
  - Adaptive guidance initialization
"""

import pytest
import torch

from aiprod_pipelines.inference.presets import preset, PresetFactory
from aiprod_pipelines.inference.guidance import AdaptiveGuidanceNode


class TestAdaptivePresetFactory:
    """Tests for PresetFactory adaptive methods."""
    
    def test_factory_methods_exist(self):
        """Test that all adaptive factory methods exist."""
        assert hasattr(PresetFactory, "t2v_one_stage_adaptive")
        assert hasattr(PresetFactory, "t2v_two_stages_adaptive")
        assert hasattr(PresetFactory, "distilled_fast_adaptive")
        assert hasattr(PresetFactory, "ic_lora_adaptive")
        assert hasattr(PresetFactory, "keyframe_interpolation_adaptive")


class TestT2VOneStageAdaptivePreset:
    """Tests for t2v_one_stage_adaptive preset."""
    
    def test_preset_creation(self, mock_denoise_model, mock_scheduler):
        """Test creating t2v_one_stage_adaptive preset."""
        graph = PresetFactory.t2v_one_stage_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
        )
        
        assert graph is not None
        assert graph.name == "t2v_one_stage_adaptive"
        assert len(graph.nodes) > 0
    
    def test_preset_nodes(self, mock_denoise_model, mock_scheduler):
        """Test preset contains expected nodes."""
        graph = PresetFactory.t2v_one_stage_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
        )
        
        node_ids = list(graph.nodes.keys())
        
        assert "encode" in node_ids
        assert "denoise_adaptive" in node_ids
        assert "decode" in node_ids
        assert "cleanup" in node_ids
    
    def test_denoise_node_is_adaptive(self, mock_denoise_model, mock_scheduler):
        """Test that denoise node is AdaptiveGuidanceNode."""
        graph = PresetFactory.t2v_one_stage_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
        )
        
        denoise_node = graph.nodes["denoise_adaptive"]
        
        assert isinstance(denoise_node, AdaptiveGuidanceNode)
        assert denoise_node.profile.enable_prompt_analysis is True
        assert denoise_node.profile.enable_timestep_scaling is True
        assert denoise_node.profile.enable_quality_adjustment is True
        assert denoise_node.profile.enable_early_exit is True
    
    def test_preset_with_config_override(self, mock_denoise_model, mock_scheduler):
        """Test preset with custom configuration."""
        graph = PresetFactory.t2v_one_stage_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
            num_inference_steps=50,
            enable_early_exit=False,
        )
        
        denoise_node = graph.nodes["denoise_adaptive"]
        
        # Config should be applied
        assert denoise_node.num_steps == 50
        assert denoise_node.profile.enable_early_exit is False
    
    def test_preset_connectivity(self, mock_denoise_model, mock_scheduler):
        """Test node connectivity."""
        graph = PresetFactory.t2v_one_stage_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
        )
        
        # Check edges
        assert graph.edges is not None
        assert len(graph.edges) > 0
        
        # Should have encode → denoise → decode → cleanup
        edge_list = [(e[0], e[1]) for e in graph.edges]
        assert ("encode", "denoise_adaptive") in edge_list
        assert ("denoise_adaptive", "decode") in edge_list


class TestT2VTwoStagesAdaptivePreset:
    """Tests for t2v_two_stages_adaptive preset."""
    
    def test_preset_creation(self, mock_denoise_model, mock_scheduler):
        """Test creating t2v_two_stages_adaptive preset."""
        graph = PresetFactory.t2v_two_stages_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            upsampler=mock_denoise_model,
            vae_decoder=mock_denoise_model,
        )
        
        assert graph is not None
        assert graph.name == "t2v_two_stages_adaptive"
    
    def test_preset_two_stage_nodes(self, mock_denoise_model, mock_scheduler):
        """Test two-stage adaptive preset has both stages."""
        graph = PresetFactory.t2v_two_stages_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            upsampler=mock_denoise_model,
            vae_decoder=mock_denoise_model,
        )
        
        node_ids = list(graph.nodes.keys())
        
        # Stage 1
        assert "denoise_stage1_adaptive" in node_ids
        assert "decode_stage1" in node_ids
        
        # Stage 2
        assert "denoise_stage2_adaptive" in node_ids
        assert "decode_stage2" in node_ids
        
        # Upsampler
        assert "upsample" in node_ids
    
    def test_both_stages_adaptive(self, mock_denoise_model, mock_scheduler):
        """Test both stages use adaptive denoising."""
        graph = PresetFactory.t2v_two_stages_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            upsampler=mock_denoise_model,
            vae_decoder=mock_denoise_model,
        )
        
        stage1 = graph.nodes["denoise_stage1_adaptive"]
        stage2 = graph.nodes["denoise_stage2_adaptive"]
        
        assert isinstance(stage1, AdaptiveGuidanceNode)
        assert isinstance(stage2, AdaptiveGuidanceNode)
    
    def test_two_stage_connectivity(self, mock_denoise_model, mock_scheduler):
        """Test two-stage connectivity."""
        graph = PresetFactory.t2v_two_stages_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            upsampler=mock_denoise_model,
            vae_decoder=mock_denoise_model,
        )
        
        edge_list = [(e[0], e[1]) for e in graph.edges]
        
        # Should flow: encode → denoise_stage1 → decode1 → upsample → denoise_stage2 → decode2
        assert ("encode", "denoise_stage1_adaptive") in edge_list
        assert ("decode_stage1", "upsample") in edge_list
        assert ("upsample", "denoise_stage2_adaptive") in edge_list


class TestDistilledFastAdaptivePreset:
    """Tests for distilled_fast_adaptive preset."""
    
    def test_preset_creation(self, mock_denoise_model, mock_scheduler):
        """Test creating distilled_fast_adaptive preset."""
        graph = PresetFactory.distilled_fast_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
        )
        
        assert graph is not None
        assert graph.name == "distilled_fast_adaptive"
    
    def test_fast_adaptive_config(self, mock_denoise_model, mock_scheduler):
        """Test fast preset has aggressive early exit settings."""
        graph = PresetFactory.distilled_fast_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
        )
        
        denoise_node = graph.nodes["denoise_adaptive"]
        
        # Should have minimal steps before early exit
        assert denoise_node.profile.min_steps == 2
        assert denoise_node.num_steps == 4  # Ultra-fast


class TestICLoRAAdaptivePreset:
    """Tests for ic_lora_adaptive preset."""
    
    def test_preset_creation(self, mock_denoise_model, mock_scheduler):
        """Test creating ic_lora_adaptive preset."""
        graph = PresetFactory.ic_lora_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
        )
        
        assert graph is not None
        assert graph.name == "ic_lora_adaptive"
    
    def test_ic_lora_with_loras(self, mock_denoise_model, mock_scheduler):
        """Test ic_lora_adaptive with LoRA paths."""
        loras = [
            ("path/to/lora1.pt", 0.7),
            ("path/to/lora2.pt", 0.5),
        ]
        
        graph = PresetFactory.ic_lora_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
            loras=loras,
        )
        
        assert graph is not None


class TestKeyframeAdaptivePreset:
    """Tests for keyframe_interpolation_adaptive preset."""
    
    def test_preset_creation(self, mock_denoise_model, mock_scheduler):
        """Test creating keyframe_interpolation_adaptive preset."""
        graph = PresetFactory.keyframe_interpolation_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
        )
        
        assert graph is not None
        assert graph.name == "keyframe_interpolation_adaptive"


class TestPresetFactoryFunction:
    """Tests for preset() factory function."""
    
    def test_preset_function_adaptive_modes(self, mock_denoise_model, mock_scheduler):
        """Test preset() function supports all adaptive modes."""
        modes = [
            "t2v_one_stage_adaptive",
            "distilled_fast_adaptive",
            "ic_lora_adaptive",
            "keyframe_adaptive",
        ]
        
        for mode in modes:
            graph = preset(
                mode,
                text_encoder=mock_denoise_model,
                model=mock_denoise_model,
                scheduler=mock_scheduler,
                vae_decoder=mock_denoise_model,
            )
            
            assert graph is not None
            assert "adaptive" in graph.name.lower()
    
    def test_preset_adaptive_two_stages_with_upsampler(self, mock_denoise_model, mock_scheduler):
        """Test t2v_two_stages_adaptive mode requires upsampler."""
        graph = preset(
            "t2v_two_stages_adaptive",
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
            upsampler=mock_denoise_model,
        )
        
        assert graph is not None
    
    def test_preset_adaptive_two_stages_without_upsampler_fails(self, mock_denoise_model, mock_scheduler):
        """Test t2v_two_stages_adaptive fails without upsampler."""
        with pytest.raises(ValueError, match="upsampler required"):
            preset(
                "t2v_two_stages_adaptive",
                text_encoder=mock_denoise_model,
                model=mock_denoise_model,
                scheduler=mock_scheduler,
                vae_decoder=mock_denoise_model,
                upsampler=None,
            )
    
    def test_preset_invalid_mode(self, mock_denoise_model, mock_scheduler):
        """Test preset() with invalid mode raises error."""
        with pytest.raises(ValueError, match="Unknown preset mode"):
            preset(
                "invalid_mode",
                text_encoder=mock_denoise_model,
                model=mock_denoise_model,
                scheduler=mock_scheduler,
                vae_decoder=mock_denoise_model,
            )
    
    def test_preset_backward_compatibility(self, mock_denoise_model, mock_scheduler):
        """Test standard modes still work."""
        # Standard modes should still work
        graph = preset(
            "t2v_one_stage",
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
        )
        
        assert graph is not None
        assert "denoise" in [n for n in graph.nodes.keys()]


class TestAdaptivePresetDocumentation:
    """Tests to verify preset documentation clarity."""
    
    def test_preset_function_has_detailed_docstring(self):
        """Test that preset() function has comprehensive docstring."""
        assert preset.__doc__ is not None
        assert "Adaptive" in preset.__doc__
        assert "adaptive" in preset.__doc__.lower()
        assert "examples" in preset.__doc__.lower()
    
    def test_adaptive_factory_methods_documented(self):
        """Test that adaptive factory methods have docstrings."""
        assert PresetFactory.t2v_one_stage_adaptive.__doc__ is not None
        assert "Adaptive" in PresetFactory.t2v_one_stage_adaptive.__doc__
        assert "expected improvements" in PresetFactory.t2v_one_stage_adaptive.__doc__.lower()


class TestAdaptivePresetRobustness:
    """Robustness tests for adaptive presets."""
    
    def test_preset_with_empty_config_overrides(self, mock_denoise_model, mock_scheduler):
        """Test preset works with empty overrides."""
        graph = PresetFactory.t2v_one_stage_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
        )
        
        assert graph is not None
    
    def test_preset_with_all_adaptive_disabled(self, mock_denoise_model, mock_scheduler):
        """Test preset where all adaptive components disabled."""
        graph = PresetFactory.t2v_one_stage_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
            enable_prompt_analysis=False,
            enable_timestep_scaling=False,
            enable_quality_adjustment=False,
            enable_early_exit=False,
        )
        
        denoise_node = graph.nodes["denoise_adaptive"]
        
        assert denoise_node.profile.enable_prompt_analysis is False
        assert denoise_node.profile.enable_timestep_scaling is False
        assert denoise_node.profile.enable_quality_adjustment is False
        assert denoise_node.profile.enable_early_exit is False
    
    def test_preset_with_pretrained_model_paths(self, mock_denoise_model, mock_scheduler):
        """Test preset accepts pretrained model paths."""
        graph = PresetFactory.t2v_one_stage_adaptive(
            text_encoder=mock_denoise_model,
            model=mock_denoise_model,
            scheduler=mock_scheduler,
            vae_decoder=mock_denoise_model,
            prompt_analyzer_path="models/prompt_analyzer.pt",
            quality_predictor_path="models/quality_predictor.pt",
        )
        
        denoise_node = graph.nodes["denoise_adaptive"]
        
        assert denoise_node.profile.prompt_analyzer_path == "models/prompt_analyzer.pt"
        assert denoise_node.profile.quality_predictor_path == "models/quality_predictor.pt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
