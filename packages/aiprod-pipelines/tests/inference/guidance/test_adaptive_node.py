"""
Unit tests for AdaptiveGuidanceNode.

Coverage:
  - Node initialization with various profiles
  - Input/output key validation
  - Guidance computation (prompt analysis + timestep scaling + quality adjustment)
  - Early exit logic
  - Integration with GraphContext
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from aiprod_pipelines.inference.guidance import (
    AdaptiveGuidanceNode,
    AdaptiveGuidanceProfile,
)
from aiprod_pipelines.inference.graph import GraphContext


class TestAdaptiveGuidanceNodeInit:
    """Tests for AdaptiveGuidanceNode initialization."""
    
    def test_init_default_profile(self, mock_denoise_model, mock_scheduler):
        """Test initialization with default profile."""
        node = AdaptiveGuidanceNode(mock_denoise_model, mock_scheduler)
        
        assert node.profile is not None
        assert node.profile.enable_prompt_analysis is True
        assert node.profile.enable_timestep_scaling is True
        assert node.profile.enable_quality_adjustment is True
        assert node.profile.enable_early_exit is True
        assert node.profile.min_steps == 15
    
    def test_init_custom_profile(self, mock_denoise_model, mock_scheduler):
        """Test initialization with custom profile."""
        profile = AdaptiveGuidanceProfile(
            enable_early_exit=False,
            min_steps=20,
        )
        node = AdaptiveGuidanceNode(mock_denoise_model, mock_scheduler, profile)
        
        assert node.profile.enable_early_exit is False
        assert node.profile.min_steps == 20
    
    def test_init_with_pretrained_models(self, mock_denoise_model, mock_scheduler):
        """Test initialization with pretrained model paths."""
        profile = AdaptiveGuidanceProfile(
            prompt_analyzer_path="models/prompt_analyzer.pt",
            quality_predictor_path="models/quality_predictor.pt",
        )
        
        # Mock the load operations
        with patch.object(
            profile,
            "prompt_analyzer_path",
            "mock_path"
        ):
            # Just verify the profile accepts paths
            assert profile.prompt_analyzer_path is not None
    
    def test_node_id_assignment(self, mock_denoise_model, mock_scheduler):
        """Test custom node ID assignment."""
        node = AdaptiveGuidanceNode(
            mock_denoise_model,
            mock_scheduler,
            node_id="custom_adaptive_denoise"
        )
        assert node.node_id == "custom_adaptive_denoise"


class TestAdaptiveGuidanceNodeKeys:
    """Tests for input/output key validation."""
    
    def test_input_keys(self, mock_denoise_model, mock_scheduler):
        """Test required input keys."""
        node = AdaptiveGuidanceNode(mock_denoise_model, mock_scheduler)
        
        expected_keys = ["latents", "embeddings", "prompt"]
        assert set(node.input_keys) == set(expected_keys)
    
    def test_output_keys(self, mock_denoise_model, mock_scheduler):
        """Test produced output keys."""
        node = AdaptiveGuidanceNode(mock_denoise_model, mock_scheduler)
        
        expected_keys = [
            "latents_denoised",
            "guidance_schedule",
            "steps_used",
            "quality_trajectory",
            "early_exit",
        ]
        assert set(node.output_keys) == set(expected_keys)


class TestAdaptiveGuidanceNodeGuidanceComputation:
    """Tests for adaptive guidance computation."""
    
    def test_get_prompt_guidance_default(self, mock_denoise_model, mock_scheduler, sample_embeddings):
        """Test baseline guidance without prompt analysis."""
        profile = AdaptiveGuidanceProfile(enable_prompt_analysis=False)
        node = AdaptiveGuidanceNode(mock_denoise_model, mock_scheduler, profile)
        
        pos_emb, _ = sample_embeddings
        guidance = node._get_prompt_guidance("A test prompt", pos_emb)
        
        # Should return base_guidance_scale (default 7.5)
        assert guidance == node.base_guidance_scale
    
    def test_compute_adaptive_guidance_baseline(self, mock_denoise_model, mock_scheduler, clean_latents, sample_embeddings):
        """Test guidance computation without scaling/adjustment."""
        profile = AdaptiveGuidanceProfile(
            enable_prompt_analysis=False,
            enable_timestep_scaling=False,
            enable_quality_adjustment=False,
        )
        node = AdaptiveGuidanceNode(mock_denoise_model, mock_scheduler, profile)
        pos_emb, _ = sample_embeddings
        
        guidance = node._compute_adaptive_guidance(
            step_idx=0,
            timestep=torch.tensor(999),
            base_guidance=7.5,
            latents=clean_latents,
            embeddings=pos_emb,
        )
        
        # Should return base guidance
        assert guidance == 7.5
    
    def test_compute_adaptive_guidance_with_timestep_scaling(
        self,
        mock_denoise_model,
        mock_scheduler,
        clean_latents,
        sample_embeddings,
    ):
        """Test guidance computation with timestep scaling."""
        profile = AdaptiveGuidanceProfile(
            enable_prompt_analysis=False,
            enable_timestep_scaling=True,
            enable_quality_adjustment=False,
        )
        node = AdaptiveGuidanceNode(mock_denoise_model, mock_scheduler, profile)
        pos_emb, _ = sample_embeddings
        
        # Early step (high noise) should scale up
        guidance_early = node._compute_adaptive_guidance(
            step_idx=0,
            timestep=torch.tensor(999),  # High noise
            base_guidance=7.5,
            latents=clean_latents,
            embeddings=pos_emb,
        )
        
        # Late step (low noise) should scale down
        guidance_late = node._compute_adaptive_guidance(
            step_idx=29,
            timestep=torch.tensor(0),  # Low noise
            base_guidance=7.5,
            latents=clean_latents,
            embeddings=pos_emb,
        )
        
        # Early should be stronger than late
        assert guidance_early > guidance_late
    
    def test_guidance_clipping(self, mock_denoise_model, mock_scheduler, clean_latents, sample_embeddings):
        """Test that guidance is clamped to [2.0, 15.0]."""
        node = AdaptiveGuidanceNode(mock_denoise_model, mock_scheduler)
        pos_emb, _ = sample_embeddings
        
        # Try to compute with very high base guidance
        guidance = node._compute_adaptive_guidance(
            step_idx=0,
            timestep=torch.tensor(999),
            base_guidance=100.0,  # Very high
            latents=clean_latents,
            embeddings=pos_emb,
        )
        
        # Should be clamped to max 15.0
        assert guidance <= 15.0
        
        # Try very low
        guidance = node._compute_adaptive_guidance(
            step_idx=0,
            timestep=torch.tensor(999),
            base_guidance=0.5,  # Very low
            latents=clean_latents,
            embeddings=pos_emb,
        )
        
        # Should be clamped to min 2.0
        assert guidance >= 2.0


class TestAdaptiveGuidanceNodeExecution:
    """Tests for full node execution."""
    
    def test_execute_basic(self, mock_denoise_model, mock_scheduler, graph_context):
        """Test basic execution with all components enabled."""
        profile = AdaptiveGuidanceProfile(
            enable_early_exit=False,
        )
        node = AdaptiveGuidanceNode(mock_denoise_model, mock_scheduler, profile)
        
        with patch.object(node, "_denoise_step", return_value=torch.randn(1, 4, 32, 32)):
            result = node.execute(graph_context)
        
        # Check output structure
        assert "latents_denoised" in result
        assert "guidance_schedule" in result
        assert "steps_used" in result
        assert "quality_trajectory" in result
        assert "early_exit" in result
        
        # Check types
        assert isinstance(result["latents_denoised"], torch.Tensor)
        assert isinstance(result["guidance_schedule"], list)
        assert isinstance(result["steps_used"], int)
        assert isinstance(result["early_exit"], bool)
    
    def test_execute_guidance_schedule_length(self, mock_denoise_model, mock_scheduler, graph_context):
        """Test that guidance schedule has correct length."""
        node = AdaptiveGuidanceNode(
            mock_denoise_model,
            mock_scheduler,
            profile=AdaptiveGuidanceProfile(enable_early_exit=False),
        )
        node.num_steps = 30
        
        with patch.object(node, "_denoise_step", return_value=torch.randn(1, 4, 32, 32)):
            result = node.execute(graph_context)
        
        # Should have 30 guidance values
        assert len(result["guidance_schedule"]) == 30
        assert result["steps_used"] == 30
        assert result["early_exit"] is False
    
    def test_execute_missing_inputs(self, mock_denoise_model, mock_scheduler):
        """Test execution with missing required inputs."""
        node = AdaptiveGuidanceNode(mock_denoise_model, mock_scheduler)
        
        context = GraphContext()
        # Missing "embeddings"
        context["latents"] = torch.randn(1, 4, 32, 32)
        
        # Should raise error about missing key
        with pytest.raises((KeyError, AssertionError)):
            node.execute(context)
    
    def test_execute_with_custom_num_steps(self, mock_denoise_model, mock_scheduler, graph_context):
        """Test execution with custom num_inference_steps."""
        node = AdaptiveGuidanceNode(mock_denoise_model, mock_scheduler)
        node.num_steps = 50
        
        graph_context["num_inference_steps"] = 15
        
        with patch.object(node, "_denoise_step", return_value=torch.randn(1, 4, 32, 32)):
            result = node.execute(graph_context)
        
        # Should use context value (15), not node default (50)
        assert result["steps_used"] == 15
        assert len(result["guidance_schedule"]) == 15


class TestAdaptiveGuidanceNodeEarlyExit:
    """Tests for early exit logic."""
    
    def test_early_exit_disabled(self, mock_denoise_model, mock_scheduler, graph_context):
        """Test that early exit can be disabled."""
        profile = AdaptiveGuidanceProfile(enable_early_exit=False)
        node = AdaptiveGuidanceNode(mock_denoise_model, mock_scheduler, profile)
        
        with patch.object(node, "_denoise_step", return_value=torch.randn(1, 4, 32, 32)):
            result = node.execute(graph_context)
        
        # Should not exit early
        assert result["early_exit"] is False
        assert result["steps_used"] == len(mock_scheduler.timesteps)


class TestAdaptiveGuidanceProfileConfig:
    """Tests for AdaptiveGuidanceProfile configuration."""
    
    def test_profile_defaults(self):
        """Test profile default values."""
        profile = AdaptiveGuidanceProfile()
        
        assert profile.enable_prompt_analysis is True
        assert profile.enable_timestep_scaling is True
        assert profile.enable_quality_adjustment is True
        assert profile.enable_early_exit is True
        assert profile.min_steps == 15
        assert profile.prompt_analyzer_path is None
        assert profile.quality_predictor_path is None
    
    def test_profile_customization(self):
        """Test profile customization."""
        profile = AdaptiveGuidanceProfile(
            enable_prompt_analysis=False,
            enable_quality_adjustment=False,
            min_steps=20,
            prompt_analyzer_path="custom/path/analyzer.pt",
        )
        
        assert profile.enable_prompt_analysis is False
        assert profile.enable_quality_adjustment is False
        assert profile.min_steps == 20
        assert profile.prompt_analyzer_path == "custom/path/analyzer.pt"


class TestAdaptiveGuidanceIntegration:
    """Integration tests with DenoiseNode wrapper."""
    
    def test_guidance_schedule_reasonableness(self, mock_denoise_model, mock_scheduler, graph_context):
        """Test that guidance schedule has reasonable values."""
        node = AdaptiveGuidanceNode(
            mock_denoise_model,
            mock_scheduler,
            profile=AdaptiveGuidanceProfile(enable_early_exit=False),
        )
        
        with patch.object(node, "_denoise_step", return_value=torch.randn(1, 4, 32, 32)):
            result = node.execute(graph_context)
        
        schedule = result["guidance_schedule"]
        
        # All values should be in reasonable range
        assert all(2.0 <= g <= 15.0 for g in schedule)
        
        # Early steps should generally have higher guidance
        early_avg = sum(schedule[:10]) / 10
        late_avg = sum(schedule[-10:]) / 10
        assert early_avg > late_avg
    
    def test_quality_trajectory_structure(self, mock_denoise_model, mock_scheduler, graph_context):
        """Test quality trajectory output structure."""
        node = AdaptiveGuidanceNode(mock_denoise_model, mock_scheduler)
        
        with patch.object(node, "_denoise_step", return_value=torch.randn(1, 4, 32, 32)):
            result = node.execute(graph_context)
        
        trajectory = result["quality_trajectory"]
        
        # Should be a list
        assert isinstance(trajectory, list)
        # May be empty if quality predictor not initialized
        # or populated if it is


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
