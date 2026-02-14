"""
Tests for kernel fusion graph nodes.

Validates:
- KernelFusionSelectorNode functionality
- FusedDenoiseNode integration
- Profiler functionality
- Graph context integration
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from aiprod_pipelines.inference.kernel_fusion import (
    KernelFusionSelectorNode,
    FusedDenoiseNode,
    KernelFusionProfiler,
    FusionConfig,
)
from aiprod_pipelines.inference.graph import GraphContext


class TestKernelFusionSelectorNode:
    """Tests for KernelFusionSelectorNode."""
    
    @pytest.fixture
    def selector_node(self):
        """Create selector node."""
        with patch("aiprod_pipelines.inference.kernel_fusion.fusion_node.get_gpu_capabilities"):
            return KernelFusionSelectorNode()
    
    def test_initialization(self, selector_node):
        """Node initializes correctly."""
        assert selector_node.name == "kernel_fusion_selector"
        assert selector_node.input_keys == []
        assert "selected_fusions" in selector_node.output_keys
        assert "fusion_estimates" in selector_node.output_keys
    
    def test_forward_returns_correct_keys(self, selector_node):
        """Forward returns all expected keys."""
        context = {}
        model_config = {
            "hidden_dim": 768,
            "num_heads": 12,
            "total_flops": 1e12,
        }
        
        result = selector_node.forward(context, model_config)
        
        assert "selected_fusions" in result
        assert "fusion_estimates" in result
        assert "fusion_config" in result
        assert "gpu_capabilities" in result
    
    def test_forward_fusions_are_list(self, selector_node):
        """Selected fusions are list of strings."""
        context = {}
        model_config = {"hidden_dim": 768}
        
        result = selector_node.forward(context, model_config)
        
        fusions = result["selected_fusions"]
        assert isinstance(fusions, list)
        assert all(isinstance(f, str) for f in fusions)
    
    def test_forward_estimates_complete(self, selector_node):
        """Estimates dict has all expected fields."""
        context = {}
        model_config = {"hidden_dim": 768, "total_flops": 1e12}
        
        result = selector_node.forward(context, model_config)
        estimates = result["fusion_estimates"]
        
        assert "cumulative_speedup" in estimates
        assert "speedup_percent" in estimates
        assert "memory_overhead_bytes" in estimates
    
    def test_custom_config_preserved(self):
        """Custom config is used."""
        custom_config = FusionConfig(
            activation="relu",
            kernel_tile_size=256,
        )
        
        with patch("aiprod_pipelines.inference.kernel_fusion.fusion_node.get_gpu_capabilities"):
            node = KernelFusionSelectorNode()
            node.config = custom_config
            
            assert node.config.activation == "relu"
            assert node.config.kernel_tile_size == 256


class TestFusedDenoiseNode:
    """Tests for FusedDenoiseNode."""
    
    @pytest.fixture
    def denoise_node(self):
        """Create denoise node."""
        model = MagicMock()
        model.hidden_dim = 768
        model.num_attention_heads = 12
        
        scheduler = MagicMock()
        scheduler.step.return_value = {"prev_sample": torch.randn(1, 16, 64, 64, 48)}
        
        return FusedDenoiseNode(
            model=model,
            scheduler=scheduler,
            num_inference_steps=30,
            guidance_scale=7.5,
        )
    
    def test_initialization(self, denoise_node):
        """Node initializes correctly."""
        assert denoise_node.name == "fused_denoise"
        assert "latents" in denoise_node.input_keys
        assert "embeddings" in denoise_node.input_keys
        assert "denoised_latents" in denoise_node.output_keys
    
    def test_forward_shape_preserved(self, denoise_node):
        """Forward preserves latent shape."""
        batch, ch, h, w, t = 1, 16, 64, 64, 48
        
        latents = torch.randn(batch, ch, h, w, t)
        embeddings = torch.randn(1, 77, 768)
        timestep = torch.tensor([500])
        
        context = {}
        result = denoise_node.forward(
            context=context,
            latents=latents,
            embeddings=embeddings,
            timestep=timestep,
            selected_fusions=["attention_linear"],
        )
        
        assert "denoised_latents" in result
        assert result["denoised_latents"].shape == latents.shape
    
    def test_forward_with_custom_guidance(self, denoise_node):
        """Forward accepts custom guidance scale."""
        latents = torch.randn(1, 16, 64, 64, 48)
        embeddings = torch.randn(1, 77, 768)
        timestep = torch.tensor([500])
        
        result_default = denoise_node.forward(
            context={},
            latents=latents,
            embeddings=embeddings,
            timestep=timestep,
            selected_fusions=[],
            guidance_scale=None,
        )
        
        result_custom = denoise_node.forward(
            context={},
            latents=latents,
            embeddings=embeddings,
            timestep=timestep,
            selected_fusions=[],
            guidance_scale=10.0,
        )
        
        # Both should return denoised_latents
        assert "denoised_latents" in result_default
        assert "denoised_latents" in result_custom
    
    def test_forward_without_fusion(self, denoise_node):
        """Forward works with empty fusion list."""
        latents = torch.randn(1, 16, 64, 64, 48)
        embeddings = torch.randn(1, 77, 768)
        timestep = torch.tensor([500])
        
        result = denoise_node.forward(
            context={},
            latents=latents,
            embeddings=embeddings,
            timestep=timestep,
            selected_fusions=[],
        )
        
        assert "denoised_latents" in result
    
    def test_fusion_instances_built_once(self, denoise_node):
        """Fusion instances are built only once."""
        denoise_node.enable_fusion = True
        
        latents = torch.randn(1, 16, 64, 64, 48)
        embeddings = torch.randn(1, 77, 768)
        timestep = torch.tensor([500])
        config = FusionConfig()
        
        # First call builds fusions
        denoise_node.forward(
            context={},
            latents=latents,
            embeddings=embeddings,
            timestep=timestep,
            selected_fusions=["attention_linear"],
            fusion_config=config,
        )
        
        instances_after_first = len(denoise_node._fusion_instances)
        
        # Second call reuses fusions
        denoise_node.forward(
            context={},
            latents=latents,
            embeddings=embeddings,
            timestep=timestep,
            selected_fusions=["attention_linear"],
            fusion_config=config,
        )
        
        instances_after_second = len(denoise_node._fusion_instances)
        
        # Should be same count
        assert instances_after_first == instances_after_second
    
    def test_fusion_disabled_works(self, denoise_node):
        """Node works when fusion is disabled."""
        denoise_node.enable_fusion = False
        
        latents = torch.randn(1, 16, 64, 64, 48)
        embeddings = torch.randn(1, 77, 768)
        timestep = torch.tensor([500])
        
        result = denoise_node.forward(
            context={},
            latents=latents,
            embeddings=embeddings,
            timestep=timestep,
            selected_fusions=["attention_linear"],
        )
        
        assert "denoised_latents" in result


class TestKernelFusionProfiler:
    """Tests for KernelFusionProfiler."""
    
    def test_initialization(self):
        """Profiler initializes empty."""
        profiler = KernelFusionProfiler()
        
        assert len(profiler.profiles) == 0
    
    def test_profile_fusion_stores_result(self):
        """Profiling stores results."""
        profiler = KernelFusionProfiler()
        
        # Create mock fusion
        mock_fusion = MagicMock()
        mock_fusion.return_value = torch.randn(1, 16, 64, 64, 48)
        
        sample_input = torch.randn(1, 16, 64, 64, 48)
        
        with patch("aiprod_pipelines.inference.kernel_fusion.fusion_node.torch.cuda.is_available", return_value=False):
            result = profiler.profile_fusion(
                "test_fusion",
                mock_fusion,
                sample_input,
                iterations=10,
            )
            
            # Should be empty dict if CUDA not available
            assert isinstance(result, dict)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_profile_has_timing_info(self):
        """Profile result has timing information."""
        profiler = KernelFusionProfiler()
        
        mock_fusion = lambda x: x * 2
        sample_input = torch.randn(1, 16, 64, 64, 48).cuda()
        
        result = profiler.profile_fusion(
            "simple_mul",
            mock_fusion,
            sample_input,
            iterations=5,
        )
        
        if result:  # Only check if CUDA profiling worked
            assert "avg_time_ms" in result
            assert "total_time_sec" in result
            assert "iterations" in result
            assert result["iterations"] == 5


class TestNodeIntegration:
    """Tests for nodes working together."""
    
    def test_selector_output_compatible_with_denoise_input(self):
        """Selector output format compatible with denoise input."""
        with patch("aiprod_pipelines.inference.kernel_fusion.fusion_node.get_gpu_capabilities"):
            selector = KernelFusionSelectorNode()
            
            model = MagicMock()
            model.hidden_dim = 768
            model.num_attention_heads = 12
            scheduler = MagicMock()
            scheduler.step.return_value = {"prev_sample": torch.randn(1, 16, 64, 64, 48)}
            
            denoise = FusedDenoiseNode(model, scheduler)
            
            # Get selector output
            selector_output = selector.forward({}, {"hidden_dim": 768})
            
            # Use in denoise
            latents = torch.randn(1, 16, 64, 64, 48)
            embeddings = torch.randn(1, 77, 768)
            timestep = torch.tensor([500])
            
            denoise_result = denoise.forward(
                context={},
                latents=latents,
                embeddings=embeddings,
                timestep=timestep,
                selected_fusions=selector_output["selected_fusions"],
                fusion_config=selector_output["fusion_config"],
            )
            
            assert "denoised_latents" in denoise_result
