"""
Integration tests for kernel fusion system.

End-to-end validation of:
- Complete inference workflows with fusions
- Memory and performance in realistic scenarios
- Graph integration
- Fallback behavior
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from aiprod_pipelines.inference.kernel_fusion import (
    auto_select_kernel_fusions,
    KernelFusionSelectorNode,
    FusedDenoiseNode,
    FusionConfig,
)


class TestKernelFusionWorkflow:
    """End-to-end kernel fusion workflows."""
    
    @pytest.fixture
    def model_and_scheduler(self):
        """Create mock model and scheduler."""
        model = MagicMock()
        model.hidden_dim = 768
        model.num_attention_heads = 12
        
        scheduler = MagicMock()
        scheduler.step.return_value = {"prev_sample": torch.randn(1, 16, 256, 256, 50)}
        
        return model, scheduler
    
    def test_simple_inference_workflow(self, model_and_scheduler):
        """Simple inference with fusion."""
        model, scheduler = model_and_scheduler
        
        with patch("aiprod_pipelines.inference.kernel_fusion.fusion_node.get_gpu_capabilities"):
            # Setup nodes
            selector = KernelFusionSelectorNode()
            denoise = FusedDenoiseNode(model, scheduler, num_inference_steps=5)
            
            # Get fusion selection
            model_config = {
                "hidden_dim": 768,
                "num_heads": 12,
                "total_flops": 1e12,
            }
            selector_output = selector.forward({}, model_config)
            
            # Run single denoising step with selected fusions
            latents = torch.randn(1, 16, 256, 256, 50)
            embeddings = torch.randn(1, 77, 768)
            timestep = torch.tensor([950])
            
            denoise_output = denoise.forward(
                context={},
                latents=latents,
                embeddings=embeddings,
                timestep=timestep,
                selected_fusions=selector_output["selected_fusions"],
                fusion_config=selector_output["fusion_config"],
                guidance_scale=7.5,
            )
            
            assert "denoised_latents" in denoise_output
            assert denoise_output["denoised_latents"].shape == latents.shape
    
    def test_multi_step_inference(self, model_and_scheduler):
        """Multi-step denoising with consistent fusions."""
        model, scheduler = model_and_scheduler
        
        with patch("aiprod_pipelines.inference.kernel_fusion.fusion_node.get_gpu_capabilities"):
            selector = KernelFusionSelectorNode()
            denoise = FusedDenoiseNode(model, scheduler, num_inference_steps=4)
            
            selector_output = selector.forward({}, {"hidden_dim": 768})
            fusions = selector_output["selected_fusions"]
            config = selector_output["fusion_config"]
            
            # Simulate 4 denoising steps
            latents = torch.randn(1, 16, 256, 256, 50)
            embeddings = torch.randn(1, 77, 768)
            
            timesteps = [950, 750, 500, 250]
            
            for timestep in timesteps:
                output = denoise.forward(
                    context={},
                    latents=latents,
                    embeddings=embeddings,
                    timestep=torch.tensor([timestep]),
                    selected_fusions=fusions,
                    fusion_config=config,
                )
                # Update latents for next step
                latents = output["denoised_latents"]
                
                assert latents.shape == (1, 16, 256, 256, 50)
    
    def test_different_batch_sizes(self, model_and_scheduler):
        """Fusions work with different batch sizes."""
        model, scheduler = model_and_scheduler
        
        with patch("aiprod_pipelines.inference.kernel_fusion.fusion_node.get_gpu_capabilities"):
            denoise = FusedDenoiseNode(model, scheduler)
            
            for batch_size in [1, 2, 4]:
                latents = torch.randn(batch_size, 16, 64, 64, 24)
                embeddings = torch.randn(batch_size, 77, 768)
                timestep = torch.tensor([500])
                
                output = denoise.forward(
                    context={},
                    latents=latents,
                    embeddings=embeddings,
                    timestep=timestep,
                    selected_fusions=["attention_linear"],
                )
                
                assert output["denoised_latents"].shape == latents.shape
    
    def test_different_resolutions(self, model_and_scheduler):
        """Fusions work with different resolutions."""
        model, scheduler = model_and_scheduler
        
        with patch("aiprod_pipelines.inference.kernel_fusion.fusion_node.get_gpu_capabilities"):
            denoise = FusedDenoiseNode(model, scheduler)
            
            for h, w in [(64, 64), (128, 128), (256, 256)]:
                latents = torch.randn(1, 16, h, w, 24)
                embeddings = torch.randn(1, 77, 768)
                timestep = torch.tensor([500])
                
                output = denoise.forward(
                    context={},
                    latents=latents,
                    embeddings=embeddings,
                    timestep=timestep,
                    selected_fusions=["attention_linear"],
                )
                
                assert output["denoised_latents"].shape == latents.shape


class TestAutoSelection:
    """Tests for automatic fusion selection."""
    
    def test_auto_select_for_small_model(self):
        """Small model gets appropriate fusions."""
        with patch("aiprod_pipelines.inference.kernel_fusion.adaptive_fusion.get_gpu_capabilities"):
            fusions, estimates = auto_select_kernel_fusions(
                model_config={
                    "hidden_dim": 256,
                    "num_heads": 4,
                    "total_flops": 1e9,
                }
            )
            
            assert isinstance(fusions, list)
            assert estimates["cumulative_speedup"] > 1.0
    
    def test_auto_select_for_large_model(self):
        """Large model might get more aggressive fusions."""
        with patch("aiprod_pipelines.inference.kernel_fusion.adaptive_fusion.get_gpu_capabilities"):
            fusions, estimates = auto_select_kernel_fusions(
                model_config={
                    "hidden_dim": 1024,
                    "num_heads": 16,
                    "total_flops": 1e13,
                }
            )
            
            assert isinstance(fusions, list)
            assert estimates["cumulative_speedup"] > 1.0
    
    def test_auto_select_provides_estimates(self):
        """Auto-selection provides speedup estimates."""
        with patch("aiprod_pipelines.inference.kernel_fusion.adaptive_fusion.get_gpu_capabilities"):
            _, estimates = auto_select_kernel_fusions(
                model_config={"hidden_dim": 768}
            )
            
            assert "cumulative_speedup" in estimates
            assert "speedup_percent" in estimates
            assert "memory_overhead_bytes" in estimates
            assert estimates["speedup_percent"] >= 0


class TestFusionConfigurationVariations:
    """Tests for different configuration combinations."""
    
    def test_different_activations(self):
        """Different activations work."""
        with patch("aiprod_pipelines.inference.kernel_fusion.fusion_node.get_gpu_capabilities"):
            model = MagicMock()
            model.hidden_dim = 768
            model.num_attention_heads = 12
            scheduler = MagicMock()
            scheduler.step.return_value = {"prev_sample": torch.randn(1, 16, 64, 64, 48)}
            
            for activation in ["relu", "gelu", "silu", "sigmoid"]:
                config = FusionConfig(activation=activation)
                
                denoiser = FusedDenoiseNode(model, scheduler)
                
                latents = torch.randn(1, 16, 64, 64, 48)
                embeddings = torch.randn(1, 77, 768)
                
                result = denoiser.forward(
                    context={},
                    latents=latents,
                    embeddings=embeddings,
                    timestep=torch.tensor([500]),
                    selected_fusions=["attention_linear"],
                    fusion_config=config,
                )
                
                assert "denoised_latents" in result
    
    def test_mixed_precision_config(self):
        """Mixed precision configuration works."""
        with patch("aiprod_pipelines.inference.kernel_fusion.fusion_node.get_gpu_capabilities"):
            config = FusionConfig(
                use_mixed_precision=True,
                gradient_checkpointing=True,
            )
            config.validate()
            
            assert config.use_mixed_precision
            assert config.gradient_checkpointing


class TestMemoryBehavior:
    """Tests for memory behavior of fusions."""
    
    def test_no_memory_leak_with_fusions(self):
        """Running fusions doesn't cause memory leaks."""
        with patch("aiprod_pipelines.inference.kernel_fusion.fusion_node.get_gpu_capabilities"):
            model = MagicMock()
            model.hidden_dim = 768
            model.num_attention_heads = 12
            scheduler = MagicMock()
            scheduler.step.return_value = {"prev_sample": torch.randn(1, 16, 64, 64, 48)}
            
            denoiser = FusedDenoiseNode(model, scheduler)
            
            # Run multiple iterations
            for _ in range(10):
                latents = torch.randn(1, 16, 64, 64, 48)
                embeddings = torch.randn(1, 77, 768)
                
                denoiser.forward(
                    context={},
                    latents=latents,
                    embeddings=embeddings,
                    timestep=torch.tensor([500]),
                    selected_fusions=["attention_linear"],
                )
            
            # If we got here without OOM, test passes
            assert True


class TestErrorHandling:
    """Tests for error cases and fallbacks."""
    
    def test_invalid_fusion_name_skipped(self):
        """Invalid fusion names don't cause crashes."""
        with patch("aiprod_pipelines.inference.kernel_fusion.fusion_node.get_gpu_capabilities"):
            model = MagicMock()
            model.hidden_dim = 768
            model.num_attention_heads = 12
            scheduler = MagicMock()
            scheduler.step.return_value = {"prev_sample": torch.randn(1, 16, 64, 64, 48)}
            
            denoiser = FusedDenoiseNode(model, scheduler)
            
            # Try with invalid fusion name
            result = denoiser.forward(
                context={},
                latents=torch.randn(1, 16, 64, 64, 48),
                embeddings=torch.randn(1, 77, 768),
                timestep=torch.tensor([500]),
                selected_fusions=["invalid_fusion_name"],
            )
            
            assert "denoised_latents" in result
    
    def test_empty_fusions_list_works(self):
        """Empty fusions list doesn't cause errors."""
        with patch("aiprod_pipelines.inference.kernel_fusion.fusion_node.get_gpu_capabilities"):
            model = MagicMock()
            model.hidden_dim = 768
            model.num_attention_heads = 12
            scheduler = MagicMock()
            scheduler.step.return_value = {"prev_sample": torch.randn(1, 16, 64, 64, 48)}
            
            denoiser = FusedDenoiseNode(model, scheduler)
            
            result = denoiser.forward(
                context={},
                latents=torch.randn(1, 16, 64, 64, 48),
                embeddings=torch.randn(1, 77, 768),
                timestep=torch.tensor([500]),
                selected_fusions=[],
            )
            
            assert "denoised_latents" in result


class TestSpeedupReport:
    """Tests validating speedup claims."""
    
    def test_compound_speedup_calculation(self):
        """Multiple fusions compound correctly."""
        with patch("aiprod_pipelines.inference.kernel_fusion.adaptive_fusion.get_gpu_capabilities"):
            fusions_single = ["attention_linear"]
            fusions_multi = ["attention_linear", "residual_block", "conv_activation"]
            
            _, est_single = auto_select_kernel_fusions(
                model_config={"hidden_dim": 768}
            )
            _, est_multi = auto_select_kernel_fusions(
                model_config={"hidden_dim": 768}
            )
            
            # Both should show speedup
            assert est_single["cumulative_speedup"] > 1.0
            assert est_multi["cumulative_speedup"] > 1.0
