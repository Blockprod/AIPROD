"""
Unit tests for quantization_node.py module.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from aiprod_pipelines.inference.quantization import QuantizationConfig
from aiprod_pipelines.inference.quantization_node import (
    QuantizationProfile, ModelQuantizationNode, QuantizedInferenceNode,
    QuantizationAdaptiveNode
)


class TestQuantizationProfile:
    """Tests for QuantizationProfile dataclass."""
    
    def test_profile_defaults(self):
        """Test default profile values."""
        profile = QuantizationProfile()
        assert profile.enable_quantization is True
        assert profile.quantization_method == "int8"
        assert profile.calibration_method == "histogram"
        assert profile.per_channel is True
        assert profile.dynamic is False
        assert profile.quality_target_percent == 95.0
    
    def test_profile_custom_values(self):
        """Test custom profile values."""
        profile = QuantizationProfile(
            enable_quantization=False,
            quantization_method="bf16",
            quality_target_percent=98.5,
            enable_dynamic_selection=True
        )
        assert profile.enable_quantization is False
        assert profile.quantization_method == "bf16"
        assert profile.quality_target_percent == 98.5
        assert profile.enable_dynamic_selection is True
    
    def test_profile_validation_method(self):
        """Test profile validation of quantization method."""
        with pytest.raises(AssertionError):
            QuantizationProfile(quantization_method="invalid")
    
    def test_profile_validation_quality_target(self):
        """Test profile validation of quality target."""
        with pytest.raises(AssertionError):
            QuantizationProfile(quality_target_percent=0)
        with pytest.raises(AssertionError):
            QuantizationProfile(quality_target_percent=101)


class TestModelQuantizationNode:
    """Tests for ModelQuantizationNode."""
    
    def test_node_initialization(self):
        """Test node initialization."""
        profile = QuantizationProfile()
        node = ModelQuantizationNode(profile, device="cpu")
        assert node.profile.quantization_method == "int8"
        assert len(node.metrics_history) == 0
    
    def test_node_input_keys(self):
        """Test node input_keys property."""
        profile = QuantizationProfile()
        node = ModelQuantizationNode(profile, device="cpu")
        assert "models" in node.input_keys
    
    def test_node_output_keys(self):
        """Test node output_keys property."""
        profile = QuantizationProfile()
        node = ModelQuantizationNode(profile, device="cpu")
        output_keys = node.output_keys
        assert "quantized_models" in output_keys
        assert "quantization_metrics" in output_keys
        assert "speedup_summary" in output_keys
    
    def test_quantization_disabled(self):
        """Test pass-through when quantization disabled."""
        profile = QuantizationProfile(enable_quantization=False)
        node = ModelQuantizationNode(profile, device="cpu")
        
        models = {
            "encoder": nn.Linear(32, 64),
            "decoder": nn.Linear(64, 32)
        }
        context = {"models": models}
        
        result = node.execute(context)
        
        assert result["quantized_models"] is models
        assert result["speedup_summary"]["speedup_factor"] == 1.0
    
    def test_execute_missing_models(self):
        """Test execution fails without models in context."""
        profile = QuantizationProfile()
        node = ModelQuantizationNode(profile, device="cpu")
        
        context = {}
        
        with pytest.raises(KeyError):
            node.execute(context)
    
    def test_execute_with_models(self):
        """Test execution with models in context."""
        profile = QuantizationProfile()
        node = ModelQuantizationNode(profile, device="cpu")
        
        models = {
            "encoder": nn.Linear(32, 64),
            "denoiser": nn.Linear(64, 64),
            "decoder": nn.Linear(64, 32)
        }
        context = {"models": models}
        
        result = node.execute(context)
        
        assert "quantized_models" in result
        assert "quantization_metrics" in result
        assert "speedup_summary" in result
        assert len(result["quantized_models"]) > 0
    
    def test_speedup_summary_computation(self):
        """Test speedup summary computation."""
        from aiprod_pipelines.inference.quantization import QuantizationMetrics
        
        profile = QuantizationProfile()
        node = ModelQuantizationNode(profile, device="cpu")
        
        metrics = {
            "encoder": QuantizationMetrics(
                original_model_size_mb=50.0,
                quantized_model_size_mb=12.5,
                compression_ratio=4.0,
                quality_retention_percent=98.0
            ),
            "decoder": QuantizationMetrics(
                original_model_size_mb=50.0,
                quantized_model_size_mb=12.5,
                compression_ratio=4.0,
                quality_retention_percent=97.5
            )
        }
        
        summary = node._compute_speedup_summary(metrics)
        
        assert "speedup_factor" in summary
        assert "average_quality_retention" in summary
        assert "models_quantized" in summary
        assert summary["models_quantized"] == 2
    
    def test_quantization_summary(self):
        """Test quantization summary generation."""
        profile = QuantizationProfile()
        node = ModelQuantizationNode(profile, device="cpu")
        
        # Execute first
        models = {"encoder": nn.Linear(32, 64)}
        context = {"models": models}
        _ = node.execute(context)
        
        summary = node.get_quantization_summary()
        
        assert summary["history_length"] == 1
        assert summary["models_quantized"] > 0
    
    def test_metrics_history_tracking(self):
        """Test metrics history tracking."""
        profile = QuantizationProfile()
        node = ModelQuantizationNode(profile, device="cpu")
        
        models = {"encoder": nn.Linear(32, 64)}
        context = {"models": models}
        
        # Execute multiple times
        for _ in range(3):
            _ = node.execute(context)
        
        assert len(node.metrics_history) == 3
    
    def test_reset_history(self):
        """Test history reset."""
        profile = QuantizationProfile()
        node = ModelQuantizationNode(profile, device="cpu")
        
        models = {"encoder": nn.Linear(32, 64)}
        context = {"models": models}
        
        _ = node.execute(context)
        assert len(node.metrics_history) == 1
        
        node.reset_history()
        assert len(node.metrics_history) == 0


class TestQuantizedInferenceNode:
    """Tests for QuantizedInferenceNode."""
    
    def test_inference_node_initialization(self):
        """Test inference node initialization."""
        profile = QuantizationProfile()
        node = QuantizedInferenceNode(profile, device="cpu")
        assert node.profile.enable_quantization is True
    
    def test_inference_node_input_keys(self):
        """Test inference node input keys."""
        profile = QuantizationProfile()
        node = QuantizedInferenceNode(profile, device="cpu")
        
        expected_keys = ["quantized_models", "latents", "embeddings", "timestep"]
        assert all(key in node.input_keys for key in expected_keys)
    
    def test_inference_node_output_keys(self):
        """Test inference node output keys."""
        profile = QuantizationProfile()
        node = QuantizedInferenceNode(profile, device="cpu")
        
        output_keys = node.output_keys
        assert "latents_denoised" in output_keys
        assert "inference_time_ms" in output_keys
        assert "quantization_active" in output_keys
    
    def test_inference_disabled(self):
        """Test inference with quantization disabled."""
        profile = QuantizationProfile(enable_quantization=False)
        node = QuantizedInferenceNode(profile, device="cpu")
        
        context = {
            "quantized_models": {"denoiser": nn.Linear(64, 64)},
            "latents": torch.randn(2, 64),
            "embeddings": torch.randn(2, 768),
            "timestep": 500
        }
        
        result = node.execute(context)
        
        assert torch.allclose(result["latents_denoised"], context["latents"])
        assert result["inference_time_ms"] == 0.0
        assert result["quantization_active"] is False
    
    def test_inference_missing_keys(self):
        """Test inference fails with missing required keys."""
        profile = QuantizationProfile()
        node = QuantizedInferenceNode(profile, device="cpu")
        
        context = {
            "latents": torch.randn(2, 64)
            # Missing other required keys
        }
        
        with pytest.raises(KeyError):
            node.execute(context)
    
    def test_inference_missing_denoiser(self):
        """Test inference fails without denoiser model."""
        profile = QuantizationProfile()
        node = QuantizedInferenceNode(profile, device="cpu")
        
        context = {
            "quantized_models": {"encoder": nn.Linear(32, 64)},  # No denoiser
            "latents": torch.randn(2, 64),
            "embeddings": torch.randn(2, 768),
            "timestep": 500
        }
        
        with pytest.raises(KeyError):
            node.execute(context)
    
    def test_inference_timing(self):
        """Test inference timing measurement."""
        profile = QuantizationProfile()
        node = QuantizedInferenceNode(profile, device="cpu")
        
        context = {
            "quantized_models": {"denoiser": nn.Linear(64, 64)},
            "latents": torch.randn(2, 64),
            "embeddings": torch.randn(2, 768),
            "timestep": 500
        }
        
        result = node.execute(context)
        
        assert result["inference_time_ms"] >= 0
        assert result["quantization_active"] is True


class TestQuantizationAdaptiveNode:
    """Tests for QuantizationAdaptiveNode."""
    
    def test_adaptive_node_initialization(self):
        """Test adaptive node initialization."""
        node = QuantizationAdaptiveNode(device="cpu")
        assert node.device == "cpu"
    
    def test_adaptive_node_input_keys(self):
        """Test adaptive node input keys."""
        node = QuantizationAdaptiveNode(device="cpu")
        assert "models" in node.input_keys
    
    def test_adaptive_node_output_keys(self):
        """Test adaptive node output keys."""
        node = QuantizationAdaptiveNode(device="cpu")
        
        output_keys = node.output_keys
        assert "recommended_method" in output_keys
        assert "quantized_models" in output_keys
        assert "selection_reasoning" in output_keys
    
    def test_adaptive_model_selection(self):
        """Test automatic model selection."""
        node = QuantizationAdaptiveNode(device="cpu")
        
        models = {
            "encoder": nn.Linear(32, 64),
            "decoder": nn.Linear(64, 32)
        }
        context = {"models": models}
        
        result = node.execute(context)
        
        assert "recommended_method" in result
        assert result["recommended_method"] in ["int8", "bf16", "fp8"]
    
    def test_adaptive_with_quality_target(self):
        """Test adaptive selection with quality target."""
        node = QuantizationAdaptiveNode(device="cpu")
        
        models = {
            "encoder": nn.Linear(32, 64),
        }
        context = {
            "models": models,
            "target_quality_percent": 99.0
        }
        
        result = node.execute(context)
        
        # High quality target might prefer BF16 or FP8
        assert result["recommended_method"] is not None
    
    def test_adaptive_missing_models(self):
        """Test adaptive fails without models."""
        node = QuantizationAdaptiveNode(device="cpu")
        
        context = {}
        
        with pytest.raises(KeyError):
            node.execute(context)
    
    def test_adaptive_selection_reasoning(self):
        """Test selection reasoning generation."""
        node = QuantizationAdaptiveNode(device="cpu")
        
        models = {"encoder": nn.Linear(32, 64)}
        context = {"models": models}
        
        result = node.execute(context)
        
        assert "selection_reasoning" in result
        assert len(result["selection_reasoning"]) > 0


class TestQuantizationNodeIntegration:
    """Integration tests for quantization nodes."""
    
    def test_quantization_inference_pipeline(self):
        """Test combined quantization and inference pipeline."""
        # Quantization
        quant_profile = QuantizationProfile()
        quant_node = ModelQuantizationNode(quant_profile, device="cpu")
        
        models = {
            "encoder": nn.Linear(32, 64),
            "denoiser": nn.Linear(64, 64),
            "decoder": nn.Linear(64, 32)
        }
        quant_context = {"models": models}
        quant_result = quant_node.execute(quant_context)
        
        # Inference
        inf_profile = QuantizationProfile()
        inf_node = QuantizedInferenceNode(inf_profile, device="cpu")
        
        inf_context = {
            "quantized_models": quant_result["quantized_models"],
            "latents": torch.randn(2, 64),
            "embeddings": torch.randn(2, 768),
            "timestep": 500
        }
        inf_result = inf_node.execute(inf_context)
        
        assert inf_result["latents_denoised"].shape == inf_context["latents"].shape
    
    def test_adaptive_fallback(self):
        """Test adaptive fallback mechanism."""
        profile = QuantizationProfile(
            quantization_method="int8",
            fallback_method="bf16",
            quality_target_percent=98.0
        )
        node = ModelQuantizationNode(profile, device="cpu")
        
        models = {"encoder": nn.Linear(32, 64)}
        context = {"models": models}
        
        result = node.execute(context)
        
        # Should have either original or fallback quantized model
        assert "quantized_models" in result
        assert "encoder" in result["quantized_models"]
    
    def test_multiple_model_quantization(self):
        """Test quantization of multiple different model sizes."""
        profile = QuantizationProfile()
        node = ModelQuantizationNode(profile, device="cpu")
        
        models = {
            "small_encoder": nn.Linear(16, 32),
            "medium_encoder": nn.Linear(64, 128),
            "large_encoder": nn.Linear(256, 512),
            "denoiser": nn.Conv2d(4, 4, 3, padding=1)
        }
        context = {"models": models}
        
        result = node.execute(context)
        
        assert len(result["quantized_models"]) >= 3
        metrics = result["quantization_metrics"]
        assert all(m.compression_ratio > 1.0 for m in metrics.values())
