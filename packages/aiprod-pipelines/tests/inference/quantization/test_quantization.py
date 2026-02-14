"""
Unit tests for quantization.py module.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

from aiprod_pipelines.inference.quantization import (
    QuantizationConfig, QuantizationScheme, QuantizedTensor,
    QuantizationMetrics, ModelQuantizer, CalibrationDataCollector,
    QuantizationBenchmark
)


class TestQuantizationConfig:
    """Tests for QuantizationConfig dataclass."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = QuantizationConfig()
        assert config.quantization_method == "int8"
        assert config.calibration_method == "histogram"
        assert config.per_channel is True
        assert config.dynamic is False
        assert config.calibration_samples == 32
        assert config.calibration_percentile == 99.9
    
    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = QuantizationConfig(
            quantization_method="bf16",
            calibration_method="entropy",
            per_channel=False,
            dynamic=True,
            calibration_samples=64
        )
        assert config.quantization_method == "bf16"
        assert config.calibration_method == "entropy"
        assert config.per_channel is False
        assert config.dynamic is True
        assert config.calibration_samples == 64
    
    def test_config_validation_invalid_method(self):
        """Test validation of invalid quantization method."""
        with pytest.raises(AssertionError):
            QuantizationConfig(quantization_method="invalid")
    
    def test_config_validation_invalid_calibration(self):
        """Test validation of invalid calibration method."""
        with pytest.raises(AssertionError):
            QuantizationConfig(calibration_method="invalid")
    
    def test_config_validation_percentile_range(self):
        """Test validation of calibration percentile range."""
        with pytest.raises(AssertionError):
            QuantizationConfig(calibration_percentile=0)
        with pytest.raises(AssertionError):
            QuantizationConfig(calibration_percentile=101)


class TestQuantizationScheme:
    """Tests for QuantizationScheme dataclass."""
    
    def test_scheme_creation(self):
        """Test scheme creation with valid parameters."""
        scale = torch.tensor(0.01)
        zero_point = torch.tensor(0.0)
        scheme = QuantizationScheme(
            scale=scale,
            zero_point=zero_point,
            min_val=-1.0,
            max_val=1.0,
            num_bits=8
        )
        assert scheme.scale.item() == pytest.approx(0.01)
        assert scheme.zero_point.item() == pytest.approx(0.0)
        assert scheme.min_val == -1.0
        assert scheme.max_val == 1.0
        assert scheme.num_bits == 8
    
    def test_scheme_quantization_range(self):
        """Test quantization range calculation."""
        scheme = QuantizationScheme(
            scale=torch.tensor(0.01),
            zero_point=torch.tensor(0.0),
            min_val=-1.0,
            max_val=1.0
        )
        assert scheme.quantization_range == pytest.approx(2.0)
    
    def test_scheme_step_size(self):
        """Test step size calculation."""
        scheme = QuantizationScheme(
            scale=torch.tensor(0.01),
            zero_point=torch.tensor(0.0),
            min_val=-1.0,
            max_val=1.0,
            num_bits=8
        )
        # Step size = range / (2^bits - 1) = 2.0 / 255 ≈ 0.0078
        expected = 2.0 / 255.0
        assert scheme.step_size == pytest.approx(expected, rel=0.01)


class TestQuantizedTensor:
    """Tests for QuantizedTensor wrapper."""
    
    def test_quantized_tensor_creation(self):
        """Test quantized tensor creation."""
        data = torch.randint(0, 256, (2, 8))
        scale = torch.tensor(0.1)
        zero_point = torch.tensor(0.0)
        
        q_tensor = QuantizedTensor(
            data=data,
            scale=scale,
            zero_point=zero_point,
            original_dtype=torch.float32,
            quantization_method="int8",
            shape=(2, 8)
        )
        
        assert q_tensor.data.shape == (2, 8)
        assert q_tensor.scale.item() == pytest.approx(0.1)
        assert q_tensor.quantization_method == "int8"
    
    def test_dequantize_int8(self):
        """Test dequantization from INT8."""
        data = torch.ones(4, 4).long() * 100
        scale = torch.tensor(0.01)
        zero_point = torch.tensor(0.0)
        
        q_tensor = QuantizedTensor(
            data=data,
            scale=scale,
            zero_point=zero_point,
            original_dtype=torch.float32,
            quantization_method="int8",
            shape=(4, 4)
        )
        
        dequantized = q_tensor.dequantize()
        expected = torch.ones(4, 4) * 1.0  # 100 * 0.01
        assert torch.allclose(dequantized, expected, atol=1e-3)
    
    def test_dequantize_bf16(self):
        """Test dequantization from BF16."""
        data = torch.randn(4, 4)
        
        q_tensor = QuantizedTensor(
            data=data,
            scale=torch.tensor(1.0),
            zero_point=torch.tensor(0.0),
            original_dtype=torch.float32,
            quantization_method="bf16",
            shape=(4, 4)
        )
        
        dequantized = q_tensor.dequantize()
        assert dequantized.shape == (4, 4)


class TestQuantizationMetrics:
    """Tests for QuantizationMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test metrics creation."""
        metrics = QuantizationMetrics(
            original_model_size_mb=100.0,
            quantized_model_size_mb=25.0,
            compression_ratio=4.0,
            quality_retention_percent=98.5
        )
        assert metrics.original_model_size_mb == 100.0
        assert metrics.quantized_model_size_mb == 25.0
        assert metrics.compression_ratio == 4.0
        assert metrics.memory_saved_mb == pytest.approx(75.0)
    
    def test_metrics_speedup_calculation(self):
        """Test inference speedup calculation."""
        metrics = QuantizationMetrics(
            original_model_size_mb=100.0,
            quantized_model_size_mb=25.0,
            compression_ratio=4.0,
            inference_time_ms_before=100.0,
            inference_time_ms_after=40.0,
            quality_retention_percent=98.0
        )
        assert metrics.inference_speedup_factor == pytest.approx(2.5)
    
    def test_metrics_value_ranges(self):
        """Test metrics value ranges."""
        metrics = QuantizationMetrics(
            original_model_size_mb=100.0,
            quantized_model_size_mb=25.0,
            compression_ratio=4.0,
            quality_retention_percent=95.0
        )
        assert metrics.compression_ratio >= 1.0
        assert 0 <= metrics.quality_retention_percent <= 100
        assert metrics.memory_saved_mb >= 0


class TestCalibrationDataCollector:
    """Tests for CalibrationDataCollector."""
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        config = QuantizationConfig()
        collector = CalibrationDataCollector(config)
        assert len(collector.activations) == 0
    
    def test_collector_hook_registration(self):
        """Test hook registration on modules."""
        config = QuantizationConfig()
        collector = CalibrationDataCollector(config)
        
        module = nn.Linear(10, 20)
        collector.register_hook(module, "test_linear")
        
        # Run forward pass
        x = torch.randn(2, 10)
        _ = module(x)
        
        assert "test_linear" in collector.activations
        assert len(collector.activations["test_linear"]) == 1
    
    def test_collector_activation_tracking(self):
        """Test activation tracking across batches."""
        config = QuantizationConfig()
        collector = CalibrationDataCollector(config)
        
        module = nn.Linear(10, 20)
        collector.register_hook(module, "linear")
        
        # Multiple forward passes
        for _ in range(3):
            x = torch.randn(2, 10)
            _ = module(x)
        
        assert len(collector.activations["linear"]) == 3


class TestModelQuantizer:
    """Tests for ModelQuantizer engine."""
    
    def test_quantizer_initialization(self):
        """Test quantizer initialization."""
        config = QuantizationConfig(quantization_method="int8")
        quantizer = ModelQuantizer(config, device="cpu")
        assert quantizer.config.quantization_method == "int8"
    
    def test_model_size_computation(self):
        """Test model size computation."""
        config = QuantizationConfig()
        quantizer = ModelQuantizer(config, device="cpu")
        
        model = nn.Linear(100, 100)
        size_mb = quantizer._compute_model_size_mb(model)
        
        # Linear(100, 100): 100*100 weights + 100 bias = 10100 params
        # 10100 * 4 bytes = 40400 bytes ≈ 0.0385 MB
        assert size_mb > 0
    
    def test_quantize_linear_model(self):
        """Test quantization of linear model."""
        config = QuantizationConfig(quantization_method="int8")
        quantizer = ModelQuantizer(config, device="cpu")
        
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        quantized_model, metrics = quantizer.quantize_model(model)
        
        assert isinstance(quantized_model, nn.Module)
        assert isinstance(metrics, QuantizationMetrics)
        assert metrics.quantization_method == "int8"
    
    def test_quantize_bf16_model(self):
        """Test BF16 quantization."""
        config = QuantizationConfig(quantization_method="bf16")
        quantizer = ModelQuantizer(config, device="cpu")
        
        model = nn.Linear(32, 32)
        quantized_model, metrics = quantizer.quantize_model(model)
        
        assert isinstance(metrics, QuantizationMetrics)
        assert metrics.quantization_method == "bf16"
    
    def test_quantize_with_calibration(self):
        """Test quantization with calibration data."""
        config = QuantizationConfig(
            quantization_method="int8",
            dynamic=False,
            calibration_samples=4
        )
        quantizer = ModelQuantizer(config, device="cpu")
        
        model = nn.Linear(32, 32)
        calibration_data = [torch.randn(4, 32) for _ in range(4)]
        
        quantized_model, metrics = quantizer.quantize_model(
            model,
            calibration_data=calibration_data
        )
        
        assert metrics.calibration_samples_used == 4
    
    def test_quantize_metrics_quality(self):
        """Test quantization metrics quality retention."""
        config = QuantizationConfig(quantization_method="int8")
        quantizer = ModelQuantizer(config, device="cpu")
        
        model = nn.Linear(16, 16)
        _, metrics = quantizer.quantize_model(model)
        
        assert metrics.quality_retention_percent >= 95.0
        assert metrics.compression_ratio >= 1.0


class TestQuantizationBenchmark:
    """Tests for QuantizationBenchmark utilities."""
    
    def test_quality_metrics_computation(self):
        """Test quality metric computation."""
        original = torch.randn(8, 16)
        quantized = original + 0.01 * torch.randn(8, 16)
        
        metrics = QuantizationBenchmark.measure_quality(original, quantized)
        
        assert "mse" in metrics
        assert "mae" in metrics
        assert "psnr" in metrics
        assert "quality_retention_percent" in metrics
        
        assert metrics["mse"] >= 0
        assert metrics["mae"] >= 0
        assert 0 <= metrics["quality_retention_percent"] <= 100
    
    def test_quality_perfect_match(self):
        """Test quality metrics for perfect match."""
        original = torch.randn(8, 16)
        quantized = original.clone()
        
        metrics = QuantizationBenchmark.measure_quality(original, quantized)
        
        # MSE should be ~0, quality retention should be 100
        assert metrics["mse"] == pytest.approx(0, abs=1e-6)
        assert metrics["quality_retention_percent"] == pytest.approx(100, rel=1e-2)


class TestQuantizationIntegration:
    """Integration tests for quantization workflow."""
    
    def test_full_quantization_pipeline(self):
        """Test complete quantization pipeline."""
        # Create model
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Create quantizer
        config = QuantizationConfig(
            quantization_method="int8",
            per_channel=True
        )
        quantizer = ModelQuantizer(config, device="cpu")
        
        # Quantize
        quantized_model, metrics = quantizer.quantize_model(model)
        
        # Verify
        assert quantized_model is not None
        assert metrics.compression_ratio >= 2.0  # Target 2x for INT8
        assert metrics.quality_retention_percent >= 90.0
    
    def test_quantization_preserves_output_shape(self):
        """Test that quantization preserves output shapes."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        
        config = QuantizationConfig(quantization_method="int8")
        quantizer = ModelQuantizer(config, device="cpu")
        
        quantized_model, _ = quantizer.quantize_model(model)
        
        # Test forward pass
        input_x = torch.randn(4, 32)
        output = quantized_model(input_x)
        
        assert output.shape == (4, 16)
    
    def test_batch_quantization(self):
        """Test quantization of multiple models."""
        models = {
            "encoder": nn.Linear(32, 64),
            "decoder": nn.Linear(64, 32),
            "classifier": nn.Linear(64, 10)
        }
        
        config = QuantizationConfig(quantization_method="int8")
        quantizer = ModelQuantizer(config, device="cpu")
        
        quantized_models = {}
        for name, model in models.items():
            q_model, _ = quantizer.quantize_model(model)
            quantized_models[name] = q_model
        
        assert len(quantized_models) == 3
        assert all(isinstance(m, nn.Module) for m in quantized_models.values())
