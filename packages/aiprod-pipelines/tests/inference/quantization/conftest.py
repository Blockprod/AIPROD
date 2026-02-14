"""
Test fixtures for quantization module testing.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, List

from aiprod_pipelines.inference.quantization import (
    QuantizationConfig, QuantizationProfile, ModelQuantizer
)


class SimpleLinearModel(nn.Module):
    """Simple model for quantization testing."""
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 64):
        """Initialize simple linear model."""
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleConvModel(nn.Module):
    """Simple convolutional model for testing."""
    
    def __init__(self, in_channels: int = 4, out_channels: int = 4, hidden_dim: int = 32):
        """Initialize simple conv model."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class MockGraphContext:
    """Mock GraphContext for testing."""
    
    def __init__(self, **kwargs):
        """Initialize with arbitrary context data."""
        self.data = kwargs
    
    def __getitem__(self, key: str):
        """Get context value."""
        return self.data[key]
    
    def __setitem__(self, key: str, value):
        """Set context value."""
        self.data[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.data
    
    def get(self, key: str, default=None):
        """Get with default."""
        return self.data.get(key, default)


@pytest.fixture
def simple_linear_model() -> nn.Module:
    """Provide a simple linear model for testing."""
    model = SimpleLinearModel(input_dim=64, hidden_dim=128, output_dim=64)
    model.eval()
    return model


@pytest.fixture
def simple_conv_model() -> nn.Module:
    """Provide a simple convolutional model for testing."""
    model = SimpleConvModel(in_channels=4, out_channels=4, hidden_dim=32)
    model.eval()
    return model


@pytest.fixture
def sample_calibration_data() -> List[torch.Tensor]:
    """Provide sample calibration data."""
    data = [torch.randn(4, 64) for _ in range(8)]
    return data


@pytest.fixture
def sample_conv_calibration_data() -> List[torch.Tensor]:
    """Provide sample calibration data for conv models."""
    data = [torch.randn(2, 4, 32, 32) for _ in range(8)]
    return data


@pytest.fixture
def quantization_config_int8() -> QuantizationConfig:
    """Provide INT8 quantization config."""
    return QuantizationConfig(
        quantization_method="int8",
        calibration_method="histogram",
        per_channel=True,
        dynamic=False,
        calibration_samples=8
    )


@pytest.fixture
def quantization_config_bf16() -> QuantizationConfig:
    """Provide BF16 quantization config."""
    return QuantizationConfig(
        quantization_method="bf16",
        per_channel=False,
        dynamic=False
    )


@pytest.fixture
def quantization_profile_int8() -> QuantizationProfile:
    """Provide INT8 quantization profile."""
    return QuantizationProfile(
        enable_quantization=True,
        quantization_method="int8",
        calibration_method="histogram",
        per_channel=True,
        quality_target_percent=95.0
    )


@pytest.fixture
def quantization_profile_bf16() -> QuantizationProfile:
    """Provide BF16 quantization profile."""
    return QuantizationProfile(
        enable_quantization=True,
        quantization_method="bf16",
        quality_target_percent=98.0
    )


@pytest.fixture
def model_dict(simple_linear_model, simple_conv_model) -> Dict[str, nn.Module]:
    """Provide dictionary of models for testing."""
    return {
        "encoder": simple_linear_model,
        "denoiser": simple_conv_model,
        "decoder": simple_linear_model
    }


@pytest.fixture
def graph_context_with_models(model_dict, sample_calibration_data) -> Dict:
    """Provide graph context with models."""
    return {
        "models": model_dict,
        "calibration_data": sample_calibration_data
    }


@pytest.fixture
def graph_context_with_latents() -> Dict:
    """Provide graph context with latent tensors."""
    return {
        "latents": torch.randn(2, 4, 16, 16),
        "embeddings": torch.randn(2, 768),
        "timestep": 500,
        "quantized_models": {
            "denoiser": SimpleConvModel()
        }
    }


@pytest.fixture
def device() -> str:
    """Provide compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"
