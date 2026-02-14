"""
Test fixtures for adaptive guidance module.

Provides:
  - Mock models (prompt analyzer, quality predictor)
  - Sample prompts and embeddings
  - Mock schedulers
  - Predefined test latents with various quality characteristics
"""

from typing import List, Tuple

import pytest
import torch
from torch import nn


class MockPromptAnalyzer(nn.Module):
    """Mock prompt analyzer for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 3)
    
    def forward(self, embeddings):
        """Analyze embeddings."""
        # Return (complexity, base_guidance, confidence)
        output = self.linear(embeddings.mean(dim=1))
        return output


class MockQualityPredictor(nn.Module):
    """Mock quality predictor for testing."""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)
    
    def forward(self, features):
        """Predict adjustment and confidence."""
        # Input: [variance, smoothness, alignment, artifacts]
        # Output: [adjustment, confidence, early_exit_prob]
        return self.linear(features)


class MockScheduler:
    """Mock noise scheduler for testing."""
    
    def __init__(self, num_steps: int = 30):
        self.num_steps = num_steps
        self.timesteps = list(range(999, -1, 1000 // num_steps))[:num_steps]
    
    def step(self, model_output, timestep, sample):
        """Mock scheduler step."""
        # Just add small noise for testing
        noise = torch.randn_like(sample) * 0.01
        return {"prev_sample": sample + noise}


@pytest.fixture
def mock_prompt_analyzer():
    """Fixture: Mock prompt analyzer."""
    return MockPromptAnalyzer()


@pytest.fixture
def mock_quality_predictor():
    """Fixture: Mock quality predictor."""
    return MockQualityPredictor()


@pytest.fixture
def mock_scheduler():
    """Fixture: Mock scheduler with 30 steps."""
    return MockScheduler(num_steps=30)


@pytest.fixture
def sample_prompts() -> List[str]:
    """Fixture: Sample test prompts."""
    return [
        "A cat sitting on a chair",
        "A dog running in a field",
        "A person dancing in the rain at night with detailed background",
        "Abstract geometric shapes",
        "A car driving down a road",
    ]


@pytest.fixture
def sample_embeddings(batch_size: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fixture: Sample text embeddings.
    
    Returns:
        (positive_embeddings, negative_embeddings)
    """
    # Positive embeddings
    positive = torch.randn(batch_size, 77, 768)
    
    # Negative embeddings (unconditional)
    negative = torch.randn(batch_size, 77, 768)
    
    return positive, negative


@pytest.fixture
def clean_latents(batch_size: int = 1) -> torch.Tensor:
    """Fixture: Clean latents (converged state - low variance, high smoothness)."""
    # Low variance, high temporal smoothness
    return torch.randn(batch_size, 4, 32, 32) * 0.1


@pytest.fixture
def noisy_latents(batch_size: int = 1) -> torch.Tensor:
    """Fixture: Noisy latents (unconverged state - high variance)."""
    # High variance
    return torch.randn(batch_size, 4, 32, 32) * 2.0


@pytest.fixture
def diverging_latents(batch_size: int = 1) -> torch.Tensor:
    """Fixture: Diverging latents (mode collapse - low variance, artifacts)."""
    # Repeated pattern (artifacts)
    return torch.ones(batch_size, 4, 32, 32) * 0.5


@pytest.fixture
def quality_trajectory_data():
    """Fixture: Mock quality trajectory data across 30 steps."""
    return {
        "timesteps": list(range(999, -1, -34))[:30],  # 30 timesteps
        "variances": [2.0 - (i * 0.05) for i in range(30)],  # Decreasing variance
        "smoothness": [0.2 + (i * 0.025) for i in range(30)],  # Increasing smoothness
        "alignments": [0.3 + (i * 0.02) for i in range(30)],  # Increasing alignment
        "artifacts": [0.6 - (i * 0.015) for i in range(30)],  # Decreasing artifacts
    }


@pytest.fixture
def mock_denoise_model():
    """Fixture: Simple mock denoising model."""
    
    class SimpleDenoise(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
        
        def forward(self, x, t, embeddings):
            """Mock forward pass."""
            # Ignore inputs, return simple noise
            return torch.randn_like(x)
    
    return SimpleDenoise()


@pytest.fixture
def guidance_profile_dict():
    """Fixture: Sample guidance profile configuration."""
    return {
        "enable_prompt_analysis": True,
        "enable_timestep_scaling": True,
        "enable_quality_adjustment": True,
        "enable_early_exit": True,
        "min_steps": 15,
    }


@pytest.fixture
def graph_context():
    """Fixture: Mock GraphContext for node execution."""
    
    class MockGraphContext(dict):
        """Simple dict-based context."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.node_outputs = {}
        
        def validate_keys(self, required_keys: List[str]) -> None:
            """Check required keys exist."""
            missing = [k for k in required_keys if k not in self]
            if missing:
                raise KeyError(f"Missing keys: {missing}")
    
    context = MockGraphContext()
    context["latents"] = torch.randn(1, 4, 32, 32)
    context["embeddings"] = torch.randn(1, 77, 768)
    context["prompt"] = "A test prompt"
    
    return context
