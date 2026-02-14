"""
Pytest configuration and fixtures for inference graph tests.
"""

import pytest
import torch
from typing import Dict, Any

from aiprod_pipelines.inference import (
    GraphNode,
    GraphContext,
    TextEncodeNode,
    DenoiseNode,
    DecodeVideoNode,
    CleanupNode,
)


@pytest.fixture
def mock_text_encoder():
    """Mock Gemma 3 text encoder."""
    class MockTextEncoder:
        def __init__(self):
            self.hidden_size = 4096
    
    return MockTextEncoder()


@pytest.fixture
def mock_denoising_model():
    """Mock denoising transformer model."""
    class MockModel:
        def __init__(self):
            self.hidden_size = 4096
    
    return MockModel()


@pytest.fixture
def mock_scheduler():
    """Mock noise scheduler."""
    class MockScheduler:
        def __init__(self):
            self.timesteps = torch.linspace(999, 0, 30, dtype=torch.long)
        
        def step(self, model_output, timestep, sample):
            # Simulate scheduler step
            return {"prev_sample": sample * 0.99}
    
    return MockScheduler()


@pytest.fixture
def mock_vae_decoder():
    """Mock VAE decoder model."""
    class MockVAEDecoder:
        pass
    
    return MockVAEDecoder()


@pytest.fixture
def mock_upsampler():
    """Mock upsampler model."""
    class MockUpsampler:
        pass
    
    return MockUpsampler()


@pytest.fixture
def sample_latents():
    """Sample latent tensor."""
    return torch.randn(1, 8, 16, 64, 64)  # [batch, channels, frames, height, width]


@pytest.fixture
def sample_embeddings():
    """Sample text embeddings."""
    return torch.randn(2, 77, 4096)  # [batch*2 (pos+neg), seq_len, hidden_dim]


@pytest.fixture
def sample_context(sample_latents, sample_embeddings):
    """Sample GraphContext with embeddings."""
    context = GraphContext()
    context.inputs = {
        "prompt": "A cat walking through a forest",
        "embeddings": sample_embeddings,
        "latents": sample_latents,
    }
    return context


@pytest.fixture
def device():
    """Get device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
