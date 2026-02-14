"""Pytest configuration and shared fixtures for AIPROD testing."""

import os
import pytest
import sys
import torch
import numpy as np
from pathlib import Path

# Add package sources to Python path for testing
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root / "packages" / "aiprod-core" / "src"))
sys.path.insert(0, str(workspace_root / "packages" / "aiprod-trainer" / "src"))
sys.path.insert(0, str(workspace_root / "packages" / "aiprod-pipelines" / "src"))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# Set device for tests
@pytest.fixture(scope="session")
def torch_device():
    """Get the device for testing (CPU or GPU if available)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def dummy_video_tensor():
    """Return a dummy video tensor for testing."""
    # [B, C, T, H, W]
    return torch.randn(2, 3, 8, 64, 64)


@pytest.fixture
def dummy_audio_tensor():
    """Return a dummy audio tensor for testing."""
    # [B, T] - audio waveform
    return torch.randn(2, 16000)


@pytest.fixture
def dummy_latent_video():
    """Return a dummy video latent tensor."""
    # [B, C, T, H, W] - latent space
    return torch.randn(2, 4, 2, 8, 8)


@pytest.fixture
def dummy_latent_audio():
    """Return a dummy audio latent tensor."""
    # [B, C, T] - audio latent
    return torch.randn(2, 8, 40)


@pytest.fixture
def dummy_text_embeddings():
    """Return dummy text embeddings."""
    # [B, seq_len, hidden_dim]
    return torch.randn(2, 77, 768)


@pytest.fixture
def dummy_prompt():
    """Return a simple test prompt."""
    return "A person speaking in a professional setting"


@pytest.fixture
def dummy_negative_prompt():
    """Return a simple negative prompt."""
    return "blurry, low quality"


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return seed


@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Clean up CUDA memory after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
