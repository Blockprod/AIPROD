"""
Pytest configuration and fixtures for aiprod-core tests.
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Create a temporary directory for test models."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


@pytest.fixture
def sample_config():
    """Provide a sample configuration for tests."""
    return {
        "model_variant": "distilled",
        "precision": "fp16",
        "device": "cpu",
        "cache_enabled": True,
        "max_batch_size": 1,
    }


@pytest.fixture
def mock_model_path(tmp_models_dir):
    """Create a mock model file path."""
    model_file = tmp_models_dir / "mock_model.pt"
    model_file.touch()
    return str(model_file)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Add markers for slow tests
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add markers for integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
