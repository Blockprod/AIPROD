"""
Shared test fixtures and utilities for streaming tests.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create temporary directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create test latents and conditions
        latents_dir = tmp_path / "latents"
        conditions_dir = tmp_path / "conditions"
        latents_dir.mkdir()
        conditions_dir.mkdir()
        
        # Generate 100 test samples
        for i in range(100):
            # Save latent
            latent = {
                "latents": torch.randn(128, 5, 32, 32),
                "num_frames": 5,
                "height": 32,
                "width": 32,
                "fps": 24,
            }
            torch.save(latent, latents_dir / f"latent_{i:04d}.pt")
            
            # Save condition
            condition = {
                "prompt_embeds": torch.randn(256, 4096),
                "prompt_attention_mask": torch.ones(256, dtype=torch.bool),
            }
            torch.save(condition, conditions_dir / f"condition_{i:04d}.pt")
        
        yield tmp_path


@pytest.fixture
def sample_tensor_dict() -> dict:
    """Create sample dict with tensors for cache testing."""
    return {
        "latents": torch.randn(128, 5, 32, 32),
        "prompts": torch.randn(256, 4096),
        "mask": torch.ones(256, dtype=torch.bool),
    }


@pytest.fixture
def async_event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def mock_fetch_fn():
    """Mock async fetch function that returns test data."""
    counter = {"value": 0}
    
    async def fetch(key: str) -> dict:
        counter["value"] += 1
        await asyncio.sleep(0.01)  # Simulate I/O
        return {
            "data": torch.randn(128, 5, 32, 32),
            "key": key,
            "fetch_count": counter["value"],
        }
    
    return fetch
