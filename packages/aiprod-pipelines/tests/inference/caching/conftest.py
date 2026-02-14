"""
Test fixtures for caching module testing.
"""

import pytest
import torch
import torch.nn as nn
from typing import List, Dict

from aiprod_pipelines.inference.caching import (
    CacheConfig, InferenceCache, EmbeddingCache, FeatureCache, VAECache
)
from aiprod_pipelines.inference.caching_node import CachingProfile


class MockTextEncoder(nn.Module):
    """Mock text encoder for testing."""
    
    def __init__(self, embed_dim: int = 768):
        """Initialize mock text encoder."""
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(1000, embed_dim)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Mock encoding - just hash and embed."""
        batch_size = len(texts)
        # Simulate embeddings based on text hash
        indices = torch.tensor([hash(t) % 1000 for t in texts])
        return self.embedding(indices)


class MockVAEDecoder(nn.Module):
    """Mock VAE decoder for testing."""
    
    def __init__(self, latent_dim: int = 4, output_channels: int = 3):
        """Initialize mock VAE decoder."""
        super().__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.fc = nn.Linear(latent_dim * 16 * 16, 3 * 64 * 64)
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Mock decoding."""
        batch_size = latents.shape[0]
        flat = latents.reshape(batch_size, -1)
        output = self.fc(flat)
        return output.reshape(batch_size, self.output_channels, 64, 64)


@pytest.fixture
def cache_config() -> CacheConfig:
    """Provide cache configuration."""
    return CacheConfig(
        embed_cache_size_mb=64,
        feature_cache_size_mb=128,
        vae_cache_size_mb=32,
        enable_embedding_cache=True,
        enable_feature_cache=True,
        enable_vae_cache=True
    )


@pytest.fixture
def caching_profile() -> CachingProfile:
    """Provide caching profile."""
    return CachingProfile(
        enable_caching=True,
        embed_cache_size_mb=64,
        feature_cache_size_mb=128,
        vae_cache_size_mb=32,
        enable_embedding_cache=True,
        enable_feature_cache=True,
        enable_vae_cache=True
    )


@pytest.fixture
def sample_texts() -> List[str]:
    """Provide sample text prompts."""
    return [
        "A dog running in the park",
        "A cat sleeping on a couch",
        "A bird flying in the sky",
        "A fish swimming in water"
    ]


@pytest.fixture
def sample_duplicate_texts() -> List[List[str]]:
    """Provide prompts with duplicates to test cache hits."""
    return [
        ["A dog running in the park"],
        ["A cat sleeping on a couch"],
        ["A dog running in the park"],  # Duplicate - should hit cache
        ["A cat sleeping on a couch"],  # Duplicate - should hit cache
    ]


@pytest.fixture
def sample_latents() -> torch.Tensor:
    """Provide sample latents."""
    return torch.randn(2, 4, 16, 16)


@pytest.fixture
def text_encoder() -> nn.Module:
    """Provide mock text encoder."""
    encoder = MockTextEncoder(embed_dim=768)
    encoder.eval()
    return encoder


@pytest.fixture
def vae_decoder() -> nn.Module:
    """Provide mock VAE decoder."""
    decoder = MockVAEDecoder(latent_dim=4, output_channels=3)
    decoder.eval()
    return decoder


@pytest.fixture
def inference_cache(cache_config) -> InferenceCache:
    """Provide configured inference cache."""
    return InferenceCache(cache_config, device="cpu")


@pytest.fixture
def embedding_cache() -> EmbeddingCache:
    """Provide embedding cache."""
    return EmbeddingCache(max_size_mb=64)


@pytest.fixture
def feature_cache() -> FeatureCache:
    """Provide feature cache."""
    return FeatureCache(max_size_mb=128)


@pytest.fixture
def vae_cache() -> VAECache:
    """Provide VAE cache."""
    return VAECache(max_size_mb=32)


@pytest.fixture
def device() -> str:
    """Provide compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


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
