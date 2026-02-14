"""
Test fixtures for latent distillation module.

Provides:
  - Sample latent tensors with various characteristics
  - Mock compression models
  - Degraded latent samples (for quality testing)
  - Metrics templates
"""

import pytest
import torch


@pytest.fixture
def sample_latents(batch_size: int = 2) -> torch.Tensor:
    """Fixture: Clean latent tensors [batch, 4, 32, 32]."""
    return torch.randn(batch_size, 4, 32, 32)


@pytest.fixture
def high_variance_latents(batch_size: int = 2) -> torch.Tensor:
    """Fixture: High-variance latents with sharp features."""
    return torch.randn(batch_size, 4, 32, 32) * 2.0


@pytest.fixture
def low_variance_latents(batch_size: int = 2) -> torch.Tensor:
    """Fixture: Low-variance smooth latents."""
    return torch.randn(batch_size, 4, 32, 32) * 0.2


@pytest.fixture
def structured_latents(batch_size: int = 2) -> torch.Tensor:
    """Fixture: Structured latents with few distinct values."""
    # Create blocky pattern (high compressibility)
    x = torch.zeros(batch_size, 4, 32, 32)
    for i in range(0, 32, 8):
        for j in range(0, 32, 8):
            x[:, :, i:i+8, j:j+8] = torch.randn(batch_size, 4, 1, 1)
    return x


@pytest.fixture
def random_noise_latents(batch_size: int = 2) -> torch.Tensor:
    """Fixture: Pure random noise (low compressibility)."""
    return torch.rand(batch_size, 4, 32, 32) * 2.0 - 1.0


@pytest.fixture
def large_batch_latents() -> torch.Tensor:
    """Fixture: Larger batch for real-world testing."""
    return torch.randn(16, 4, 32, 32)


@pytest.fixture
def compression_config():
    """Fixture: Compression configuration."""
    from aiprod_pipelines.inference.latent_distillation import LatentCompressionConfig
    
    return LatentCompressionConfig(
        codebook_size=256,
        embedding_dim=64,
        num_quantizers=4,
        use_exponential_moving_average=True,
        commitment_loss_weight=0.25,
    )


@pytest.fixture
def distillation_profile():
    """Fixture: Distillation profile."""
    from aiprod_pipelines.inference.latent_distillation_node import DistillationProfile
    
    return DistillationProfile(
        enable_compression=True,
        enable_reconstruction_loss=True,
        codebook_size=256,
        num_quantizers=4,
        quality_target_percent=95.0,
    )


@pytest.fixture
def graph_context_with_latents(sample_latents):
    """Fixture: GraphContext with latents."""
    class MockGraphContext(dict):
        pass
    
    context = MockGraphContext()
    context["latents"] = sample_latents
    context["embeddings"] = torch.randn(sample_latents.shape[0], 77, 768)
    context["prompt"] = "A test prompt"
    
    return context


@pytest.fixture
def graph_context_with_codes():
    """Fixture: GraphContext with compressed codes."""
    class MockGraphContext(dict):
        pass
    
    context = MockGraphContext()
    context["compressed_codes"] = torch.randint(0, 256, (2, 4, 32, 32))
    
    return context
