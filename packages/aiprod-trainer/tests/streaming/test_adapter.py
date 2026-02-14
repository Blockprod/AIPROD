"""
Integration tests for StreamingDatasetAdapter.

Tests:
- Dataset creation and initialization
- Multi-source support
- Batch prefetching
- DataLoader compatibility
- Performance characteristics
"""

import asyncio
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from aiprod_trainer.streaming import DataSourceConfig, StreamingDatasetAdapter


@pytest.mark.asyncio
async def test_adapter_basic_creation(temp_data_dir):
    """Test creating adapter with single local source."""
    config = DataSourceConfig(
        name="local",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    adapter = StreamingDatasetAdapter(
        sources=[config],
        data_mapping={"latents": "latent_conditions"},
    )
    
    assert len(adapter) == 100
    assert adapter._total_samples == 100


@pytest.mark.asyncio
async def test_adapter_getitem(temp_data_dir):
    """Test retrieving items from adapter."""
    config = DataSourceConfig(
        name="local",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    adapter = StreamingDatasetAdapter(
        sources=[config],
        data_mapping={"latents": "latent_conditions"},
    )
    
    # Get first item
    item = adapter[0]
    
    assert "latent_conditions" in item
    assert "idx" in item
    assert item["idx"] == 0
    assert isinstance(item["latent_conditions"], dict)


@pytest.mark.asyncio
async def test_adapter_multiple_sources(temp_data_dir):
    """Test adapter with multiple data sources."""
    latents_config = DataSourceConfig(
        name="latents",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    conditions_config = DataSourceConfig(
        name="conditions",
        source_type="local",
        path_or_uri=str(temp_data_dir / "conditions"),
    )
    
    adapter = StreamingDatasetAdapter(
        sources=[latents_config, conditions_config],
        data_mapping={
            "latents": "latent_conditions",
            "conditions": "text_conditions",
        },
    )
    
    item = adapter[0]
    
    assert "latent_conditions" in item
    assert "text_conditions" in item


@pytest.mark.asyncio
async def test_adapter_cache_stats(temp_data_dir):
    """Test cache statistics."""
    config = DataSourceConfig(
        name="local",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    adapter = StreamingDatasetAdapter(
        sources=[config],
        cache_size_gb=1.0,
    )
    
    # Access some items
    adapter[0]
    adapter[1]
    adapter[0]  # Hit
    
    stats = adapter.get_cache_stats()
    
    assert "hit_rate" in stats
    assert "num_items" in stats
    assert stats["num_items"] >= 2


@pytest.mark.asyncio
async def test_adapter_prefetch_batch(temp_data_dir):
    """Test batch prefetching functionality."""
    config = DataSourceConfig(
        name="local",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    adapter = StreamingDatasetAdapter(
        sources=[config],
        enable_prefetch=True,
        prefetch_ahead=5,
    )
    
    adapter.start_prefetch_worker()
    
    # Queue batch for prefetch
    batch_indices = list(range(10, 15))
    adapter.prefetch_batch(batch_indices)
    
    # Wait briefly
    await asyncio.sleep(0.2)
    
    # Items should be faster to retrieve
    item = adapter[10]
    assert item is not None
    
    adapter.stop_prefetch_worker()


@pytest.mark.asyncio
async def test_adapter_dataloader_compatibility(temp_data_dir):
    """Test adapter works with PyTorch DataLoader."""
    config = DataSourceConfig(
        name="local",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    adapter = StreamingDatasetAdapter(
        sources=[config],
        enable_prefetch=False,  # Disable for DataLoader (multiprocess)
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        adapter,
        batch_size=8,
        num_workers=0,
        shuffle=False,
    )
    
    # Get one batch
    batch = next(iter(dataloader))
    
    assert len(batch) == 8
    assert all("latent_conditions" in item for item in batch)


@pytest.mark.asyncio
async def test_adapter_clear_cache(temp_data_dir):
    """Test cache clearing."""
    config = DataSourceConfig(
        name="local",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    adapter = StreamingDatasetAdapter(sources=[config])
    
    # Populate cache
    adapter[0]
    adapter[1]
    
    stats_before = adapter.get_cache_stats()
    assert stats_before["num_items"] > 0
    
    # Clear
    adapter.clear_cache()
    
    stats_after = adapter.get_cache_stats()
    assert stats_after["num_items"] == 0


@pytest.mark.asyncio
async def test_adapter_reset_metrics(temp_data_dir):
    """Test metric resetting."""
    config = DataSourceConfig(
        name="local",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    adapter = StreamingDatasetAdapter(sources=[config])
    
    # Generate activity
    adapter[0]
    
    stats_before = adapter.get_cache_stats()
    assert stats_before["total_hits"] + stats_before["total_misses"] > 0
    
    # Reset
    adapter.reset_metrics()
    
    stats_after = adapter.get_cache_stats()
    assert stats_after["total_hits"] == 0
    assert stats_after["total_misses"] == 0


@pytest.mark.asyncio
async def test_adapter_invalid_index():
    """Test error handling for invalid index."""
    with pytest.raises(IndexError):
        # Create mock adapter
        sources = []  # Empty sources
        adapter = StreamingDatasetAdapter(sources=sources)
        # This will fail during init


@pytest.mark.asyncio
async def test_adapter_compression_overhead(temp_data_dir):
    """Test memory savings from compression."""
    config = DataSourceConfig(
        name="local",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    adapter_uncompressed = StreamingDatasetAdapter(
        sources=[config],
        compression_type="none",
        cache_size_gb=1.0,
    )
    
    adapter_compressed = StreamingDatasetAdapter(
        sources=[config],
        compression_type="zstd",
        cache_size_gb=1.0,
    )
    
    # Load same items
    for i in range(10):
        adapter_uncompressed[i]
        adapter_compressed[i]
    
    stats_uncompressed = adapter_uncompressed.get_cache_stats()
    stats_compressed = adapter_compressed.get_cache_stats()
    
    # Compressed should use less space
    assert stats_compressed["current_size_gb"] <= stats_uncompressed["current_size_gb"]


class TestAdapterMultiSource:
    """Tests for multi-source behavior."""
    
    @pytest.mark.asyncio
    async def test_multi_source_data_mapping(self, temp_data_dir):
        """Test correct mapping of sources to output keys."""
        latents_config = DataSourceConfig(
            name="latents",
            source_type="local",
            path_or_uri=str(temp_data_dir / "latents"),
        )
        
        conditions_config = DataSourceConfig(
            name="conditions",
            source_type="local",
            path_or_uri=str(temp_data_dir / "conditions"),
        )
        
        data_mapping = {
            "latents": "my_latents",
            "conditions": "my_conditions",
        }
        
        adapter = StreamingDatasetAdapter(
            sources=[latents_config, conditions_config],
            data_mapping=data_mapping,
        )
        
        item = adapter[0]
        
        assert "my_latents" in item
        assert "my_conditions" in item
        assert "latents" not in item
        assert "conditions" not in item


class TestAdapterPerformance:
    """Performance-related tests."""
    
    @pytest.mark.asyncio
    async def test_sequential_access_pattern(self, temp_data_dir, benchmark):
        """Benchmark sequential access pattern."""
        config = DataSourceConfig(
            name="local",
            source_type="local",
            path_or_uri=str(temp_data_dir / "latents"),
        )
        
        adapter = StreamingDatasetAdapter(
            sources=[config],
            enable_prefetch=False,
        )
        
        def sequential_access():
            results = []
            for i in range(10):
                results.append(adapter[i])
            return results
        
        benchmark(sequential_access)
    
    @pytest.mark.asyncio
    async def test_random_access_pattern(self, temp_data_dir, benchmark):
        """Benchmark random access pattern."""
        import random
        
        config = DataSourceConfig(
            name="local",
            source_type="local",
            path_or_uri=str(temp_data_dir / "latents"),
        )
        
        adapter = StreamingDatasetAdapter(
            sources=[config],
            enable_prefetch=False,
        )
        
        indices = list(range(100))
        
        def random_access():
            random.shuffle(indices)
            results = []
            for i in indices[:20]:
                results.append(adapter[i])
            return results
        
        benchmark(random_access)
    
    @pytest.mark.asyncio
    async def test_cache_warmup_effect(self, temp_data_dir):
        """Test cache hit rate after warmup."""
        config = DataSourceConfig(
            name="local",
            source_type="local",
            path_or_uri=str(temp_data_dir / "latents"),
        )
        
        adapter = StreamingDatasetAdapter(
            sources=[config],
            cache_size_gb=1.0,
        )
        
        # First pass - mostly misses
        for i in range(20):
            adapter[i]
        
        stats_pass1 = adapter.get_cache_stats()
        miss_rate_pass1 = 1.0 - stats_pass1["hit_rate"]
        
        adapter.reset_metrics()
        
        # Second pass - should have hits from cache
        for i in range(20):
            adapter[i]
        
        stats_pass2 = adapter.get_cache_stats()
        miss_rate_pass2 = 1.0 - stats_pass2["hit_rate"]
        
        # Second pass should have better hit rate
        assert miss_rate_pass2 < miss_rate_pass1
