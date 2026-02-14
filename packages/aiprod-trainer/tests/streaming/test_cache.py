"""
Unit tests for SmartLRUCache.

Tests:
- LRU eviction policy
- Cache hit/miss tracking
- TTL expiration
- Compression/decompression
- Concurrent access with locks
- Prefetch functionality
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest
import torch

from aiprod_trainer.streaming.cache import CacheMetrics, SmartLRUCache


@pytest.mark.asyncio
async def test_cache_basic_put_get():
    """Test basic put and get operations."""
    cache = SmartLRUCache(max_size_gb=1.0, compression_type="none")
    
    data = {"tensor": torch.randn(10, 10)}
    
    await cache.put("key_1", data, compress=False)
    retrieved = await cache.get("key_1")
    
    assert retrieved is not None
    assert torch.allclose(retrieved["tensor"], data["tensor"])


@pytest.mark.asyncio
async def test_cache_hit_rate_tracking():
    """Test hit and miss tracking."""
    cache = SmartLRUCache(max_size_gb=1.0, compression_type="none")
    
    data = {"tensor": torch.randn(10, 10)}
    await cache.put("key_1", data)
    
    # Hit
    _ = await cache.get("key_1")
    # Hit
    _ = await cache.get("key_1")
    # Miss
    _ = await cache.get("key_nonexistent")
    
    metrics = cache.get_metrics()
    assert metrics.hits == 2
    assert metrics.misses == 1
    assert metrics.hit_rate == 2 / 3


@pytest.mark.asyncio
async def test_cache_lru_eviction():
    """Test LRU eviction when cache is full."""
    cache = SmartLRUCache(
        max_size_gb=0.001,  # Very small cache to trigger eviction
        compression_type="none"
    )
    
    # Add multiple items
    for i in range(5):
        data = {"tensor": torch.randn(100, 100)}
        await cache.put(f"key_{i}", data)
    
    metrics = cache.get_metrics()
    
    # Should have evictions
    assert metrics.evictions > 0
    
    # Oldest items should be evicted
    oldest_data = await cache.get("key_0")
    assert oldest_data is None  # Should be evicted


@pytest.mark.asyncio
async def test_cache_ttl_expiration():
    """Test TTL-based cache entry expiration."""
    cache = SmartLRUCache(
        max_size_gb=1.0,
        max_ttl_seconds=0.5,  # 500ms TTL
        compression_type="none"
    )
    
    data = {"tensor": torch.randn(10, 10)}
    await cache.put("key_1", data)
    
    # Should exist immediately
    retrieved = await cache.get("key_1")
    assert retrieved is not None
    
    # Wait for TTL to expire
    await asyncio.sleep(0.6)
    
    # Should be expired
    retrieved = await cache.get("key_1")
    assert retrieved is None
    
    metrics = cache.get_metrics()
    assert metrics.misses > 0


@pytest.mark.asyncio
async def test_cache_compression():
    """Test compression/decompression."""
    cache = SmartLRUCache(
        max_size_gb=1.0,
        compression_type="zstd"
    )
    
    # Large data to see compression benefit
    large_data = {
        "tensor1": torch.randn(1000, 1000),
        "tensor2": torch.randn(1000, 1000),
    }
    
    await cache.put("key_1", large_data, compress=True)
    retrieved = await cache.get("key_1")
    
    # Data should match exactly
    assert torch.allclose(retrieved["tensor1"], large_data["tensor1"])
    assert torch.allclose(retrieved["tensor2"], large_data["tensor2"])
    
    # Check compression savings
    stats = cache.get_cache_stats()
    assert stats["compression_savings_gb"] > 0


@pytest.mark.asyncio
async def test_cache_stats():
    """Test cache statistics reporting."""
    cache = SmartLRUCache(max_size_gb=1.0, compression_type="none")
    
    # Add data
    data = {"tensor": torch.randn(100, 100)}
    await cache.put("key_1", data)
    await cache.put("key_2", data)
    
    # Access some
    await cache.get("key_1")
    await cache.get("key_1")
    
    stats = cache.get_cache_stats()
    
    assert stats["num_items"] == 2
    assert stats["total_hits"] == 2
    assert stats["total_misses"] == 0
    assert stats["hit_rate"] == 1.0


@pytest.mark.asyncio
async def test_cache_clear():
    """Test cache clearing."""
    cache = SmartLRUCache(max_size_gb=1.0)
    
    data = {"tensor": torch.randn(10, 10)}
    await cache.put("key_1", data)
    await cache.put("key_2", data)
    
    stats_before = cache.get_cache_stats()
    assert stats_before["num_items"] == 2
    
    cache.clear()
    
    stats_after = cache.get_cache_stats()
    assert stats_after["num_items"] == 0
    assert stats_after["current_size_gb"] == 0


@pytest.mark.asyncio
async def test_cache_concurrent_access(sample_tensor_dict):
    """Test concurrent access with thread safety."""
    cache = SmartLRUCache(max_size_gb=1.0)
    
    # Add initial data
    await cache.put("shared_key", sample_tensor_dict)
    
    # Concurrent gets
    tasks = [cache.get("shared_key") for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    # All should succeed
    assert all(r is not None for r in results)
    
    # All should return same data
    assert all(torch.allclose(r["latents"], results[0]["latents"]) for r in results)


@pytest.mark.asyncio
async def test_cache_update_existing_key():
    """Test updating an existing cache key (in-place replacement)."""
    cache = SmartLRUCache(max_size_gb=1.0)
    
    data1 = {"value": 1}
    data2 = {"value": 2}
    
    await cache.put("key", data1)
    retrieved1 = await cache.get("key")
    assert retrieved1["value"] == 1
    
    # Update with new data
    await cache.put("key", data2)
    retrieved2 = await cache.get("key")
    assert retrieved2["value"] == 2


@pytest.mark.asyncio
async def test_cache_prefetch():
    """Test prefetch method."""
    cache = SmartLRUCache(max_size_gb=1.0, prefetch_ahead=3)
    
    fetch_calls = []
    
    async def mock_fetch(key: str):
        fetch_calls.append(key)
        return {"data": torch.randn(10, 10)}
    
    keys = ["key_1", "key_2", "key_3", "key_4", "key_5"]
    results = await cache.prefetch(keys, mock_fetch)
    
    # Should prefetch up to prefetch_ahead items
    assert len(fetch_calls) <= 3
    assert all(results[k] for k in keys[:3])


@pytest.mark.asyncio
async def test_cache_lru_move_to_end():
    """Test that accessed items are moved to end (most recently used)."""
    cache = SmartLRUCache(max_size_gb=0.001, compression_type="none")
    
    # Add items
    for i in range(3):
        data = {"tensor": torch.randn(50, 50)}
        await cache.put(f"key_{i}", data)
    
    # Access key_0 to make it most recently used
    await cache.get("key_0")
    
    # Add new item (should evict key_1 or key_2, not key_0)
    new_data = {"tensor": torch.randn(50, 50)}
    await cache.put("key_new", new_data)
    
    # key_0 should still exist (was accessed)
    data = await cache.get("key_0")
    assert data is not None


class TestCacheMetrics:
    """Test suite for metrics tracking."""
    
    @pytest.mark.asyncio
    async def test_metrics_reset(self):
        """Test metrics reset functionality."""
        cache = SmartLRUCache()
        
        # Generate some activity
        await cache.put("key", {"data": 1})
        await cache.get("key")
        
        metrics_before = cache.get_metrics()
        assert metrics_before.hits > 0
        
        # Reset
        cache.reset_metrics()
        
        metrics_after = cache.get_metrics()
        assert metrics_after.hits == 0
        assert metrics_after.misses == 0


class TestCacheMemoryManagement:
    """Test memory-related behavior."""
    
    @pytest.mark.asyncio
    async def test_memory_estimation(self):
        """Test tensor size estimation."""
        cache = SmartLRUCache()
        
        data_dict = {
            "tensor": torch.randn(100, 100),  # 40KB (float32)
        }
        
        size = SmartLRUCache._estimate_dict_size(data_dict)
        
        # Should be roughly 100*100*4 = 40,000 bytes
        assert 39000 < size < 41000
    
    @pytest.mark.asyncio
    async def test_max_size_respected(self):
        """Test that cache respects max_size_gb limit."""
        cache = SmartLRUCache(max_size_gb=0.000001)  # Very tiny
        
        data = {"tensor": torch.randn(100, 100)}
        
        # Add multiple items - should trigger aggressive eviction
        for i in range(5):
            await cache.put(f"key_{i}", data)
        
        stats = cache.get_cache_stats()
        
        # Current size should never exceed max
        assert stats["current_size_gb"] <= 0.000001


class TestCacheErrorHandling:
    """Test error handling."""
    
    @pytest.mark.asyncio
    async def test_invalid_compression_type(self):
        """Test error on invalid compression type."""
        cache = SmartLRUCache(compression_type="invalid")
        
        # Should handle gracefully
        data = {"value": 1}
        await cache.put("key", data, compress=True)
        
        # Get should still work
        retrieved = await cache.get("key")
        assert retrieved is not None
