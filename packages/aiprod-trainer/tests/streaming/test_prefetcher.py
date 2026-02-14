"""
Unit tests for AsyncPrefetcher.

Tests:
- Prefetch queue behavior
- Worker task lifecycle
- Non-blocking queue operations
- Prefetch hit rate
"""

import asyncio

import pytest
import torch

from aiprod_trainer.streaming.cache import AsyncPrefetcher, SmartLRUCache


@pytest.mark.asyncio
async def test_prefetcher_basic_start_stop():
    """Test starting and stopping prefetcher worker."""
    cache = SmartLRUCache(max_size_gb=1.0)
    
    async def dummy_fetch(key):
        return {"data": torch.randn(10, 10)}
    
    prefetcher = AsyncPrefetcher(cache, dummy_fetch, queue_size=5)
    
    # Start worker
    prefetcher.start()
    assert prefetcher._task is not None
    assert not prefetcher._task.done()
    
    # Stop worker
    prefetcher.stop()
    await asyncio.sleep(0.1)
    assert prefetcher._task.done()


@pytest.mark.asyncio
async def test_prefetcher_queue_item():
    """Test queueing items for prefetch."""
    cache = SmartLRUCache(max_size_gb=1.0)
    
    fetch_count = {"value": 0}
    
    async def counting_fetch(key):
        fetch_count["value"] += 1
        await asyncio.sleep(0.05)
        return {"data": torch.randn(10, 10), "key": key}
    
    prefetcher = AsyncPrefetcher(cache, counting_fetch, queue_size=5)
    prefetcher.start()
    
    # Queue items
    for i in range(3):
        prefetcher.queue_item(f"key_{i}")
    
    # Wait for prefetch to complete
    await asyncio.sleep(0.5)
    
    # Items should be in cache
    assert fetch_count["value"] > 0


@pytest.mark.asyncio
async def test_prefetcher_get_prefetched():
    """Test getting prefetched items."""
    cache = SmartLRUCache(max_size_gb=1.0)
    
    async def test_fetch(key):
        await asyncio.sleep(0.02)
        return {"data": torch.randn(10, 10), "key": key}
    
    prefetcher = AsyncPrefetcher(cache, test_fetch, queue_size=5)
    prefetcher.start()
    
    # Queue item for prefetch
    prefetcher.queue_item("test_key")
    
    # Wait briefly
    await asyncio.sleep(0.1)
    
    # Get should return quickly if prefetched
    result = await prefetcher.get("test_key")
    
    assert result is not None
    assert result["key"] == "test_key"
    
    prefetcher.stop()


@pytest.mark.asyncio
async def test_prefetcher_get_not_prefetched():
    """Test getting item that wasn't prefetched (blocks until fetched)."""
    cache = SmartLRUCache(max_size_gb=1.0)
    
    fetch_count = {"value": 0}
    
    async def counting_fetch(key):
        fetch_count["value"] += 1
        await asyncio.sleep(0.05)
        return {"data": torch.randn(10, 10), "key": key}
    
    prefetcher = AsyncPrefetcher(cache, counting_fetch, queue_size=5)
    
    # Get without prefetch - should fetch on-demand
    result = await prefetcher.get("unprefetched_key")
    
    assert result is not None
    assert fetch_count["value"] == 1


@pytest.mark.asyncio
async def test_prefetcher_queue_full():
    """Test behavior when prefetch queue is full."""
    cache = SmartLRUCache(max_size_gb=1.0)
    
    async def slow_fetch(key):
        await asyncio.sleep(0.2)
        return {"data": torch.randn(10, 10)}
    
    prefetcher = AsyncPrefetcher(cache, slow_fetch, queue_size=2)
    prefetcher.start()
    
    # Fill queue
    prefetcher.queue_item("key_1")
    prefetcher.queue_item("key_2")
    
    # This should not raise, just skip
    prefetcher.queue_item("key_3")
    prefetcher.queue_item("key_4")
    
    prefetcher.stop()


@pytest.mark.asyncio
async def test_prefetcher_already_cached():
    """Test prefetch skips already-cached items."""
    cache = SmartLRUCache(max_size_gb=1.0)
    
    # Pre-populate cache
    await cache.put("key_1", {"data": torch.randn(10, 10)})
    
    fetch_count = {"value": 0}
    
    async def counting_fetch(key):
        fetch_count["value"] += 1
        return {"data": torch.randn(10, 10)}
    
    prefetcher = AsyncPrefetcher(cache, counting_fetch, queue_size=5)
    prefetcher.start()
    
    # Queue already-cached item
    prefetcher.queue_item("key_1")
    
    await asyncio.sleep(0.2)
    
    # Fetch should not be called
    assert fetch_count["value"] == 0
    
    prefetcher.stop()


@pytest.mark.asyncio
async def test_prefetcher_multiple_items():
    """Test prefetching multiple items concurrently."""
    cache = SmartLRUCache(max_size_gb=1.0)
    
    items_fetched = []
    
    async def tracking_fetch(key):
        items_fetched.append(key)
        await asyncio.sleep(0.02)
        return {"data": torch.randn(10, 10), "key": key}
    
    prefetcher = AsyncPrefetcher(cache, tracking_fetch, queue_size=10)
    prefetcher.start()
    
    # Queue multiple items
    for i in range(5):
        prefetcher.queue_item(f"key_{i}")
    
    # Wait for prefetch
    await asyncio.sleep(0.3)
    
    # Should have fetched items
    assert len(items_fetched) > 0
    
    prefetcher.stop()


@pytest.mark.asyncio
async def test_prefetcher_prefetch_hit_tracking():
    """Test tracking of prefetch hits."""
    cache = SmartLRUCache(max_size_gb=1.0)
    
    async def test_fetch(key):
        await asyncio.sleep(0.05)
        return {"data": torch.randn(10, 10), "key": key}
    
    prefetcher = AsyncPrefetcher(cache, test_fetch, queue_size=5)
    prefetcher.start()
    
    # Queue for prefetch
    prefetcher.queue_item("key_1")
    await asyncio.sleep(0.2)
    
    # Get should be a prefetch hit
    result = await prefetcher.get("key_1")
    
    # Check metrics
    metrics = cache.get_metrics()
    assert metrics.prefetch_hits > 0
    
    prefetcher.stop()


@pytest.mark.asyncio
async def test_prefetcher_respects_limit():
    """Test that prefetcher respects prefetch_ahead limit."""
    cache = SmartLRUCache(max_size_gb=1.0, prefetch_ahead=2)
    
    items_fetched = []
    
    async def tracking_fetch(key):
        items_fetched.append(key)
        await asyncio.sleep(0.05)
        return {"data": torch.randn(10, 10)}
    
    prefetcher = AsyncPrefetcher(cache, tracking_fetch, queue_size=10)
    prefetcher.start()
    
    # Queue many items but should only prefetch up to limit
    for i in range(10):
        prefetcher.queue_item(f"key_{i}")
    
    await asyncio.sleep(0.3)
    
    # Should have only fetched up to prefetch_ahead items
    assert len(items_fetched) <= 2
    
    prefetcher.stop()


@pytest.mark.asyncio
async def test_prefetcher_fetch_exception_handling():
    """Test prefetcher handles fetch exceptions gracefully."""
    cache = SmartLRUCache(max_size_gb=1.0)
    
    async def failing_fetch(key):
        if key == "bad_key":
            raise Exception("Simulated fetch error")
        return {"data": torch.randn(10, 10)}
    
    prefetcher = AsyncPrefetcher(cache, failing_fetch, queue_size=5)
    prefetcher.start()
    
    # Queue bad key
    prefetcher.queue_item("bad_key")
    
    # Wait
    await asyncio.sleep(0.2)
    
    # Worker should still be running
    assert not prefetcher._task.done()
    
    prefetcher.stop()


class TestPrefetcherIntegration:
    """Integration tests for prefetcher with cache."""
    
    @pytest.mark.asyncio
    async def test_prefetcher_improves_hit_rate(self):
        """Test that prefetcher improves cache hit rate."""
        cache = SmartLRUCache(max_size_gb=1.0)
        
        async def test_fetch(key):
            await asyncio.sleep(0.01)
            return {"data": torch.randn(10, 10)}
        
        prefetcher = AsyncPrefetcher(cache, test_fetch, queue_size=10)
        prefetcher.start()
        
        # Prefetch a sequence
        sequence = [f"key_{i}" for i in range(10)]
        for key in sequence[:5]:
            prefetcher.queue_item(key)
        
        # Wait
        await asyncio.sleep(0.2)
        
        # Get items - should have good hit rate
        for key in sequence[:5]:
            await prefetcher.get(key)
        
        metrics = cache.get_metrics()
        assert metrics.hit_rate > 0.5
        
        prefetcher.stop()
    
    @pytest.mark.asyncio
    async def test_prefetcher_with_async_loop(self):
        """Test prefetcher works correctly in async loop."""
        cache = SmartLRUCache(max_size_gb=1.0)
        
        async def test_fetch(key):
            await asyncio.sleep(0.02)
            return {"data": torch.randn(10, 10)}
        
        prefetcher = AsyncPrefetcher(cache, test_fetch, queue_size=5)
        
        async def training_loop():
            prefetcher.start()
            
            results = []
            for i in range(5):
                # Prefetch next item
                prefetcher.queue_item(f"key_{i+1}")
                
                # Get current item
                result = await prefetcher.get(f"key_{i}")
                results.append(result)
            
            prefetcher.stop()
            return results
        
        results = await training_loop()
        assert len(results) == 5
        assert all(r is not None for r in results)
