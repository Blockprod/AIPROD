"""
Performance benchmarks for streaming architecture.

Benchmarks:
- Cache vs no-cache throughput
- Prefetch effectiveness
- Multi-source overhead
- Compression vs speed tradeoff
"""

import asyncio
import time
from pathlib import Path

import pytest
import torch

from aiprod_trainer.streaming import DataSourceConfig, SmartLRUCache, StreamingDatasetAdapter


class TestPerformanceBenchmarks:
    """Performance benchmark tests using pytest-benchmark."""
    
    def test_local_source_fetch_speed(self, temp_data_dir, benchmark):
        """Benchmark fetching from local source."""
        from aiprod_trainer.streaming.sources import LocalDataSource
        
        config = DataSourceConfig(
            name="local",
            source_type="local",
            path_or_uri=str(temp_data_dir / "latents"),
        )
        
        source = LocalDataSource(config)
        
        def fetch_one():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(source.fetch_file("latent_0050.pt"))
            loop.close()
            return result
        
        benchmark(fetch_one)
    
    def test_cache_hit_speed(self, benchmark):
        """Benchmark cache hit speed."""
        cache = SmartLRUCache(compression_type="none")
        
        data = {"tensor": torch.randn(100, 100)}
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(cache.put("key", data))
        
        def cache_get():
            result = loop.run_until_complete(cache.get("key"))
            return result
        
        benchmark(cache_get)
        loop.close()
    
    def test_cache_miss_speed(self, benchmark):
        """Benchmark cache miss speed (should be fast - just check)."""
        cache = SmartLRUCache()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        def cache_miss():
            result = loop.run_until_complete(cache.get("nonexistent_key"))
            return result
        
        benchmark(cache_miss)
        loop.close()
    
    def test_compression_encode_speed(self, benchmark):
        """Benchmark compression speed."""
        cache = SmartLRUCache(compression_type="zstd")
        
        data = {"tensor": torch.randn(1000, 1000)}
        
        def compress_dict():
            return cache._compress_dict(data)
        
        benchmark(compress_dict)
    
    def test_compression_decode_speed(self, benchmark):
        """Benchmark decompression speed."""
        cache = SmartLRUCache(compression_type="zstd")
        
        data = {"tensor": torch.randn(1000, 1000)}
        compressed = cache._compress_dict(data)
        
        def decompress_dict():
            return cache._decompress_dict(compressed)
        
        benchmark(decompress_dict)


class TestThroughputComparison:
    """Compare throughput with/without caching."""
    
    @pytest.mark.asyncio
    async def test_throughput_no_cache(self, temp_data_dir):
        """Measure throughput WITHOUT caching."""
        from aiprod_trainer.streaming.sources import LocalDataSource
        
        config = DataSourceConfig(
            name="local",
            source_type="local",
            path_or_uri=str(temp_data_dir / "latents"),
        )
        
        source = LocalDataSource(config)
        
        start = time.time()
        items_loaded = 0
        
        # Load 50 items sequentially (all misses, no cache)
        for i in range(50):
            file_name = f"latent_{i:04d}.pt"
            data = await source.fetch_file(file_name)
            items_loaded += 1
        
        elapsed = time.time() - start
        throughput_no_cache = items_loaded / elapsed
        
        return {
            "throughput": throughput_no_cache,
            "items": items_loaded,
            "time_sec": elapsed,
        }
    
    @pytest.mark.asyncio
    async def test_throughput_with_cache(self, temp_data_dir):
        """Measure throughput WITH caching."""
        from aiprod_trainer.streaming.sources import LocalDataSource
        
        config = DataSourceConfig(
            name="local",
            source_type="local",
            path_or_uri=str(temp_data_dir / "latents"),
        )
        
        cache = SmartLRUCache(max_size_gb=10.0, compression_type="zstd")
        source = LocalDataSource(config)
        
        start = time.time()
        items_loaded = 0
        
        # Load 50 items with caching
        for i in range(50):
            file_name = f"latent_{i:04d}.pt"
            
            # Try cache first
            cached = await cache.get(file_name)
            if cached is None:
                # Fetch from source
                data = await source.fetch_file(file_name)
                await cache.put(file_name, data, compress=True)
            
            items_loaded += 1
        
        elapsed = time.time() - start
        throughput_with_cache = items_loaded / elapsed
        
        cache_stats = cache.get_cache_stats()
        
        return {
            "throughput": throughput_with_cache,
            "items": items_loaded,
            "time_sec": elapsed,
            "hit_rate": cache_stats["hit_rate"],
        }
    
    @pytest.mark.asyncio
    async def test_throughput_with_prefetch(self, temp_data_dir):
        """Measure throughput WITH caching and prefetch."""
        from aiprod_trainer.streaming.cache import AsyncPrefetcher
        from aiprod_trainer.streaming.sources import LocalDataSource
        
        config = DataSourceConfig(
            name="local",
            source_type="local",
            path_or_uri=str(temp_data_dir / "latents"),
        )
        
        cache = SmartLRUCache(max_size_gb=10.0)
        source = LocalDataSource(config)
        
        async def fetch_file(key):
            return await source.fetch_file(key)
        
        prefetcher = AsyncPrefetcher(cache, fetch_file, queue_size=10)
        prefetcher.start()
        
        start = time.time()
        items_loaded = 0
        
        # Load 50 items with prefetch
        for i in range(50):
            file_name = f"latent_{i:04d}.pt"
            
            # Queue ALL next items for prefetch
            for j in range(i + 1, min(i + 5, 100)):
                prefetcher.queue_item(f"latent_{j:04d}.pt")
            
            # Get current item
            data = await prefetcher.get(file_name)
            items_loaded += 1
        
        elapsed = time.time() - start
        throughput_prefetch = items_loaded / elapsed
        
        cache_stats = cache.get_cache_stats()
        
        prefetcher.stop()
        
        return {
            "throughput": throughput_prefetch,
            "items": items_loaded,
            "time_sec": elapsed,
            "hit_rate": cache_stats["hit_rate"],
            "prefetch_hits": cache_stats["prefetch_hits"],
        }


class TestMemoryUsage:
    """Memory efficiency tests."""
    
    @pytest.mark.asyncio
    async def test_compression_savings(self, temp_data_dir):
        """Measure actual compression savings."""
        cache_uncompressed = SmartLRUCache(
            max_size_gb=10.0,
            compression_type="none"
        )
        
        cache_compressed = SmartLRUCache(
            max_size_gb=10.0,
            compression_type="zstd"
        )
        
        from aiprod_trainer.streaming.sources import LocalDataSource
        
        config = DataSourceConfig(
            name="local",
            source_type="local",
            path_or_uri=str(temp_data_dir / "latents"),
        )
        
        source = LocalDataSource(config)
        
        # Load 30 items into both caches
        for i in range(30):
            file_name = f"latent_{i:04d}.pt"
            data = await source.fetch_file(file_name)
            
            await cache_uncompressed.put(file_name, data, compress=False)
            await cache_compressed.put(file_name, data, compress=True)
        
        stats_uncompressed = cache_uncompressed.get_cache_stats()
        stats_compressed = cache_compressed.get_cache_stats()
        
        size_uncompressed = stats_uncompressed["current_size_gb"]
        size_compressed = stats_compressed["current_size_gb"]
        
        compression_ratio = size_uncompressed / size_compressed
        savings_gb = size_uncompressed - size_compressed
        
        return {
            "uncompressed_gb": size_uncompressed,
            "compressed_gb": size_compressed,
            "ratio": compression_ratio,
            "savings_gb": savings_gb,
        }
    
    @pytest.mark.asyncio
    async def test_cache_eviction_behavior(self):
        """Test cache eviction doesn't cause memory explosion."""
        cache = SmartLRUCache(max_size_gb=0.01)  # 10MB
        
        initial_stats = cache.get_cache_stats()
        
        # Add many items that should trigger eviction
        for i in range(100):
            data = {"tensor": torch.randn(100, 100)}
            await cache.put(f"key_{i}", data)
        
        final_stats = cache.get_cache_stats()
        
        # Final size should be under max
        assert final_stats["current_size_gb"] <= 0.01
        assert final_stats["total_evictions"] > 0


class TestLatencyCharacteristics:
    """Latency distribution tests."""
    
    @pytest.mark.asyncio
    async def test_p50_p99_latency_cache_hit(self, temp_data_dir):
        """Measure P50/P99 latency for cache hits."""
        cache = SmartLRUCache(compression_type="none")
        
        from aiprod_trainer.streaming.sources import LocalDataSource
        
        config = DataSourceConfig(
            name="local",
            source_type="local",
            path_or_uri=str(temp_data_dir / "latents"),
        )
        
        source = LocalDataSource(config)
        
        # Warm up cache
        for i in range(10):
            data = await source.fetch_file(f"latent_{i:04d}.pt")
            await cache.put(f"latent_{i:04d}.pt", data)
        
        # Measure cache hit latencies
        latencies = []
        for _ in range(100):
            start = time.time()
            await cache.get(f"latent_0000.pt")
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
        
        latencies.sort()
        p50 = latencies[50]
        p99 = latencies[99]
        
        return {
            "p50_ms": p50,
            "p99_ms": p99,
            "mean_ms": sum(latencies) / len(latencies),
            "max_ms": max(latencies),
        }
    
    @pytest.mark.asyncio
    async def test_p50_p99_latency_cache_miss(self, temp_data_dir):
        """Measure P50/P99 latency for cache misses."""
        cache = SmartLRUCache(compression_type="none")
        
        from aiprod_trainer.streaming.sources import LocalDataSource
        
        config = DataSourceConfig(
            name="local",
            source_type="local",
            path_or_uri=str(temp_data_dir / "latents"),
        )
        
        source = LocalDataSource(config)
        
        # Measure cache miss latencies (from source)
        latencies = []
        for i in range(100):
            start = time.time()
            data = await source.fetch_file(f"latent_{i:04d}.pt")
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
        
        latencies.sort()
        p50 = latencies[50]
        p99 = latencies[99]
        
        return {
            "p50_ms": p50,
            "p99_ms": p99,
            "mean_ms": sum(latencies) / len(latencies),
            "max_ms": max(latencies),
        }
