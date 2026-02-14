"""
Unit tests for caching.py module.
"""

import pytest
import torch
from typing import List

from aiprod_pipelines.inference.caching import (
    CacheConfig, CacheEntry, CacheStatistics,
    EmbeddingCache, FeatureCache, VAECache, InferenceCache
)


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.embed_cache_size_mb == 256
        assert config.feature_cache_size_mb == 512
        assert config.vae_cache_size_mb == 128
        assert config.enable_embedding_cache is True
        assert config.eviction_policy == "lru"
    
    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = CacheConfig(
            embed_cache_size_mb=512,
            feature_cache_size_mb=1024,
            vae_cache_size_mb=256,
            enable_embedding_cache=False
        )
        assert config.embed_cache_size_mb == 512
        assert config.feature_cache_size_mb == 1024
        assert config.enable_embedding_cache is False
    
    def test_config_validation_size(self):
        """Test validation of cache sizes."""
        with pytest.raises(AssertionError):
            CacheConfig(embed_cache_size_mb=0)
        with pytest.raises(AssertionError):
            CacheConfig(embed_cache_size_mb=-1)
    
    def test_config_validation_eviction(self):
        """Test validation of eviction policy."""
        with pytest.raises(AssertionError):
            CacheConfig(eviction_policy="invalid")
    
    def test_valid_eviction_policies(self):
        """Test all valid eviction policies."""
        for policy in ["lru", "lfu", "fifo"]:
            config = CacheConfig(eviction_policy=policy)
            assert config.eviction_policy == policy


class TestCacheEntry:
    """Tests for CacheEntry wrapper."""
    
    def test_entry_creation(self):
        """Test cache entry creation."""
        value = torch.randn(4, 8)
        entry = CacheEntry(
            value=value,
            key="test_key",
            size_bytes=256
        )
        assert entry.key == "test_key"
        assert entry.access_count == 0
        assert entry.size_bytes == 256
    
    def test_entry_update_access(self):
        """Test access metadata update."""
        value = torch.randn(4, 8)
        entry = CacheEntry(value=value, key="test", size_bytes=256)
        
        assert entry.access_count == 0
        entry.update_access()
        assert entry.access_count == 1
        entry.update_access()
        assert entry.access_count == 2


class TestCacheStatistics:
    """Tests for CacheStatistics dataclass."""
    
    def test_stats_defaults(self):
        """Test default statistics values."""
        stats = CacheStatistics()
        assert stats.total_hits == 0
        assert stats.total_misses == 0
        assert stats.hit_rate_percent == 0.0
    
    def test_hit_rate_calculation(self):
        """Test hit rate percentage calculation."""
        stats = CacheStatistics(total_hits=70, total_misses=30)
        assert stats.hit_rate_percent == pytest.approx(70.0)
    
    def test_hit_rate_no_samples(self):
        """Test hit rate when no samples."""
        stats = CacheStatistics(total_hits=0, total_misses=0)
        assert stats.hit_rate_percent == 0.0
    
    def test_speedup_factor(self):
        """Test speedup factor calculation."""
        stats = CacheStatistics(
            inference_time_ms_without_cache=100.0,
            inference_time_ms_with_cache=50.0
        )
        assert stats.speedup_factor == pytest.approx(2.0)
    
    def test_speedup_invalid(self):
        """Test speedup with invalid times."""
        stats = CacheStatistics(
            inference_time_ms_without_cache=0.0,
            inference_time_ms_with_cache=50.0
        )
        assert stats.speedup_factor == 1.0


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""
    
    def test_cache_initialization(self):
        """Test embedding cache initialization."""
        cache = EmbeddingCache(max_size_mb=64)
        assert cache.max_size_mb == 64
        assert len(cache.cache) == 0
    
    def test_cache_put_get(self):
        """Test basic cache put and get."""
        cache = EmbeddingCache(max_size_mb=64)
        texts = ["Hello world", "Test prompt"]
        embeddings = torch.randn(2, 768)
        
        cache.put(texts, embeddings)
        assert len(cache.cache) == 1
        
        retrieved = cache.get(texts)
        assert retrieved is not None
        assert torch.allclose(retrieved, embeddings)
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = EmbeddingCache(max_size_mb=64)
        texts = ["Hello"]
        embeddings = torch.randn(1, 768)
        cache.put(texts, embeddings)
        
        other_texts = ["Different"]
        retrieved = cache.get(other_texts)
        assert retrieved is None
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache full."""
        cache = EmbeddingCache(max_size_mb=1)  # Very small cache
        
        # Add entries until eviction happens
        for i in range(5):
            texts = [f"Text {i}"]
            embeddings = torch.randn(1, 768)
            cache.put(texts, embeddings)
        
        # Cache should have evicted old entries
        assert len(cache.cache) < 5
    
    def test_cache_clear(self):
        """Test cache clearing."""
        cache = EmbeddingCache(max_size_mb=64)
        texts = ["Test"]
        embeddings = torch.randn(1, 768)
        cache.put(texts, embeddings)
        
        assert len(cache.cache) == 1
        cache.clear()
        assert len(cache.cache) == 0
        assert cache.current_size_bytes == 0
    
    def test_cache_size_computation(self):
        """Test cache size computation."""
        cache = EmbeddingCache(max_size_mb=64)
        texts = ["Test"]
        embeddings = torch.randn(1, 768)
        cache.put(texts, embeddings)
        
        size_mb = cache.get_size_mb()
        assert size_mb > 0
        assert size_mb <= 64


class TestFeatureCache:
    """Tests for FeatureCache."""
    
    def test_feature_cache_init(self):
        """Test feature cache initialization."""
        cache = FeatureCache(max_size_mb=128)
        assert cache.max_size_mb == 128
        assert len(cache.cache) == 0
    
    def test_feature_put_get(self):
        """Test feature cache put and get."""
        cache = FeatureCache(max_size_mb=128)
        layer_name = "transformer_layer_4"
        input_hash = "abc123"
        features = torch.randn(4, 256, 768)
        
        cache.put(layer_name, input_hash, features)
        retrieved = cache.get(layer_name, input_hash)
        
        assert retrieved is not None
        assert torch.allclose(retrieved, features)
    
    def test_feature_cache_miss(self):
        """Test feature cache miss."""
        cache = FeatureCache(max_size_mb=128)
        retrieved = cache.get("nonexistent_layer", "nonexistent_hash")
        assert retrieved is None
    
    def test_feature_cache_eviction(self):
        """Test feature cache eviction."""
        cache = FeatureCache(max_size_mb=1)  # Very small
        
        for i in range(3):
            layer = f"layer_{i}"
            features = torch.randn(4, 256, 768)
            cache.put(layer, f"hash_{i}", features)
        
        assert len(cache.cache) < 3


class TestVAECache:
    """Tests for VAECache."""
    
    def test_vae_cache_init(self):
        """Test VAE cache initialization."""
        cache = VAECache(max_size_mb=32)
        assert cache.max_size_mb == 32
    
    def test_vae_encode_cache(self):
        """Test VAE encode output caching."""
        cache = VAECache(max_size_mb=32)
        
        input_tensor = torch.randn(2, 3, 64, 64)
        latents = torch.randn(2, 4, 16, 16)
        input_hash = "abc123"
        
        cache.put("encode", input_hash, latents)
        retrieved = cache.get("encode", input_hash)
        
        assert retrieved is not None
        assert torch.allclose(retrieved, latents)
    
    def test_vae_decode_cache(self):
        """Test VAE decode output caching."""
        cache = VAECache(max_size_mb=32)
        
        latents = torch.randn(2, 4, 16, 16)
        video = torch.randn(2, 3, 512, 512)
        input_hash = "def456"
        
        cache.put("decode", input_hash, video)
        retrieved = cache.get("decode", input_hash)
        
        assert retrieved is not None
        assert torch.allclose(retrieved, video)


class TestInferenceCache:
    """Tests for unified InferenceCache."""
    
    def test_inference_cache_init(self):
        """Test inference cache initialization."""
        config = CacheConfig()
        cache = InferenceCache(config, device="cpu")
        
        assert cache.embedding_cache is not None
        assert cache.feature_cache is not None
        assert cache.vae_cache is not None
    
    def test_inference_cache_disabled(self):
        """Test with caching disabled."""
        config = CacheConfig(enable_embedding_cache=False)
        cache = InferenceCache(config, device="cpu")
        
        assert cache.embedding_cache is None
    
    def test_get_embeddings_hit(self):
        """Test embedding retrieval with cache hit."""
        config = CacheConfig()
        cache = InferenceCache(config, device="cpu")
        
        texts = ["Test prompt"]
        embeddings = torch.randn(1, 768)
        
        # Put in cache
        cache.embedding_cache.put(texts, embeddings)
        
        # Retrieve - should hit
        hits_before = cache.hits
        retrieved = cache.get_embeddings(texts, nn.Identity(), "cpu")
        
        assert cache.hits > hits_before
    
    def test_get_statistics(self):
        """Test statistics computation."""
        config = CacheConfig()
        cache = InferenceCache(config, device="cpu")
        
        cache.hits = 10
        cache.misses = 5
        
        stats = cache.get_statistics()
        assert stats.total_hits == 10
        assert stats.total_misses == 5
        assert stats.hit_rate_percent == pytest.approx(66.67, rel=0.1)
    
    def test_cache_clear_all(self):
        """Test clearing all caches."""
        config = CacheConfig()
        cache = InferenceCache(config, device="cpu")
        
        # Put some data
        cache.embedding_cache.put(["test"], torch.randn(1, 768))
        assert len(cache.embedding_cache.cache) == 1
        
        # Clear
        cache.clear()
        assert len(cache.embedding_cache.cache) == 0
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        config = CacheConfig()
        cache = InferenceCache(config, device="cpu")
        
        cache.hits = 10
        cache.misses = 5
        
        cache.reset_statistics()
        assert cache.hits == 0
        assert cache.misses == 0


class TestCachingIntegration:
    """Integration tests for caching system."""
    
    def test_full_cache_workflow(self):
        """Test complete caching workflow."""
        # Create cache
        config = CacheConfig(
            embed_cache_size_mb=64,
            feature_cache_size_mb=128,
            vae_cache_size_mb=32
        )
        cache = InferenceCache(config, device="cpu")
        
        # Simulate multiple embeddings with repeats
        prompts_list = [
            ["Prompt A"],
            ["Prompt B"],
            ["Prompt A"],  # Repeat - should hit
            ["Prompt C"],
            ["Prompt B"],  # Repeat - should hit
        ]
        
        for prompts in prompts_list:
            embeddings = torch.randn(1, 768)
            cache.embedding_cache.put(prompts, embeddings)
        
        # Check statistics
        stats = cache.embedding_cache.get_stats()
        assert stats["num_entries"] > 0
        assert stats["utilization_percent"] > 0
    
    def test_mixed_cache_operations(self):
        """Test mixed operations across cache types."""
        config = CacheConfig()
        cache = InferenceCache(config, device="cpu")
        
        # Embedding cache
        cache.embedding_cache.put(["test"], torch.randn(1, 768))
        
        # Feature cache
        cache.feature_cache.put("layer1", "hash1", torch.randn(2, 256, 768))
        
        # VAE cache
        cache.vae_cache.put("decode", "vae_hash", torch.randn(2, 3, 512, 512))
        
        # Verify all populated
        assert len(cache.embedding_cache.cache) == 1
        assert len(cache.feature_cache.cache) == 1
        assert len(cache.vae_cache.cache) == 1
        
        # Clear all
        cache.clear()
        
        assert len(cache.embedding_cache.cache) == 0
        assert len(cache.feature_cache.cache) == 0
        assert len(cache.vae_cache.cache) == 0
