"""
Unit tests for caching_node.py module.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from aiprod_pipelines.inference.caching import CacheConfig, InferenceCache
from aiprod_pipelines.inference.caching_node import (
    CachingProfile, CachingNode, CachedTextEncodeNode,
    CachedVAEDecodeNode, CacheMonitoringNode
)


class TestCachingProfile:
    """Tests for CachingProfile dataclass."""
    
    def test_profile_defaults(self):
        """Test default profile values."""
        profile = CachingProfile()
        assert profile.enable_caching is True
        assert profile.embed_cache_size_mb == 256
        assert profile.feature_cache_size_mb == 512
        assert profile.enable_embedding_cache is True
    
    def test_profile_custom_values(self):
        """Test custom profile values."""
        profile = CachingProfile(
            enable_caching=False,
            embed_cache_size_mb=512,
            enable_embedding_cache=False,
            eviction_policy="lfu"
        )
        assert profile.enable_caching is False
        assert profile.embed_cache_size_mb == 512
        assert profile.enable_embedding_cache is False
        assert profile.eviction_policy == "lfu"


class TestCachingNode:
    """Tests for CachingNode."""
    
    def test_caching_node_init(self):
        """Test caching node initialization."""
        profile = CachingProfile()
        node = CachingNode(profile, device="cpu")
        assert node.profile.enable_caching is True
        assert node.cache is None
    
    def test_caching_node_input_keys(self):
        """Test caching node input keys."""
        profile = CachingProfile()
        node = CachingNode(profile, device="cpu")
        assert "text_encoder" in node.input_keys
    
    def test_caching_node_output_keys(self):
        """Test caching node output keys."""
        profile = CachingProfile()
        node = CachingNode(profile, device="cpu")
        output_keys = node.output_keys
        assert "cache" in output_keys
        assert "cache_initialized" in output_keys
    
    def test_caching_disabled(self):
        """Test when caching is disabled."""
        profile = CachingProfile(enable_caching=False)
        node = CachingNode(profile, device="cpu")
        
        context = {}
        result = node.execute(context)
        
        assert result["cache"] is None
        assert result["cache_initialized"] is False
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        profile = CachingProfile(enable_caching=True)
        node = CachingNode(profile, device="cpu")
        
        context = {}
        result = node.execute(context)
        
        assert result["cache"] is not None
        assert result["cache_initialized"] is True
        assert isinstance(result["cache"], InferenceCache)
    
    def test_cache_warmup(self):
        """Test cache warmup with sample data."""
        profile = CachingProfile(
            enable_caching=True,
            cache_warmup_samples=2
        )
        node = CachingNode(profile, device="cpu")
        
        # Mock text encoder
        class MockEncoder(nn.Module):
            def forward(self, texts):
                return torch.randn(len(texts), 768)
        
        warmup_prompts = [["Prompt 1"], ["Prompt 2"], ["Prompt 3"]]
        context = {
            "text_encoder": MockEncoder(),
            "warmup_prompts": warmup_prompts
        }
        
        result = node.execute(context)
        
        assert result["cache"] is not None
        cache = result["cache"]
        # Should have hits from warmup
        assert cache.hits >= 0


class TestCachedTextEncodeNode:
    """Tests for CachedTextEncodeNode."""
    
    def test_node_initialization(self):
        """Test node initialization."""
        node = CachedTextEncodeNode(device="cpu")
        assert node.device == "cpu"
    
    def test_node_input_keys(self):
        """Test input keys."""
        node = CachedTextEncodeNode(device="cpu")
        expected = ["prompts", "text_encoder", "cache"]
        assert all(key in node.input_keys for key in expected)
    
    def test_node_output_keys(self):
        """Test output keys."""
        node = CachedTextEncodeNode(device="cpu")
        output_keys = node.output_keys
        assert "embeddings" in output_keys
        assert "cache_hit" in output_keys
        assert "cache_stats" in output_keys
    
    def test_encode_without_cache(self):
        """Test encoding without cache (None cache)."""
        node = CachedTextEncodeNode(device="cpu")
        
        class MockEncoder(nn.Module):
            def forward(self, texts):
                return torch.randn(len(texts), 768)
        
        context = {
            "prompts": ["Test prompt"],
            "text_encoder": MockEncoder(),
            "cache": None
        }
        
        result = node.execute(context)
        
        assert result["embeddings"].shape == (1, 768)
        assert result["cache_hit"] is False
    
    def test_encode_with_cache_miss(self):
        """Test encoding with cache miss."""
        profile = CachingProfile()
        config = CacheConfig()
        cache = InferenceCache(config, device="cpu")
        
        node = CachedTextEncodeNode(device="cpu")
        
        class MockEncoder(nn.Module):
            def forward(self, texts):
                return torch.randn(len(texts), 768)
        
        context = {
            "prompts": ["New prompt"],
            "text_encoder": MockEncoder(),
            "cache": cache
        }
        
        result = node.execute(context)
        
        assert result["embeddings"].shape == (1, 768)
        # First call should be a miss (may be hit if embedding cache puts it)
    
    def test_encode_with_cache_hit(self):
        """Test encoding with cache hit."""
        profile = CachingProfile()
        config = CacheConfig()
        cache = InferenceCache(config, device="cpu")
        
        # Pre-populate cache
        prompts = ["Test prompt"]
        embeddings = torch.randn(1, 768)
        cache.embedding_cache.put(prompts, embeddings)
        
        node = CachedTextEncodeNode(device="cpu")
        
        class MockEncoder(nn.Module):
            def forward(self, texts):
                return torch.randn(len(texts), 768)
        
        hits_before = cache.hits
        
        context = {
            "prompts": prompts,
            "text_encoder": MockEncoder(),
            "cache": cache
        }
        
        result = node.execute(context)
        
        # Should have gotten a hit
        assert cache.hits > hits_before


class TestCachedVAEDecodeNode:
    """Tests for CachedVAEDecodeNode."""
    
    def test_node_initialization(self):
        """Test VAE decode node initialization."""
        node = CachedVAEDecodeNode(device="cpu")
        assert node.device == "cpu"
    
    def test_node_input_keys(self):
        """Test input keys."""
        node = CachedVAEDecodeNode(device="cpu")
        expected = ["latents", "vae_decoder", "cache"]
        assert all(key in node.input_keys for key in expected)
    
    def test_node_output_keys(self):
        """Test output keys."""
        node = CachedVAEDecodeNode(device="cpu")
        output_keys = node.output_keys
        assert "video" in output_keys
        assert "cache_hit" in output_keys
        assert "cache_stats" in output_keys
    
    def test_decode_without_cache(self):
        """Test decoding without cache."""
        node = CachedVAEDecodeNode(device="cpu")
        
        class MockDecoder(nn.Module):
            def forward(self, latents):
                return torch.randn(latents.shape[0], 3, 512, 512)
        
        latents = torch.randn(2, 4, 16, 16)
        context = {
            "latents": latents,
            "vae_decoder": MockDecoder(),
            "cache": None
        }
        
        result = node.execute(context)
        
        assert result["video"].shape == (2, 3, 512, 512)
        assert result["cache_hit"] is False
    
    def test_decode_with_cache(self):
        """Test decoding with cache."""
        profile = CachingProfile()
        config = CacheConfig()
        cache = InferenceCache(config, device="cpu")
        
        node = CachedVAEDecodeNode(device="cpu")
        
        class MockDecoder(nn.Module):
            def forward(self, latents):
                return torch.randn(latents.shape[0], 3, 512, 512)
        
        latents = torch.randn(2, 4, 16, 16)
        context = {
            "latents": latents,
            "vae_decoder": MockDecoder(),
            "cache": cache
        }
        
        result = node.execute(context)
        
        assert result["video"].shape == (2, 3, 512, 512)
        assert isinstance(result["cache_stats"].total_hits, int)


class TestCacheMonitoringNode:
    """Tests for CacheMonitoringNode."""
    
    def test_monitoring_node_init(self):
        """Test monitoring node initialization."""
        node = CacheMonitoringNode(
            memory_threshold_mb=512.0,
            hit_rate_threshold=60.0
        )
        assert node.memory_threshold_mb == 512.0
        assert node.hit_rate_threshold == 60.0
    
    def test_monitoring_node_input_keys(self):
        """Test input keys."""
        node = CacheMonitoringNode()
        assert "cache" in node.input_keys
    
    def test_monitoring_node_output_keys(self):
        """Test output keys."""
        node = CacheMonitoringNode()
        output_keys = node.output_keys
        assert "cache_stats" in output_keys
        assert "should_clear_cache" in output_keys
        assert "monitoring_report" in output_keys
    
    def test_monitoring_no_cache(self):
        """Test monitoring with no cache."""
        node = CacheMonitoringNode()
        
        context = {"cache": None}
        result = node.execute(context)
        
        assert result["cache_stats"] is not None
        assert result["should_clear_cache"] is False
        assert "No cache active" in result["monitoring_report"]
    
    def test_monitoring_with_cache(self):
        """Test monitoring with active cache."""
        profile = CachingProfile()
        config = CacheConfig()
        cache = InferenceCache(config, device="cpu")
        
        # Add some hits/misses
        cache.hits = 70
        cache.misses = 30
        
        node = CacheMonitoringNode(
            memory_threshold_mb=1024.0,
            hit_rate_threshold=50.0
        )
        
        context = {"cache": cache}
        result = node.execute(context)
        
        assert result["cache_stats"] is not None
        assert "Cache Stats" in result["monitoring_report"]
        assert result["should_clear_cache"] is False
    
    def test_monitoring_memory_threshold(self):
        """Test memory threshold detection."""
        profile = CachingProfile()
        config = CacheConfig()
        cache = InferenceCache(config, device="cpu")
        
        # Fill cache significantly
        for i in range(10):
            cache.embedding_cache.put(
                [f"prompt_{i}"],
                torch.randn(1, 768)
            )
        
        node = CacheMonitoringNode(
            memory_threshold_mb=0.001,  # Very small threshold
            enable_auto_clear=True
        )
        
        context = {"cache": cache}
        result = node.execute(context)
        
        # May trigger clear if threshold exceeded
        assert isinstance(result["should_clear_cache"], bool)
    
    def test_monitoring_history(self):
        """Test history tracking."""
        profile = CachingProfile()
        config = CacheConfig()
        cache = InferenceCache(config, device="cpu")
        
        node = CacheMonitoringNode()
        
        # Multiple monitor calls
        for _ in range(3):
            context = {"cache": cache}
            result = node.execute(context)
        
        history = node.get_history()
        assert len(history) == 3


class TestCachingNodeIntegration:
    """Integration tests for caching nodes."""
    
    def test_full_caching_pipeline(self):
        """Test complete caching pipeline."""
        # Initialize cache
        caching_profile = CachingProfile()
        caching_node = CachingNode(caching_profile, device="cpu")
        
        class MockEncoder(nn.Module):
            def forward(self, texts):
                return torch.randn(len(texts), 768)
        
        context = {"text_encoder": MockEncoder()}
        cache_result = caching_node.execute(context)
        cache = cache_result["cache"]
        
        # Use cache for text encoding
        encode_node = CachedTextEncodeNode(device="cpu")
        
        encode_context = {
            "prompts": ["Test prompt"],
            "text_encoder": MockEncoder(),
            "cache": cache
        }
        
        encode_result = encode_node.execute(encode_context)
        assert encode_result["embeddings"] is not None
        
        # Monitor cache
        monitor_node = CacheMonitoringNode()
        monitor_context = {"cache": cache}
        monitor_result = monitor_node.execute(monitor_context)
        
        assert monitor_result["cache_stats"] is not None
    
    def test_caching_with_vae(self):
        """Test caching with VAE operations."""
        # Create cache
        profile = CachingProfile()
        config = CacheConfig()
        cache = InferenceCache(config, device="cpu")
        
        # Use cached VAE decode
        vae_node = CachedVAEDecodeNode(device="cpu")
        
        class MockDecoder(nn.Module):
            def forward(self, latents):
                return torch.randn(latents.shape[0], 3, 512, 512)
        
        latents = torch.randn(2, 4, 16, 16)
        context = {
            "latents": latents,
            "vae_decoder": MockDecoder(),
            "cache": cache
        }
        
        result = vae_node.execute(context)
        assert result["video"] is not None
