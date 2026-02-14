"""
Unit tests for preset caching functionality.

Tests PresetCache class and cached preset wrapper functions.
"""

import pytest
from typing import Dict, Any

from aiprod_pipelines.inference import (
    InferenceGraph,
    PresetFactory,
    PresetCache,
    preset_cache_config,
    preset_cache_clear,
    preset_cache_size,
    preset_cached_t2v_one_stage,
    preset_cached_t2v_two_stages,
    preset_cached_distilled_fast,
    preset_cached_ic_lora,
    preset_cached_keyframe_interpolation,
    preset_cached_t2v_one_stage_adaptive,
    preset_cached_t2v_two_stages_adaptive,
    preset_cached_distilled_fast_adaptive,
    preset_cached_ic_lora_adaptive,
    preset_cached_keyframe_interpolation_adaptive,
    preset_cached_t2v_one_stage_quantized,
    preset_cached_t2v_two_stages_quantized,
    preset_cached_distilled_fast_quantized,
    preset_cached_ic_lora_quantized,
    preset_cached_keyframe_interpolation_quantized,
)


class MockModel:
    """Mock model for caching tests."""
    def __init__(self, name: str = "model"):
        self.name = name


class MockEncoder:
    """Mock text encoder for caching tests."""
    def __init__(self, name: str = "encoder"):
        self.name = name


class MockScheduler:
    """Mock scheduler for caching tests."""
    def __init__(self, name: str = "scheduler"):
        self.name = name


class MockDecoder:
    """Mock VAE decoder for caching tests."""
    def __init__(self, name: str = "decoder"):
        self.name = name


class MockUpsampler:
    """Mock upsampler for caching tests."""
    def __init__(self, name: str = "upsampler"):
        self.name = name


class TestPresetCache:
    """Tests for PresetCache class."""
    
    def test_cache_initialization(self):
        """Test cache initialization with default size."""
        cache = PresetCache()
        assert cache.max_size == 32
        assert cache.size() == 0
    
    def test_cache_custom_size(self):
        """Test cache initialization with custom size."""
        cache = PresetCache(max_size=64)
        assert cache.max_size == 64
    
    def test_cache_get_or_create_new(self):
        """Test creating new graph and caching it."""
        cache = PresetCache(max_size=10)
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        call_count = 0
        def factory():
            nonlocal call_count
            call_count += 1
            graph = InferenceGraph(name="test_graph")
            return graph
        
        # First call should create graph
        graph1 = cache.get_or_create(
            "test_preset",
            encoder, model, scheduler, decoder,
            factory
        )
        assert call_count == 1
        assert cache.size() == 1
        assert isinstance(graph1, InferenceGraph)
    
    def test_cache_hit(self):
        """Test cache hit with identical models."""
        cache = PresetCache(max_size=10)
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        call_count = 0
        def factory():
            nonlocal call_count
            call_count += 1
            return InferenceGraph(name="test_graph")
        
        # First call
        graph1 = cache.get_or_create(
            "test_preset",
            encoder, model, scheduler, decoder,
            factory
        )
        
        # Second call with same models (same identity)
        graph2 = cache.get_or_create(
            "test_preset",
            encoder, model, scheduler, decoder,
            factory
        )
        
        # Factory should only be called once
        assert call_count == 1
        assert graph1 is graph2  # Same object
        assert cache.size() == 1
    
    def test_cache_miss_different_models(self):
        """Test cache miss with different model instances."""
        cache = PresetCache(max_size=10)
        encoder1 = MockEncoder("encoder1")
        encoder2 = MockEncoder("encoder2")
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        call_count = 0
        def factory():
            nonlocal call_count
            call_count += 1
            return InferenceGraph(name="test_graph")
        
        # First call with encoder1
        graph1 = cache.get_or_create(
            "test_preset",
            encoder1, model, scheduler, decoder,
            factory
        )
        
        # Second call with different encoder (encoder2)
        graph2 = cache.get_or_create(
            "test_preset",
            encoder2, model, scheduler, decoder,
            factory
        )
        
        # Factory should be called twice (different encoders)
        assert call_count == 2
        assert graph1 is not graph2  # Different objects
        assert cache.size() == 2
    
    def test_cache_miss_different_config(self):
        """Test cache miss with different configurations."""
        cache = PresetCache(max_size=10)
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        call_count = 0
        def factory():
            nonlocal call_count
            call_count += 1
            return InferenceGraph(name="test_graph")
        
        # First call with config1
        graph1 = cache.get_or_create(
            "test_preset",
            encoder, model, scheduler, decoder,
            factory,
            num_steps=30, guidance_scale=7.5
        )
        
        # Second call with config2 (different guidance_scale)
        graph2 = cache.get_or_create(
            "test_preset",
            encoder, model, scheduler, decoder,
            factory,
            num_steps=30, guidance_scale=9.0
        )
        
        # Factory should be called twice (different configs)
        assert call_count == 2
        assert graph1 is not graph2
        assert cache.size() == 2
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache exceeds max_size."""
        cache = PresetCache(max_size=3)
        
        call_count = 0
        def factory():
            nonlocal call_count
            call_count += 1
            return InferenceGraph(name="test_graph")
        
        # Create 3 graphs (fill cache)
        graphs = []
        for i in range(3):
            encoder = MockEncoder(f"encoder{i}")
            model = MockModel()
            scheduler = MockScheduler()
            decoder = MockDecoder()
            
            graph = cache.get_or_create(
                "test_preset",
                encoder, model, scheduler, decoder,
                factory
            )
            graphs.append((encoder, model, scheduler, decoder, graph))
            assert cache.size() == i + 1
        
        assert call_count == 3
        
        # Add 4th graph (should evict oldest)
        encoder4 = MockEncoder("encoder4")
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        graph4 = cache.get_or_create(
            "test_preset",
            encoder4, model, scheduler, decoder,
            factory
        )
        
        assert call_count == 4
        assert cache.size() == 3  # Still max size
        
        # Accessing first graph again should recreate it (evicted)
        encoder0, model0, scheduler0, decoder0, _ = graphs[0]
        graph0_again = cache.get_or_create(
            "test_preset",
            encoder0, model0, scheduler0, decoder0,
            factory
        )
        
        # Should be a new graph (recreated)
        assert call_count == 5
    
    def test_cache_clear(self):
        """Test clearing cache."""
        cache = PresetCache(max_size=10)
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        def factory():
            return InferenceGraph(name="test_graph")
        
        # Create a graph
        cache.get_or_create(
            "test_preset",
            encoder, model, scheduler, decoder,
            factory
        )
        assert cache.size() == 1
        
        # Clear cache
        cache.clear()
        assert cache.size() == 0
    
    def test_cache_access_order_update(self):
        """Test that accessing cached graph updates access order."""
        cache = PresetCache(max_size=2)
        
        call_count = 0
        def factory():
            nonlocal call_count
            call_count += 1
            return InferenceGraph(name="test_graph")
        
        # Create two graphs
        encoder1 = MockEncoder("encoder1")
        encoder2 = MockEncoder("encoder2")
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        graph1 = cache.get_or_create(
            "test_preset",
            encoder1, model, scheduler, decoder,
            factory
        )
        
        graph2 = cache.get_or_create(
            "test_preset",
            encoder2, model, scheduler, decoder,
            factory
        )
        
        assert call_count == 2
        assert cache.size() == 2
        
        # Access graph1 again (should update access order)
        graph1_again = cache.get_or_create(
            "test_preset",
            encoder1, model, scheduler, decoder,
            factory
        )
        assert call_count == 2  # No new creation
        assert graph1_again is graph1
        
        # Create graph3 (should evict graph2, not graph1)
        encoder3 = MockEncoder("encoder3")
        graph3 = cache.get_or_create(
            "test_preset",
            encoder3, model, scheduler, decoder,
            factory
        )
        
        assert call_count == 3
        assert cache.size() == 2
        
        # Accessing graph2 should recreate it (evicted)
        graph2_again = cache.get_or_create(
            "test_preset",
            encoder2, model, scheduler, decoder,
            factory
        )
        assert call_count == 4  # Recreated


class TestCachedPresetWrappers:
    """Tests for cached preset wrapper functions."""
    
    @pytest.fixture(autouse=True)
    def cleanup_cache(self):
        """Clear cache before and after each test."""
        preset_cache_clear()
        yield
        preset_cache_clear()
    
    def test_cached_t2v_one_stage(self):
        """Test cached_t2v_one_stage wrapper."""
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        # First call
        graph1 = preset_cached_t2v_one_stage(
            encoder, model, scheduler, decoder,
            num_inference_steps=30
        )
        assert isinstance(graph1, InferenceGraph)
        assert preset_cache_size() == 1
        
        # Second call (cached)
        graph2 = preset_cached_t2v_one_stage(
            encoder, model, scheduler, decoder,
            num_inference_steps=30
        )
        assert graph2 is graph1
        assert preset_cache_size() == 1
    
    def test_cached_t2v_two_stages(self):
        """Test cached_t2v_two_stages wrapper."""
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        upsampler = MockUpsampler()
        
        # First call
        graph1 = preset_cached_t2v_two_stages(
            encoder, model, scheduler, upsampler, decoder,
            stage1_steps=15, stage2_steps=10
        )
        assert isinstance(graph1, InferenceGraph)
        assert preset_cache_size() == 1
        
        # Second call (cached)
        graph2 = preset_cached_t2v_two_stages(
            encoder, model, scheduler, upsampler, decoder,
            stage1_steps=15, stage2_steps=10
        )
        assert graph2 is graph1
        assert preset_cache_size() == 1
    
    def test_cached_distilled_fast(self):
        """Test cached_distilled_fast wrapper."""
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        graph = preset_cached_distilled_fast(
            encoder, model, scheduler, decoder,
            num_inference_steps=4
        )
        assert isinstance(graph, InferenceGraph)
        assert preset_cache_size() == 1
    
    def test_cached_ic_lora(self):
        """Test cached_ic_lora wrapper."""
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        graph = preset_cached_ic_lora(
            encoder, model, scheduler, decoder
        )
        assert isinstance(graph, InferenceGraph)
        assert preset_cache_size() == 1
    
    def test_cached_keyframe_interpolation(self):
        """Test cached_keyframe_interpolation wrapper."""
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        graph = preset_cached_keyframe_interpolation(
            encoder, model, scheduler, decoder
        )
        assert isinstance(graph, InferenceGraph)
        assert preset_cache_size() == 1
    
    def test_cached_t2v_one_stage_adaptive(self):
        """Test cached_t2v_one_stage_adaptive wrapper."""
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        graph = preset_cached_t2v_one_stage_adaptive(
            encoder, model, scheduler, decoder,
            enable_early_exit=True
        )
        assert isinstance(graph, InferenceGraph)
        assert preset_cache_size() == 1
    
    def test_cached_presets_separate_caches(self):
        """Test that different preset types have separate cache entries."""
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        # Create different preset types
        graph1 = preset_cached_t2v_one_stage(encoder, model, scheduler, decoder)
        graph2 = preset_cached_distilled_fast(encoder, model, scheduler, decoder)
        graph3 = preset_cached_ic_lora(encoder, model, scheduler, decoder)
        
        # Should have 3 separate cache entries
        assert preset_cache_size() == 3
        assert graph1 is not graph2
        assert graph2 is not graph3
        assert graph1 is not graph3
    
    def test_cache_config(self):
        """Test preset_cache_config function."""
        # Configure cache to size 16
        preset_cache_config(16)
        
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        # Create 16 presets (should fit)
        for i in range(16):
            encoder_i = MockEncoder(f"encoder{i}")
            preset_cached_t2v_one_stage(
                encoder_i, model, scheduler, decoder
            )
        
        assert preset_cache_size() == 16
        
        # 17th preset should evict oldest
        encoder_17 = MockEncoder("encoder17")
        preset_cached_t2v_one_stage(
            encoder_17, model, scheduler, decoder
        )
        assert preset_cache_size() == 16
    
    def test_cache_clear_function(self):
        """Test preset_cache_clear function."""
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        # Create some presets
        preset_cached_t2v_one_stage(encoder, model, scheduler, decoder)
        preset_cached_distilled_fast(encoder, model, scheduler, decoder)
        
        assert preset_cache_size() == 2
        
        # Clear cache
        preset_cache_clear()
        assert preset_cache_size() == 0
    
    def test_cache_size_function(self):
        """Test preset_cache_size function."""
        assert preset_cache_size() == 0
        
        encoder = MockEncoder()
        model = MockModel()
        scheduler = MockScheduler()
        decoder = MockDecoder()
        
        # Add presets
        preset_cached_t2v_one_stage(encoder, model, scheduler, decoder)
        assert preset_cache_size() == 1
        
        preset_cached_t2v_one_stage(
            encoder, model, scheduler, decoder,
            num_inference_steps=20  # Different config
        )
        assert preset_cache_size() == 2
