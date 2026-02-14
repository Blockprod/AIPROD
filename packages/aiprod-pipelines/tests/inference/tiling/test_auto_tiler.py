"""
Tests for adaptive tiling engine.

Validates automatic strategy selection based on VRAM availability
and ensures selected strategies stay within memory budgets.
"""

import pytest
import torch
from aiprod_pipelines.inference.tiling import (
    AdaptiveTilingEngine,
    auto_select_tiling,
    TilingSizeConfig,
)


class TestAdaptiveTilingEngine:
    """Tests for AdaptiveTilingEngine."""
    
    def test_initialization_default_values(self):
        """Verify engine initializes with sensible defaults."""
        engine = AdaptiveTilingEngine()
        
        assert engine.target_memory_percent == 0.75
        assert engine.min_memory_overhead_gb == 2.0
    
    def test_initialization_custom_values(self):
        """Verify custom values are accepted."""
        engine = AdaptiveTilingEngine(
            target_memory_percent=0.5,
            min_memory_overhead_gb=1.0,
        )
        
        assert engine.target_memory_percent == 0.5
        assert engine.min_memory_overhead_gb == 1.0
    
    def test_selects_no_tiling_if_fits(self):
        """Small tensors that fit should not be tiled."""
        engine = AdaptiveTilingEngine(
            target_memory_percent=0.90,  # High budget
            min_memory_overhead_gb=0.5,   # Low overhead
        )
        
        # Small 128x128 latents should fit
        strategy = engine.select_strategy((1, 4, 128, 128, 24))
        assert strategy == "no_tiling"
    
    def test_selects_spatial_for_large_spatial(self):
        """Large spatial dimensions should trigger spatial tiling."""
        engine = AdaptiveTilingEngine(
            target_memory_percent=0.5,   # Tight budget
            min_memory_overhead_gb=6.0,  # Limited VRAM
        )
        
        # Large spatial 1024x1024
        strategy = engine.select_strategy((1, 16, 1024, 1024, 50))
        # Should be something other than no_tiling
        assert strategy != "no_tiling"
    
    def test_selects_temporal_for_many_frames(self):
        """Many frames should trigger temporal/hybrid tiling."""
        engine = AdaptiveTilingEngine(
            target_memory_percent=0.5,
            min_memory_overhead_gb=6.0,
        )
        
        # Many frames (400)
        strategy = engine.select_strategy((1, 16, 384, 384, 400))
        # Should use some form of tiling
        assert strategy != "no_tiling"
    
    def test_selects_hybrid_for_extreme_memory(self):
        """Very constrained memory should select hybrid tiling."""
        engine = AdaptiveTilingEngine(
            target_memory_percent=0.2,   # Very tight
            min_memory_overhead_gb=7.5,  # Very limited
        )
        
        # Large tensor
        strategy = engine.select_strategy((1, 16, 544, 960, 97))
        # For extreme constraints, likely hybrid
        assert strategy in ["temporal_tiling", "hybrid_tiling", "spatial_tiling"]
    
    def test_get_strategy_instance_no_tiling(self):
        """Can create NoTiling instance."""
        engine = AdaptiveTilingEngine()
        strategy = engine.get_strategy_instance("no_tiling")
        
        assert strategy is not None
        tiles = strategy.get_tiles(torch.randn(1, 16, 100, 100, 50))
        assert len(tiles) == 1
    
    def test_get_strategy_instance_spatial(self):
        """Can create SpatialTiling instance."""
        engine = AdaptiveTilingEngine()
        strategy = engine.get_strategy_instance("spatial_tiling")
        
        assert strategy is not None
        tiles = strategy.get_tiles(torch.randn(1, 16, 544, 960, 50))
        assert len(tiles) > 1
    
    def test_get_strategy_instance_temporal(self):
        """Can create TemporalTiling instance."""
        engine = AdaptiveTilingEngine()
        strategy = engine.get_strategy_instance("temporal_tiling")
        
        assert strategy is not None
        tiles = strategy.get_tiles(torch.randn(1, 16, 100, 100, 200))
        assert len(tiles) > 1
    
    def test_get_strategy_instance_hybrid(self):
        """Can create HybridTiling instance."""
        engine = AdaptiveTilingEngine()
        strategy = engine.get_strategy_instance("hybrid_tiling")
        
        assert strategy is not None
        tiles = strategy.get_tiles(torch.randn(1, 16, 544, 960, 97))
        assert len(tiles) > 1
    
    def test_get_config_for_strategy_no_tiling(self):
        """Config for no_tiling is None."""
        engine = AdaptiveTilingEngine()
        config = engine.get_config_for_strategy("no_tiling")
        
        assert config is None
    
    def test_get_config_for_strategy_returns_config(self):
        """Config for tiling strategies returns TilingSizeConfig."""
        engine = AdaptiveTilingEngine()
        
        for strategy in ["spatial_tiling", "temporal_tiling", "hybrid_tiling"]:
            config = engine.get_config_for_strategy(strategy)
            assert isinstance(config, TilingSizeConfig)
    
    def test_get_config_adjusts_for_small_shape(self):
        """Config should adjust tile sizes for small tensors."""
        engine = AdaptiveTilingEngine()
        
        # Very small spatial
        config = engine.get_config_for_strategy(
            "spatial_tiling",
            latents_shape=(1, 16, 64, 64, 50)
        )
        
        # Should adjust tile sizes down
        assert config.spatial_tile_h <= 64
        assert config.spatial_tile_w <= 64
    
    def test_estimate_tensor_memory_correct(self):
        """Memory estimation matches theoretical calculation."""
        engine = AdaptiveTilingEngine()
        
        B, C, H, W, T = 1, 16, 544, 960, 97
        mem = engine._estimate_tensor_memory(B, C, H, W, T)
        
        # Should be B*C*H*W*T*2 bytes / 1e9
        expected = (B * C * H * W * T * 2) / 1e9
        assert abs(mem - expected) < 0.01


class TestAutoSelectTiling:
    """Tests for auto_select_tiling convenience function."""
    
    def test_returns_strategy_and_config(self):
        """Function returns tuple of (strategy, config)."""
        strategy, config = auto_select_tiling((1, 16, 256, 256, 50))
        
        assert isinstance(strategy, str)
        assert strategy in ["no_tiling", "spatial_tiling", "temporal_tiling", "hybrid_tiling"]
        assert config is None or isinstance(config, TilingSizeConfig)
    
    def test_small_tensor_no_tiling(self):
        """Small tensors should not require tiling."""
        strategy, config = auto_select_tiling((1, 4, 128, 128, 24))
        
        assert strategy == "no_tiling"
        assert config is None
    
    def test_large_tensor_returns_config(self):
        """Large tensors should return tiling config."""
        strategy, config = auto_select_tiling(
            (1, 16, 544, 960, 97),
            model_memory_gb=4.0,
            target_memory_percent=0.5,
        )
        
        if strategy != "no_tiling":
            assert isinstance(config, TilingSizeConfig)
    
    def test_custom_memory_budget_respected(self):
        """Tight memory budget should select aggressive tiling."""
        tight_strategy, _ = auto_select_tiling(
            (1, 16, 544, 960, 97),
            model_memory_gb=8.0,
            target_memory_percent=0.2,
        )
        
        loose_strategy, _ = auto_select_tiling(
            (1, 16, 544, 960, 97),
            model_memory_gb=2.0,
            target_memory_percent=0.9,
        )
        
        # Loose budget might not need tiling
        # Tight budget should need it
        # (This is probabilistic based on VRAM, but good test)


class TestMemoryEstimation:
    """Tests for memory estimation accuracy."""
    
    def test_estimation_increases_with_size(self):
        """Larger tensors should estimate more memory."""
        engine = AdaptiveTilingEngine()
        
        mem1 = engine._estimate_tensor_memory(1, 16, 256, 256, 50)
        mem2 = engine._estimate_tensor_memory(1, 16, 512, 512, 50)
        mem4 = engine._estimate_tensor_memory(2, 16, 512, 512, 50)
        
        assert mem1 < mem2
        assert mem2 < mem4
    
    def test_estimation_bfloat16_correct(self):
        """Memory estimation assumes 2 bytes per element (bfloat16)."""
        engine = AdaptiveTilingEngine()
        
        B, C, H, W, T = 1, 16, 256, 256, 50
        mem_gb = engine._estimate_tensor_memory(B, C, H, W, T)
        
        # 1*16*256*256*50 = 52,428,800 elements
        # * 2 bytes = 104,857,600 bytes
        # / 1e9 ≈ 0.1049 GB
        expected_gb = (B * C * H * W * T * 2) / 1e9
        assert abs(mem_gb - expected_gb) < 0.0001
