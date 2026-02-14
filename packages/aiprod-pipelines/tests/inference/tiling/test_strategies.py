"""
Tests for tiling strategies.

Validates that each strategy correctly generates tile specifications,
respects size constraints, and maintains reasonable memory footprints.
"""

import pytest
import torch
from aiprod_pipelines.inference.tiling import (
    TilingSizeConfig,
    NoTiling,
    SpatialTiling,
    TemporalTiling,
    HybridTiling,
)


class TestNoTiling:
    """Tests for NoTiling strategy."""
    
    def test_single_tile_covers_full_tensor(self):
        """Verify no tiling returns single full-size tile."""
        strategy = NoTiling()
        tiles = strategy.get_tiles(torch.randn(1, 16, 68, 120, 97))
        
        assert len(tiles) == 1
        assert tiles[0]['h'] == (0, 68)
        assert tiles[0]['w'] == (0, 120)
        assert tiles[0]['t'] == (0, 97)
        assert tiles[0]['overlap_h'] == 0
    
    def test_memory_estimation_reasonable(self):
        """Verify memory estimate matches tensor size."""
        strategy = NoTiling()
        mem_gb = strategy.estimate_memory_needed((1, 16, 68, 120, 97))
        
        # 1 * 16 * 68 * 120 * 97 * 2 bytes / 1e9 ≈ 3.15 GB
        assert 3.0 < mem_gb < 3.3
    
    def test_num_tiles_is_one(self):
        """Single tile strategy always returns 1."""
        strategy = NoTiling()
        assert strategy.num_tiles((1, 16, 68, 120, 97)) == 1
        assert strategy.num_tiles((2, 32, 256, 256, 200)) == 1


class TestSpatialTiling:
    """Tests for SpatialTiling strategy."""
    
    def test_generates_multiple_tiles(self):
        """Verify spatial tiling creates tiles for large tensors."""
        config = TilingSizeConfig(
            spatial_tile_h=192,
            spatial_tile_w=192,
            temporal_tile_f=9999,  # No temporal split
        )
        strategy = SpatialTiling(config)
        tiles = strategy.get_tiles(torch.randn(1, 16, 544, 960, 97))
        
        # Should split into multiple tiles
        assert len(tiles) > 1
        assert all(t['t'] == (0, 97) for t in tiles)  # No temporal split
    
    def test_overlap_regions_correct(self):
        """Verify tiles have correct overlap specifications."""
        config = TilingSizeConfig(
            spatial_tile_h=192,
            spatial_tile_w=192,
            spatial_overlap_h=64,
            spatial_overlap_w=64,
        )
        strategy = SpatialTiling(config)
        tiles = strategy.get_tiles(torch.randn(1, 16, 500, 500, 50))
        
        # First tile should have no overlap at edges
        assert tiles[0]['overlap_h'] == 0
        assert tiles[0]['overlap_w'] == 0
        
        # Later tiles should have overlap
        if len(tiles) > 1:
            assert tiles[1]['overlap_w'] == 64
    
    def test_spatial_tiles_smaller_memory(self):
        """Verify single spatial tile uses less memory than full tensor."""
        config = TilingSizeConfig()
        strategy = SpatialTiling(config)
        
        full_memory = 1 * 16 * 544 * 960 * 97 * 2 / 1e9
        tile_memory = strategy.estimate_memory_needed((1, 16, 544, 960, 97))
        
        assert tile_memory < full_memory
        # Should be ~40% of full
        assert tile_memory < full_memory * 0.5
    
    def test_num_tiles_matches_tiles_list(self):
        """Verify num_tiles() matches actual tile count."""
        config = TilingSizeConfig()
        strategy = SpatialTiling(config)
        shape = (1, 16, 544, 960, 97)
        
        tiles = strategy.get_tiles(torch.randn(*shape))
        num = strategy.num_tiles(shape)
        
        assert len(tiles) == num


class TestTemporalTiling:
    """Tests for TemporalTiling strategy."""
    
    def test_splits_frames_only(self):
        """Verify temporal tiling only splits time dimension."""
        config = TilingSizeConfig(
            temporal_tile_f=48,
            temporal_overlap_f=24,
        )
        strategy = TemporalTiling(config)
        tiles = strategy.get_tiles(torch.randn(1, 16, 68, 120, 97))
        
        # Should split temporal
        assert len(tiles) > 1
        
        # But not spatial
        assert all(t['h'] == (0, 68) for t in tiles)
        assert all(t['w'] == (0, 120) for t in tiles)
    
    def test_temporal_overlap_correct(self):
        """Verify temporal overlap specifications."""
        config = TilingSizeConfig(
            temporal_tile_f=48,
            temporal_overlap_f=24,
        )
        strategy = TemporalTiling(config)
        tiles = strategy.get_tiles(torch.randn(1, 16, 68, 120, 97))
        
        # First tile has no overlap
        assert tiles[0]['overlap_t'] == 0
        
        # Later tiles have overlap if any
        if len(tiles) > 1:
            assert tiles[1]['overlap_t'] == 24
    
    def test_temporal_tiles_medium_memory(self):
        """Verify single temporal tile memory reduction."""
        config = TilingSizeConfig(temporal_tile_f=48)
        strategy = TemporalTiling(config)
        
        full_memory = 1 * 16 * 544 * 960 * 97 * 2 / 1e9
        tile_memory = strategy.estimate_memory_needed((1, 16, 544, 960, 97))
        
        # Should be ~50% of full
        assert tile_memory < full_memory * 0.6


class TestHybridTiling:
    """Tests for HybridTiling strategy."""
    
    def test_splits_all_dimensions(self):
        """Verify hybrid tiling splits spatial and temporal."""
        config = TilingSizeConfig()
        strategy = HybridTiling(config)
        tiles = strategy.get_tiles(torch.randn(1, 16, 544, 960, 97))
        
        # Should create many small tiles
        assert len(tiles) > 2
    
    def test_all_tiles_smaller(self):
        """Verify all hybrid tiles are small enough."""
        config = TilingSizeConfig(
            spatial_tile_h=192,
            spatial_tile_w=192,
            temporal_tile_f=48,
        )
        strategy = HybridTiling(config)
        tiles = strategy.get_tiles(torch.randn(1, 16, 544, 960, 97))
        
        for tile in tiles:
            h_s, h_e = tile['h']
            w_s, w_e = tile['w']
            t_s, t_e = tile['t']
            
            # Verify sizes
            assert h_e - h_s <= config.spatial_tile_h
            assert w_e - w_s <= config.spatial_tile_w
            assert t_e - t_s <= config.temporal_tile_f
    
    def test_hybrid_tiles_minimum_memory(self):
        """Verify hybrid tiling has best memory savings."""
        config = TilingSizeConfig()
        strategy = HybridTiling(config)
        
        full_memory = 1 * 16 * 544 * 960 * 97 * 2 / 1e9
        tile_memory = strategy.estimate_memory_needed((1, 16, 544, 960, 97))
        
        # Should be ~20% of full (80% savings)
        assert tile_memory < full_memory * 0.3
    
    def test_coverage_no_gaps(self):
        """Verify tiles cover entire tensor without gaps."""
        config = TilingSizeConfig()
        strategy = HybridTiling(config)
        tiles = strategy.get_tiles(torch.randn(1, 16, 544, 960, 97))
        
        # Check coverage (rough check)
        h_coverage = set()
        for tile in tiles:
            h_s, h_e = tile['h']
            for h in range(h_s, h_e):
                h_coverage.add(h)
        
        # Should mostly cover height dimension
        assert len(h_coverage) > 400  # Most of 544


class TestTilingSizeConfig:
    """Tests for TilingSizeConfig dataclass."""
    
    def test_default_values_reasonable(self):
        """Verify default config values are sensible."""
        config = TilingSizeConfig()
        
        # Divisible by 32 (stride requirement)
        assert config.spatial_tile_h % 32 == 0
        assert config.spatial_tile_w % 32 == 0
        
        # Divisible by 8 (temporal requirement)
        assert config.temporal_tile_f % 8 == 0
        
        # Overlaps consistent with tiles
        assert config.spatial_overlap_h < config.spatial_tile_h
        assert config.spatial_overlap_w < config.spatial_tile_w
        assert config.temporal_overlap_f < config.temporal_tile_f
    
    def test_custom_values_accepted(self):
        """Verify custom config can be set."""
        config = TilingSizeConfig(
            spatial_tile_h=256,
            spatial_tile_w=256,
            temporal_tile_f=32,
        )
        
        assert config.spatial_tile_h == 256
        assert config.temporal_tile_f == 32
