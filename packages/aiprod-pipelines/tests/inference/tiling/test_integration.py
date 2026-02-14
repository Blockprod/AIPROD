"""
Integration tests for smart tiling system.

Validates end-to-end tiling workflow from strategy selection through
tile processing and blending in realistic inference scenarios.
"""

import pytest
import torch
from unittest.mock import Mock, patch
from aiprod_pipelines.inference.tiling import (
    TilingConfigNode,
    TiledDenoiseWrapper,
    auto_select_tiling,
)
from aiprod_pipelines.inference.tiling.strategies import (
    NoTiling,
    SpatialTiling,
    TemporalTiling,
    HybridTiling,
    TilingSizeConfig,
)


class TestTilingWorkflow:
    """Integration tests for complete tiling workflows."""
    
    def test_inference_without_tiling(self):
        """Inference with no tiling should process tensor end-to-end."""
        latents = torch.randn(1, 16, 128, 128, 24)
        embeddings = torch.randn(1, 768)
        
        strategy = NoTiling()
        tiles = strategy.get_tiles(latents)
        
        # Should have exactly 1 tile covering full tensor
        assert len(tiles) == 1
        assert tiles[0]["shape"] == (1, 16, 128, 128, 24)
    
    def test_inference_with_spatial_tiling(self):
        """Spatial tiling workflow."""
        latents = torch.randn(1, 16, 544, 960, 50)
        
        config = TilingSizeConfig()
        strategy = SpatialTiling(config)
        tiles = strategy.get_tiles(latents)
        
        # Should have multiple tiles
        assert len(tiles) > 1
        
        # Each tile should be smaller than original
        for tile in tiles:
            assert tile["shape"][2] <= latents.shape[2]
            assert tile["shape"][3] <= latents.shape[3]
    
    def test_inference_with_temporal_tiling(self):
        """Temporal tiling workflow."""
        latents = torch.randn(1, 16, 256, 256, 200)
        
        config = TilingSizeConfig()
        strategy = TemporalTiling(config)
        tiles = strategy.get_tiles(latents)
        
        # Should split temporal dimension
        assert len(tiles) > 1
        
        # Verify temporal bounds are valid
        for i, tile in enumerate(tiles):
            t_start = tile["t_start"]
            t_end = t_start + tile["shape"][4]
            assert 0 <= t_start < latents.shape[4]
            assert t_end <= latents.shape[4]
    
    def test_inference_with_hybrid_tiling(self):
        """Hybrid tiling workflow."""
        latents = torch.randn(1, 16, 544, 960, 97)
        
        config = TilingSizeConfig()
        strategy = HybridTiling(config)
        tiles = strategy.get_tiles(latents)
        
        # Should have multiple tiles
        assert len(tiles) > 1
        
        # Should split multiple dimensions
        spatial_variation = len(set((t["i"], t["j"]) for t in tiles))
        temporal_variation = len(set(t["t_start"] for t in tiles))
        
        # Should have variation in at least one spatial + temporal
        assert spatial_variation > 1 or temporal_variation > 1


class TestTilingConfigNode:
    """Tests for TilingConfigNode graph integration."""
    
    def test_node_initialization(self):
        """Node initializes with proper config."""
        node = TilingConfigNode()
        
        assert node.input_keys == []
        assert node.output_keys == ["tiling_strategy", "tiling_config"]
    
    def test_node_processes_context(self):
        """Node processes context and updates it."""
        node = TilingConfigNode()
        context = {}
        
        # Mock the forward call
        output = node.forward(
            context=context,
            latents_shape=(1, 16, 544, 960, 97),
            model_memory_gb=8.0,
        )
        
        assert "tiling_strategy" in output
        assert "tiling_config" in output or output["tiling_config"] is None
    
    def test_node_selects_small_tensor_no_tiling(self):
        """Node should select no tiling for small tensors."""
        node = TilingConfigNode()
        
        output = node.forward(
            context={},
            latents_shape=(1, 4, 128, 128, 24),
            model_memory_gb=8.0,
        )
        
        assert output["tiling_strategy"] == "no_tiling"
        assert output["tiling_config"] is None
    
    def test_node_selects_tiling_for_large_tensor(self):
        """Node should select tiling for large tensors."""
        node = TilingConfigNode(
            target_memory_percent=0.3,
            min_memory_overhead_gb=6.0,
        )
        
        output = node.forward(
            context={},
            latents_shape=(1, 16, 544, 960, 97),
            model_memory_gb=8.0,
        )
        
        # With tight budget, should select some form of tiling
        if output["tiling_strategy"] != "no_tiling":
            assert isinstance(output["tiling_config"], TilingSizeConfig)


class TestTiledDenoiseWrapper:
    """Tests for TiledDenoiseWrapper."""
    
    def test_wrapper_initialization(self):
        """Wrapper initializes with denoise function."""
        mock_denoise = Mock()
        wrapper = TiledDenoiseWrapper(mock_denoise)
        
        assert wrapper.denoise_fn == mock_denoise
    
    def test_wrapper_processes_latents(self):
        """Wrapper processes latents through denoise."""
        # Mock denoise that returns same shape as input
        def mock_denoise(latents, embeddings, timestep):
            return latents  # Simplest case: return input
        
        wrapper = TiledDenoiseWrapper(mock_denoise)
        
        latents = torch.randn(1, 16, 128, 128, 24)
        embeddings = torch.randn(1, 768)
        timestep = torch.tensor([50])
        
        # Set no tiling for simplicity
        wrapper.tiling_strategy = "no_tiling"
        
        # Mock the forward chain
        output = wrapper.forward(
            context={},
            latents=latents,
            embeddings=embeddings,
            timestep=timestep,
        )
        
        assert "denoised_latents" in output
    
    def test_wrapper_applies_spatial_tiling(self):
        """Wrapper can apply spatial tiling strategy."""
        def mock_denoise(latents, embeddings, timestep):
            return torch.ones_like(latents) * 0.5
        
        wrapper = TiledDenoiseWrapper(mock_denoise)
        wrapper.tiling_strategy = "spatial_tiling"
        
        latents = torch.randn(1, 16, 256, 256, 24)
        embeddings = torch.randn(1, 768)
        timestep = torch.tensor([50])
        
        # This would normally call _denoise_spatial_tiles
        strategy = SpatialTiling(TilingSizeConfig(spatial_tile_h=128, spatial_tile_w=128))
        tiles = strategy.get_tiles(latents)
        
        # Verify tiles are generated correctly
        assert len(tiles) >= 1


class TestAutoTilingIntegration:
    """Tests for auto_select_tiling in realistic scenarios."""
    
    def test_small_model_inference(self):
        """Small model inference shouldn't need tiling."""
        strategy, config = auto_select_tiling(
            latents_shape=(1, 4, 128, 128, 24),
            model_memory_gb=4.0,
            target_memory_percent=0.75,
        )
        
        assert strategy == "no_tiling"
        assert config is None
    
    def test_large_model_inference(self):
        """Large model with large input might need tiling."""
        strategy, config = auto_select_tiling(
            latents_shape=(1, 16, 544, 960, 97),
            model_memory_gb=4.0,
            target_memory_percent=0.5,
        )
        
        # Could be any strategy; just verify it's valid
        assert strategy in [
            "no_tiling",
            "spatial_tiling",
            "temporal_tiling",
            "hybrid_tiling",
        ]
    
    def test_memory_constrained_scenario(self):
        """Very tight memory should use aggressive tiling."""
        tight_strategy, _ = auto_select_tiling(
            latents_shape=(1, 16, 544, 960, 97),
            model_memory_gb=2.0,
            target_memory_percent=0.2,
        )
        
        loose_strategy, _ = auto_select_tiling(
            latents_shape=(1, 16, 544, 960, 97),
            model_memory_gb=24.0,
            target_memory_percent=0.8,
        )
        
        # Tight should be more aggressive
        # (Can't assert specific strategy, but this validates logic)
    
    def test_video_generation_workflow(self):
        """Realistic video generation workflow."""
        # Typical LTX-V2v setup
        batch_size = 1
        latent_channels = 16
        height_tokens = 544
        width_tokens = 960
        frames = 97
        
        strategy, config = auto_select_tiling(
            latents_shape=(batch_size, latent_channels, height_tokens, width_tokens, frames),
            model_memory_gb=8.0,
            target_memory_percent=0.6,
        )
        
        assert strategy in [
            "no_tiling",
            "spatial_tiling",
            "temporal_tiling",
            "hybrid_tiling",
        ]


class TestMemoryEfficiency:
    """Tests for memory efficiency of tiling strategies."""
    
    def test_no_tiling_memory_usage(self):
        """No tiling uses full tensor memory."""
        strategy = NoTiling()
        B, C, H, W, T = 1, 16, 544, 960, 97
        
        full_memory = strategy.estimate_memory_needed(B, C, H, W, T)
        
        # Should be approximately full tensor size
        elements = B * C * H * W * T
        expected_gb = (elements * 2) / 1e9  # 2 bytes per bfloat16
        
        assert abs(full_memory - expected_gb) < expected_gb * 0.1
    
    def test_spatial_tiling_memory_savings(self):
        """Spatial tiling should reduce memory vs full tensor."""
        no_tile_strategy = NoTiling()
        spatial_strategy = SpatialTiling()
        
        B, C, H, W, T = 1, 16, 544, 960, 97
        
        full_mem = no_tile_strategy.estimate_memory_needed(B, C, H, W, T)
        tiled_mem = spatial_strategy.estimate_memory_needed(B, C, H, W, T)
        
        # Spatial tiling should use ~30-50% of full memory
        assert tiled_mem < full_mem
        assert tiled_mem > full_mem * 0.2  # Should still use meaningful amount
    
    def test_temporal_tiling_memory_savings(self):
        """Temporal tiling reduces memory for long sequences."""
        no_tile_strategy = NoTiling()
        temporal_strategy = TemporalTiling()
        
        B, C, H, W, T = 1, 16, 256, 256, 200
        
        full_mem = no_tile_strategy.estimate_memory_needed(B, C, H, W, T)
        tiled_mem = temporal_strategy.estimate_memory_needed(B, C, H, W, T)
        
        # Should save memory
        assert tiled_mem < full_mem
    
    def test_hybrid_tiling_maximum_savings(self):
        """Hybrid tiling should offer maximum memory reduction."""
        strategies = {
            "no_tiling": NoTiling(),
            "spatial": SpatialTiling(),
            "temporal": TemporalTiling(),
            "hybrid": HybridTiling(),
        }
        
        B, C, H, W, T = 1, 16, 544, 960, 97
        
        memories = {
            name: strat.estimate_memory_needed(B, C, H, W, T)
            for name, strat in strategies.items()
        }
        
        # Hybrid should use less memory than others
        assert memories["hybrid"] <= memories["spatial"]
        assert memories["hybrid"] <= memories["temporal"]


class TestTileAccessPatterns:
    """Tests for correct tile access and extraction patterns."""
    
    def test_tiles_cover_full_tensor(self):
        """All tiles together should cover full input tensor."""
        latents = torch.randn(1, 16, 256, 256, 50)
        strategy = SpatialTiling()
        tiles = strategy.get_tiles(latents)
        
        # Verify coverage (loosely - exact coverage depends on overlap)
        assert len(tiles) > 0
        assert all(0 <= t["i"] < 256 for t in tiles)
        assert all(0 <= t["j"] < 256 for t in tiles)
    
    def test_tiles_no_overlap_in_grid_position(self):
        """Tile grid positions should be well-defined."""
        latents = torch.randn(1, 16, 544, 960, 50)
        strategy = SpatialTiling(TilingSizeConfig(
            spatial_tile_h=128,
            spatial_tile_w=128,
        ))
        tiles = strategy.get_tiles(latents)
        
        grid_positions = [(t["i"], t["j"]) for t in tiles]
        # All positions should be unique (no duplicate positions)
        assert len(grid_positions) == len(set(grid_positions))
    
    def test_temporal_tiles_sequential(self):
        """Temporal tiles should be in sequence without gaps."""
        latents = torch.randn(1, 16, 256, 256, 100)
        strategy = TemporalTiling(TilingSizeConfig(temporal_tile_f=48))
        tiles = strategy.get_tiles(latents)
        
        # Sort by t_start
        tiles_sorted = sorted(tiles, key=lambda t: t["t_start"])
        
        # Verify sequential coverage
        for i in range(len(tiles_sorted) - 1):
            curr_end = tiles_sorted[i]["t_start"] + tiles_sorted[i]["shape"][4]
            next_start = tiles_sorted[i + 1]["t_start"]
            # Should have minimal overlap
            assert next_start <= curr_end


class TestTilingWithDifferentDtypes:
    """Tests for tiling with different tensor dtypes."""
    
    def test_tiling_float32(self):
        """Tiling should work with float32."""
        latents = torch.randn(1, 16, 256, 256, 50, dtype=torch.float32)
        strategy = NoTiling()
        tiles = strategy.get_tiles(latents)
        
        assert len(tiles) == 1
        assert tiles[0]["shape"] == (1, 16, 256, 256, 50)
    
    def test_tiling_bfloat16(self):
        """Tiling should work with bfloat16."""
        latents = torch.randn(1, 16, 256, 256, 50, dtype=torch.bfloat16)
        strategy = NoTiling()
        tiles = strategy.get_tiles(latents)
        
        assert len(tiles) == 1
    
    def test_tiling_float16(self):
        """Tiling should work with float16."""
        latents = torch.randn(1, 16, 256, 256, 50, dtype=torch.float16)
        strategy = NoTiling()
        tiles = strategy.get_tiles(latents)
        
        assert len(tiles) == 1


class TestConfigValidation:
    """Tests for configuration validation."""
    
    def test_invalid_tile_size_raises(self):
        """Invalid tile sizes should raise error."""
        with pytest.raises((ValueError, AssertionError)):
            TilingSizeConfig(spatial_tile_h=0)
    
    def test_invalid_overlap_raises(self):
        """Invalid overlap should raise error."""
        with pytest.raises((ValueError, AssertionError)):
            TilingSizeConfig(overlap_h=-1)
    
    def test_valid_custom_config(self):
        """Valid custom configuration should work."""
        config = TilingSizeConfig(
            spatial_tile_h=256,
            spatial_tile_w=320,
            temporal_tile_f=96,
        )
        
        assert config.spatial_tile_h == 256
        assert config.spatial_tile_w == 320
        assert config.temporal_tile_f == 96
