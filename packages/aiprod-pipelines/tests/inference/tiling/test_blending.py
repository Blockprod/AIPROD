"""
Tests for tile blending functionality.

Validates seamless blending of tile boundaries to eliminate visible seams
and overlapping tile regions.
"""

import pytest
import torch
from aiprod_pipelines.inference.tiling.blending import (
    create_blend_mask_1d,
    create_blend_mask_2d,
    blend_spatial_tiles,
    blend_temporal_tiles,
    blend_hybrid_tiles,
    TileBlendingManager,
)


class TestBlendMask1D:
    """Tests for 1D blend mask generation."""
    
    def test_gaussian_mask_shape(self):
        """Gaussian mask has correct shape."""
        mask = create_blend_mask_1d(128, mode="gaussian")
        
        assert mask.shape == (128,)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
    
    def test_gaussian_mask_symmetric(self):
        """Gaussian mask should be symmetric."""
        mask = create_blend_mask_1d(128, mode="gaussian")
        
        # Compare first half to flipped second half
        first_half = mask[:64]
        second_half = mask[64:].flip(0)
        
        assert torch.allclose(first_half, second_half, atol=0.01)
    
    def test_gaussian_mask_center_emphasis(self):
        """Gaussian mask should emphasize center."""
        mask = create_blend_mask_1d(128, mode="gaussian")
        
        center_val = mask[64]
        edge_val = mask[0]
        
        assert center_val > edge_val
    
    def test_linear_mask_shape(self):
        """Linear mask has correct shape."""
        mask = create_blend_mask_1d(128, mode="linear")
        
        assert mask.shape == (128,)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
    
    def test_linear_mask_increases(self):
        """Linear mask should monotonically increase."""
        mask = create_blend_mask_1d(128, mode="linear")
        
        # Should be mostly increasing (allowing small float errors)
        diffs = torch.diff(mask)
        assert (diffs >= -1e-6).all()
    
    def test_cosine_mask_shape(self):
        """Cosine mask has correct shape."""
        mask = create_blend_mask_1d(128, mode="cosine")
        
        assert mask.shape == (128,)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
    
    def test_cosine_mask_smooth(self):
        """Cosine mask should be smooth."""
        mask = create_blend_mask_1d(128, mode="cosine")
        
        # First derivative should be smooth
        diffs = torch.diff(mask)
        # No single step should be huge
        assert (torch.abs(diffs) < 0.1).all()
    
    def test_mask_fade_in_mode(self):
        """fade_in=True should make mask go from low to high."""
        mask_in = create_blend_mask_1d(128, mode="linear", fade_in=True)
        
        # Should start low, end high
        assert mask_in[0] < mask_in[64]
        assert mask_in[64] < mask_in[127]
    
    def test_mask_fade_out_mode(self):
        """fade_out=True should make mask go from high to low."""
        mask_out = create_blend_mask_1d(128, mode="linear", fade_out=True)
        
        # Should start high, end low
        assert mask_out[0] > mask_out[64]
        assert mask_out[64] > mask_out[127]
    
    def test_invalid_mode_raises(self):
        """Invalid blend mode should raise error."""
        with pytest.raises(ValueError):
            create_blend_mask_1d(128, mode="unknown_mode")


class TestBlendMask2D:
    """Tests for 2D blend mask generation."""
    
    def test_mask_2d_shape(self):
        """2D mask has correct shape."""
        mask = create_blend_mask_2d((128, 256), mode="gaussian")
        
        assert mask.shape == (1, 1, 128, 256)
    
    def test_mask_2d_values_in_range(self):
        """2D mask values in [0, 1]."""
        mask = create_blend_mask_2d((128, 256), mode="gaussian")
        
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0
    
    def test_mask_2d_center_emphasis(self):
        """2D mask should emphasize center."""
        mask = create_blend_mask_2d((128, 256), mode="gaussian")
        
        center_val = mask[0, 0, 64, 128]
        corner_val = mask[0, 0, 0, 0]
        
        assert center_val > corner_val
    
    def test_mask_2d_linear_increases(self):
        """Linear 2D mask increases from top-left."""
        mask = create_blend_mask_2d((64, 64), mode="linear")
        
        tl = mask[0, 0, 0, 0]
        br = mask[0, 0, 63, 63]
        
        assert tl < br
    
    def test_mask_2d_different_sizes(self):
        """Can create 2D masks of different sizes."""
        for h, w in [(64, 64), (128, 256), (256, 128)]:
            mask = create_blend_mask_2d((h, w), mode="gaussian")
            assert mask.shape == (1, 1, h, w)


class TestBlendSpatialTiles:
    """Tests for spatial tile blending."""
    
    def test_single_tile_passthrough(self):
        """Single tile should pass through unchanged."""
        tile = torch.randn(1, 16, 192, 192, 3)
        tiles = [{"data": tile, "i": 0, "j": 0}]
        
        result = blend_spatial_tiles(
            tiles=tiles,
            output_shape=(1, 16, 192, 192, 3)
        )
        
        assert torch.allclose(result, tile, atol=1e-5)
    
    def test_two_tiles_horizontal_blend(self):
        """Two horizontally adjacent tiles should blend."""
        tile1 = torch.ones(1, 16, 192, 100, 3)
        tile2 = torch.ones(1, 16, 192, 100, 3) * 2
        
        # Tiles overlap by 64 pixels
        tiles = [
            {"data": tile1, "i": 0, "j": 0, "overlap_w": 64},
            {"data": tile2, "i": 0, "j": 36, "overlap_w": 64},
        ]
        
        result = blend_spatial_tiles(
            tiles=tiles,
            output_shape=(1, 16, 192, 236, 3)
        )
        
        # Overlap region should be intermediate value (not 1 or 2)
        overlap_region = result[:, :, :, 100:136, :]
        assert overlap_region.min() > 1.0
        assert overlap_region.max() < 2.0
    
    def test_output_shape_respected(self):
        """Output should have specified shape."""
        tile = torch.randn(1, 16, 192, 192, 3)
        tiles = [{"data": tile, "i": 0, "j": 0}]
        
        for output_h in [192, 384, 256]:
            for output_w in [192, 384, 512]:
                result = blend_spatial_tiles(
                    tiles=tiles,
                    output_shape=(1, 16, output_h, output_w, 3)
                )
                
                # Result should fit in output shape
                assert result.shape[2] <= output_h
                assert result.shape[3] <= output_w
    
    def test_accumulation_weights_sum_to_one(self):
        """Properly blended regions should have weights sum ≈ 1."""
        tile1 = torch.ones(1, 16, 64, 96, 3)
        tile2 = torch.ones(1, 16, 64, 96, 3) * 0.5
        
        tiles = [
            {"data": tile1, "i": 0, "j": 0, "overlap_w": 32},
            {"data": tile2, "i": 0, "j": 64, "overlap_w": 32},
        ]
        
        result = blend_spatial_tiles(
            tiles=tiles,
            output_shape=(1, 16, 64, 160, 3)
        )
        
        # Non-overlap regions should be pure values
        assert torch.allclose(result[:, :, :, :32, :], tile1[:, :, :, :32, :], atol=0.1)
        assert torch.allclose(result[:, :, :, -32:, :], tile2[:, :, :, -32:, :], atol=0.1)


class TestBlendTemporalTiles:
    """Tests for temporal tile blending."""
    
    def test_single_frame_tile_passthrough(self):
        """Single frame sequence should pass through."""
        tile = torch.randn(1, 16, 192, 192, 48)
        tiles = [{"data": tile, "t_start": 0}]
        
        result = blend_temporal_tiles(
            tiles=tiles,
            output_shape=(1, 16, 192, 192, 48)
        )
        
        assert torch.allclose(result, tile, atol=1e-5)
    
    def test_two_clips_temporal_blend(self):
        """Two frames with overlap should blend temporally."""
        clip1 = torch.ones(1, 16, 64, 64, 50)
        clip2 = torch.ones(1, 16, 64, 64, 50) * 2
        
        tiles = [
            {"data": clip1, "t_start": 0, "temporal_overlap": 24},
            {"data": clip2, "t_start": 26, "temporal_overlap": 24},
        ]
        
        result = blend_temporal_tiles(
            tiles=tiles,
            output_shape=(1, 16, 64, 64, 76)
        )
        
        # Overlap frames should have intermediate values
        overlap_region = result[:, :, :, :, 26:50]
        assert overlap_region.min() > 1.0
        assert overlap_region.max() < 2.0
    
    def test_temporal_interpolation_smooth(self):
        """Frame-by-frame values should interpolate smoothly."""
        clip1 = torch.ones(1, 16, 64, 64, 50) * 1.0
        clip2 = torch.ones(1, 16, 64, 64, 50) * 3.0
        
        tiles = [
            {"data": clip1, "t_start": 0, "temporal_overlap": 25},
            {"data": clip2, "t_start": 25, "temporal_overlap": 25},
        ]
        
        result = blend_temporal_tiles(
            tiles=tiles,
            output_shape=(1, 16, 64, 64, 75)
        )
        
        # Check that middle frame is close to 2.0 (average)
        middle_frame = result[:, :, :, :, 50]
        assert middle_frame.mean() < 2.5
        assert middle_frame.mean() > 1.5


class TestBlendHybridTiles:
    """Tests for 3D hybrid tile blending."""
    
    def test_single_tile_passthrough(self):
        """Single tile should pass through unchanged."""
        tile = torch.randn(1, 16, 192, 192, 48)
        tiles = [{"data": tile, "i": 0, "j": 0, "t_start": 0, "overlap_w": 0, "temporal_overlap": 0}]
        
        result = blend_hybrid_tiles(
            tiles=tiles,
            output_shape=(1, 16, 192, 192, 48)
        )
        
        assert torch.allclose(result, tile, atol=1e-5)
    
    def test_2x2x2_cube_tiles(self):
        """2x2x2 cube of tiles should blend correctly."""
        # Create 8 tiles with different values for easy verification
        tiles = []
        val = 1.0
        for i in range(2):
            for j in range(2):
                for t in range(2):
                    tile = torch.ones(1, 16, 96, 96, 24) * val
                    tiles.append({
                        "data": tile,
                        "i": i,
                        "j": j,
                        "t_start": t * 24,
                        "overlap_w": 32,
                        "temporal_overlap": 8,
                    })
                    val += 0.5
        
        result = blend_hybrid_tiles(
            tiles=tiles,
            output_shape=(1, 16, 192, 192, 48)
        )
        
        # Check shape
        assert result.shape == (1, 16, 192, 192, 48)
        
        # Check boundaries
        assert result.min() >= 1.0  # Min of our tiles
        assert result.max() <= 5.0  # Max of our tiles


class TestTileBlendingManager:
    """Tests for TileBlendingManager orchestration."""
    
    def test_initialization_default_mode(self):
        """Manager initializes with default blending mode."""
        manager = TileBlendingManager()
        
        assert manager.blend_mode == "gaussian"
    
    def test_initialization_custom_mode(self):
        """Manager accepts custom blending mode."""
        manager = TileBlendingManager(blend_mode="linear")
        
        assert manager.blend_mode == "linear"
    
    def test_blend_2d_tiles_shape_preserved(self):
        """2D blending preserves shape."""
        manager = TileBlendingManager(blend_mode="gaussian")
        
        tile = torch.randn(1, 16, 192, 192, 3)
        tiles = [{"data": tile, "i": 0, "j": 0}]
        
        result = manager.blend_2d_tiles(tiles, (1, 16, 192, 192, 3))
        
        assert result.shape == (1, 16, 192, 192, 3)
    
    def test_blend_temporal_tiles_shape_preserved(self):
        """Temporal blending preserves shape."""
        manager = TileBlendingManager()
        
        tile = torch.randn(1, 16, 192, 192, 48)
        tiles = [{"data": tile, "t_start": 0}]
        
        result = manager.blend_temporal_tiles(tiles, (1, 16, 192, 192, 48))
        
        assert result.shape == (1, 16, 192, 192, 48)
    
    def test_blend_3d_tiles_shape_preserved(self):
        """3D blending preserves shape."""
        manager = TileBlendingManager()
        
        tile = torch.randn(1, 16, 96, 96, 24)
        tiles = [{"data": tile, "i": 0, "j": 0, "t_start": 0, "overlap_w": 0, "temporal_overlap": 0}]
        
        result = manager.blend_3d_tiles(tiles, (1, 16, 96, 96, 24))
        
        assert result.shape == (1, 16, 96, 96, 24)
    
    def test_can_change_blend_mode(self):
        """Can update blending mode after initialization."""
        manager = TileBlendingManager(blend_mode="gaussian")
        
        assert manager.blend_mode == "gaussian"
        
        manager.blend_mode = "linear"
        assert manager.blend_mode == "linear"


class TestBlendingEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_very_small_tiles(self):
        """Can blend very small tiles."""
        tile = torch.ones(1, 4, 32, 32, 16)
        tiles = [{"data": tile, "i": 0, "j": 0}]
        
        result = blend_spatial_tiles(tiles, (1, 4, 32, 32, 16))
        
        assert result.shape == (1, 4, 32, 32, 16)
    
    def test_very_large_tensors(self):
        """Can blend large tensors (memory permitting)."""
        # Smaller than full size but still substantial
        tile = torch.randn(1, 8, 256, 320, 24)
        tiles = [{"data": tile, "i": 0, "j": 0}]
        
        result = blend_spatial_tiles(tiles, (1, 8, 256, 320, 24))
        
        assert result.shape == (1, 8, 256, 320, 24)
    
    def test_blend_with_zero_overlaps(self):
        """Tiles with no overlap should not blend."""
        tile1 = torch.ones(1, 16, 64, 100, 3)
        tile2 = torch.ones(1, 16, 64, 100, 3) * 2
        
        tiles = [
            {"data": tile1, "i": 0, "j": 0, "overlap_w": 0},
            {"data": tile2, "i": 0, "j": 100, "overlap_w": 0},
        ]
        
        result = blend_spatial_tiles(tiles, (1, 16, 64, 200, 3))
        
        # First 100 should be tile1, last 100 should be tile2
        assert torch.allclose(result[:, :, :, :100, :], tile1, atol=0.1)
        assert torch.allclose(result[:, :, :, 100:, :], tile2, atol=0.1)
