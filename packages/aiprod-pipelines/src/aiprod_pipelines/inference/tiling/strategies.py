"""
Tiling strategies for memory-efficient inference.

Provides multiple strategies for splitting tensors into tiles:
- NoTiling: Process full tensor (baseline)
- SpatialTiling: Split spatial dimensions (H, W)
- TemporalTiling: Split temporal dimension (T / frames)
- HybridTiling: Combine spatial and temporal (maximum memory savings)

Each strategy handles overlapping regions for blending.
"""

from dataclasses import dataclass
from typing import List, Tuple
import torch

from einops import rearrange


@dataclass
class TilingSizeConfig:
    """Configuration for tile sizes and overlaps.
    
    All sizes must be compatible with the model architecture:
    - Spatial sizes should be divisible by 32 (for stride-32 layers)
    - Temporal sizes should be divisible by 8 (for temporal compression)
    """
    
    spatial_tile_h: int = 192
    """Spatial tile height (divisible by 32)."""
    
    spatial_tile_w: int = 192
    """Spatial tile width (divisible by 32)."""
    
    temporal_tile_f: int = 48
    """Temporal tile size in frames (divisible by 8)."""
    
    spatial_overlap_h: int = 64
    """Spatial overlap height for blending (divisible by 32)."""
    
    spatial_overlap_w: int = 64
    """Spatial overlap width for blending (divisible by 32)."""
    
    temporal_overlap_f: int = 24
    """Temporal overlap in frames for blending (divisible by 8)."""


class TilingStrategy:
    """Abstract base class for tiling strategies."""
    
    def get_tiles(self, latents: torch.Tensor) -> List[dict]:
        """Generate list of tile specifications.
        
        Args:
            latents: Tensor of shape [B, C, H, W, T]
        
        Returns:
            List of tile dicts with keys:
            - 'h': (start, end) for height
            - 'w': (start, end) for width
            - 't': (start, end) for temporal
            - 'overlap_h', 'overlap_w', 'overlap_t': overlap amounts
        """
        raise NotImplementedError
    
    def estimate_memory_needed(self, latents_shape: Tuple) -> float:
        """Estimate total GB needed for processing all tiles.
        
        Args:
            latents_shape: Shape tuple (B, C, H, W, T)
        
        Returns:
            Estimated memory in GB
        """
        raise NotImplementedError
    
    def num_tiles(self, latents_shape: Tuple) -> int:
        """Number of tiles this strategy will generate."""
        raise NotImplementedError


class NoTiling(TilingStrategy):
    """No tiling - process full tensor at once."""
    
    def get_tiles(self, latents: torch.Tensor) -> List[dict]:
        """Return single tile covering entire tensor."""
        B, C, H, W, T = latents.shape
        return [{
            'h': (0, H),
            'w': (0, W),
            't': (0, T),
            'overlap_h': 0,
            'overlap_w': 0,
            'overlap_t': 0,
        }]
    
    def estimate_memory_needed(self, latents_shape: Tuple) -> float:
        """Estimate memory for full tensor + model overhead."""
        B, C, H, W, T = latents_shape
        # Each element: 2 bytes (bfloat16)
        latent_memory = (B * C * H * W * T * 2) / 1e9
        return latent_memory
    
    def num_tiles(self, latents_shape: Tuple) -> int:
        return 1


class SpatialTiling(TilingStrategy):
    """Spatial tiling - split H, W dimensions only."""
    
    def __init__(self, config: TilingSizeConfig):
        """Initialize with tile size config."""
        self.config = config
    
    def get_tiles(self, latents: torch.Tensor) -> List[dict]:
        """Generate spatial tiles with overlap."""
        B, C, H, W, T = latents.shape
        tiles = []
        
        # Stride: tile_size - overlap
        h_stride = self.config.spatial_tile_h - self.config.spatial_overlap_h
        w_stride = self.config.spatial_tile_w - self.config.spatial_overlap_w
        
        # Generate tile positions
        h_starts = list(range(0, H, h_stride))
        w_starts = list(range(0, W, w_stride))
        
        # Ensure last tile covers remainder
        if h_starts[-1] + self.config.spatial_tile_h < H:
            h_starts.append(H - self.config.spatial_tile_h)
        if w_starts[-1] + self.config.spatial_tile_w < W:
            w_starts.append(W - self.config.spatial_tile_w)
        
        # Create tiles
        for h_s in h_starts:
            for w_s in w_starts:
                h_e = min(h_s + self.config.spatial_tile_h, H)
                w_e = min(w_s + self.config.spatial_tile_w, W)
                
                tiles.append({
                    'h': (h_s, h_e),
                    'w': (w_s, w_e),
                    't': (0, T),
                    'overlap_h': self.config.spatial_overlap_h if h_s > 0 else 0,
                    'overlap_w': self.config.spatial_overlap_w if w_s > 0 else 0,
                    'overlap_t': 0,
                })
        
        return tiles
    
    def estimate_memory_needed(self, latents_shape: Tuple) -> float:
        """Single spatial tile memory + overhead."""
        B, C, H, W, T = latents_shape
        # Memory for one tile (worst case: largest tile)
        tile_h = min(self.config.spatial_tile_h, H)
        tile_w = min(self.config.spatial_tile_w, W)
        tile_memory = (B * C * tile_h * tile_w * T * 2) / 1e9
        return tile_memory
    
    def num_tiles(self, latents_shape: Tuple) -> int:
        """Number of spatial tiles."""
        B, C, H, W, T = latents_shape
        h_stride = self.config.spatial_tile_h - self.config.spatial_overlap_h
        w_stride = self.config.spatial_tile_w - self.config.spatial_overlap_w
        
        num_h = (H + h_stride - 1) // h_stride
        num_w = (W + w_stride - 1) // w_stride
        
        return num_h * num_w


class TemporalTiling(TilingStrategy):
    """Temporal tiling - split T (frames) dimension only."""
    
    def __init__(self, config: TilingSizeConfig):
        """Initialize with tile size config."""
        self.config = config
    
    def get_tiles(self, latents: torch.Tensor) -> List[dict]:
        """Generate temporal tiles with overlap."""
        B, C, H, W, T = latents.shape
        tiles = []
        
        # Stride: tile_size - overlap
        t_stride = self.config.temporal_tile_f - self.config.temporal_overlap_f
        
        # Generate tile positions
        t_starts = list(range(0, T, t_stride))
        
        # Ensure last tile covers remainder
        if t_starts[-1] + self.config.temporal_tile_f < T:
            t_starts.append(T - self.config.temporal_tile_f)
        
        # Create tiles
        for t_s in t_starts:
            t_e = min(t_s + self.config.temporal_tile_f, T)
            
            tiles.append({
                'h': (0, H),
                'w': (0, W),
                't': (t_s, t_e),
                'overlap_h': 0,
                'overlap_w': 0,
                'overlap_t': self.config.temporal_overlap_f if t_s > 0 else 0,
            })
        
        return tiles
    
    def estimate_memory_needed(self, latents_shape: Tuple) -> float:
        """Single temporal tile memory."""
        B, C, H, W, T = latents_shape
        # Memory for one temporal tile
        tile_t = min(self.config.temporal_tile_f, T)
        tile_memory = (B * C * H * W * tile_t * 2) / 1e9
        return tile_memory
    
    def num_tiles(self, latents_shape: Tuple) -> int:
        """Number of temporal tiles."""
        B, C, H, W, T = latents_shape
        t_stride = self.config.temporal_tile_f - self.config.temporal_overlap_f
        
        return (T + t_stride - 1) // t_stride


class HybridTiling(TilingStrategy):
    """Hybrid tiling - combine spatial and temporal."""
    
    def __init__(self, config: TilingSizeConfig):
        """Initialize with tile size config."""
        self.config = config
    
    def get_tiles(self, latents: torch.Tensor) -> List[dict]:
        """Generate hybrid (spatial + temporal) tiles."""
        B, C, H, W, T = latents.shape
        tiles = []
        
        # Strides
        h_stride = self.config.spatial_tile_h - self.config.spatial_overlap_h
        w_stride = self.config.spatial_tile_w - self.config.spatial_overlap_w
        t_stride = self.config.temporal_tile_f - self.config.temporal_overlap_f
        
        # Generate positions
        h_starts = list(range(0, H, h_stride))
        w_starts = list(range(0, W, w_stride))
        t_starts = list(range(0, T, t_stride))
        
        # Ensure coverage
        if h_starts[-1] + self.config.spatial_tile_h < H:
            h_starts.append(H - self.config.spatial_tile_h)
        if w_starts[-1] + self.config.spatial_tile_w < W:
            w_starts.append(W - self.config.spatial_tile_w)
        if t_starts[-1] + self.config.temporal_tile_f < T:
            t_starts.append(T - self.config.temporal_tile_f)
        
        # Create tiles
        for h_s in h_starts:
            for w_s in w_starts:
                for t_s in t_starts:
                    h_e = min(h_s + self.config.spatial_tile_h, H)
                    w_e = min(w_s + self.config.spatial_tile_w, W)
                    t_e = min(t_s + self.config.temporal_tile_f, T)
                    
                    tiles.append({
                        'h': (h_s, h_e),
                        'w': (w_s, w_e),
                        't': (t_s, t_e),
                        'overlap_h': self.config.spatial_overlap_h if h_s > 0 else 0,
                        'overlap_w': self.config.spatial_overlap_w if w_s > 0 else 0,
                        'overlap_t': self.config.temporal_overlap_f if t_s > 0 else 0,
                    })
        
        return tiles
    
    def estimate_memory_needed(self, latents_shape: Tuple) -> float:
        """Single hybrid tile memory (smallest)."""
        B, C, H, W, T = latents_shape
        # Memory for one hybrid tile
        tile_h = min(self.config.spatial_tile_h, H)
        tile_w = min(self.config.spatial_tile_w, W)
        tile_t = min(self.config.temporal_tile_f, T)
        tile_memory = (B * C * tile_h * tile_w * tile_t * 2) / 1e9
        return tile_memory
    
    def num_tiles(self, latents_shape: Tuple) -> int:
        """Number of hybrid tiles."""
        B, C, H, W, T = latents_shape
        h_stride = self.config.spatial_tile_h - self.config.spatial_overlap_h
        w_stride = self.config.spatial_tile_w - self.config.spatial_overlap_w
        t_stride = self.config.temporal_tile_f - self.config.temporal_overlap_f
        
        num_h = (H + h_stride - 1) // h_stride
        num_w = (W + w_stride - 1) // w_stride
        num_t = (T + t_stride - 1) // t_stride
        
        return num_h * num_w * num_t
