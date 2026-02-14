"""
GraphNode wrappers for tiling integration into inference graphs.

Provides two main nodes:
- TilingConfigNode: Automatically determine tiling strategy
- TiledDenoiseNode: Wrapper for DenoiseNode with tiling support
"""

from typing import Dict, Literal
import torch

from ..graph import GraphNode, GraphContext
from .strategies import TilingSizeConfig, NoTiling, SpatialTiling, TemporalTiling, HybridTiling
from .auto_tiler import AdaptiveTilingEngine
from .blending import TileBlendingManager


class TilingConfigNode(GraphNode):
    """Configuration node that determines optimal tiling strategy.
    
    Analyzes available VRAM and input tensor shape to select the best
    tiling strategy automatically.
    
    Outputs:
    - tiling_strategy: str (no_tiling, spatial_tiling, temporal_tiling, hybrid_tiling)
    - tiling_config: TilingSizeConfig or None
    """
    
    def __init__(
        self,
        adaptive: bool = True,
        target_memory_percent: float = 0.75,
        min_memory_overhead_gb: float = 2.0,
        model_memory_gb: float = 8.0,
    ):
        """Initialize tiling config node.
        
        Args:
            adaptive: Use adaptive strategy selection
            target_memory_percent: Target VRAM utilization
            min_memory_overhead_gb: Minimum free VRAM to keep
            model_memory_gb: Model memory footprint
        """
        self.adaptive = adaptive
        self.target_memory_percent = target_memory_percent
        self.min_memory_overhead_gb = min_memory_overhead_gb
        self.model_memory_gb = model_memory_gb
        self._engine = AdaptiveTilingEngine(
            target_memory_percent=target_memory_percent,
            min_memory_overhead_gb=min_memory_overhead_gb,
        )
    
    @property
    def input_keys(self) -> list[str]:
        return ["latents"]
    
    @property
    def output_keys(self) -> list[str]:
        return ["tiling_strategy", "tiling_config"]
    
    def execute(self, context: GraphContext, **kwargs) -> GraphContext:
        """Determine tiling strategy based on inputs."""
        latents = context.get("latents")
        
        if latents is None:
            raise ValueError("latents required in context for TilingConfigNode")
        
        # Select strategy
        strategy = self._engine.select_strategy(latents.shape, self.model_memory_gb)
        config = self._engine.get_config_for_strategy(strategy, latents.shape)
        
        context["tiling_strategy"] = strategy
        context["tiling_config"] = config
        
        return context


class TiledDenoiseWrapper(GraphNode):
    """Wrapper that adds tiling support to denoising.
    
    Automatically handles:
    - Tile extraction from latents
    - Per-tile denoising
    - Tile blending with spatial/temporal consistency
    
    Inputs:
    - latents: [B, C, H, W, T] tensor
    - embeddings: Text embeddings
    - tiling_strategy: Strategy string (from TilingConfigNode)
    - tiling_config: Strategy config
    
    Outputs:
    - latents_denoised: Denoised latents (same shape as input)
    """
    
    def __init__(
        self,
        denoise_fn,
        window_type: str = "gaussian",
    ):
        """Initialize tiled denoise wrapper.
        
        Args:
            denoise_fn: Callable that denoises a single tile
                       Signature: denoise_fn(latents, embeddings) -> denoised_latents
            window_type: Window type for blending (gaussian, linear, cosine)
        """
        self.denoise_fn = denoise_fn
        self.window_type = window_type
    
    @property
    def input_keys(self) -> list[str]:
        return ["latents", "embeddings"]
    
    @property
    def output_keys(self) -> list[str]:
        return ["latents_denoised"]
    
    def execute(self, context: GraphContext, **kwargs) -> GraphContext:
        """Execute tiled denoising."""
        latents = context["latents"]
        embeddings = context["embeddings"]
        
        strategy = context.get("tiling_strategy", "no_tiling")
        config = context.get("tiling_config")
        
        # Denoise based on strategy
        latents_denoised = self._denoise_with_strategy(
            latents, embeddings, strategy, config
        )
        
        context["latents_denoised"] = latents_denoised
        return context
    
    def _denoise_with_strategy(
        self,
        latents: torch.Tensor,
        embeddings: torch.Tensor,
        strategy: str,
        config: TilingSizeConfig | None,
    ) -> torch.Tensor:
        """Denoise using specified tiling strategy."""
        
        if strategy == "no_tiling" or config is None:
            # Standard denoising - no tiling
            return self.denoise_fn(latents, embeddings)
        
        # Get tiler
        if strategy == "spatial_tiling":
            tiler = SpatialTiling(config)
        elif strategy == "temporal_tiling":
            tiler = TemporalTiling(config)
        else:  # hybrid_tiling
            tiler = HybridTiling(config)
        
        # Get tiles
        tiles_spec = tiler.get_tiles(latents)
        
        # Process each tile
        if strategy == "spatial_tiling":
            return self._denoise_spatial_tiles(latents, embeddings, tiles_spec)
        elif strategy == "temporal_tiling":
            return self._denoise_temporal_tiles(latents, embeddings, tiles_spec)
        else:  # hybrid
            return self._denoise_hybrid_tiles(latents, embeddings, tiles_spec, config)
    
    def _denoise_spatial_tiles(
        self,
        latents: torch.Tensor,
        embeddings: torch.Tensor,
        tiles_spec: list[dict],
    ) -> torch.Tensor:
        """Denoise with spatial tiling."""
        B, C, H, W, T = latents.shape
        output = torch.zeros_like(latents)
        
        tiles_dict = {}
        for tile in tiles_spec:
            h_s, h_e = tile['h']
            w_s, w_e = tile['w']
            
            # Extract tile
            tile_latents = latents[:, :, h_s:h_e, w_s:w_e, :]
            
            # Denoise
            tile_denoised = self.denoise_fn(tile_latents, embeddings)
            
            # Store for blending
            tiles_dict[(h_s, h_e, w_s, w_e)] = tile_denoised
        
        # Blend tiles
        output = TileBlendingManager.blend_spatial_tiles(
            output, tiles_dict,
            overlap_h=tile['overlap_h'],
            overlap_w=tile['overlap_w'],
            window_type=self.window_type,
        )
        
        return output
    
    def _denoise_temporal_tiles(
        self,
        latents: torch.Tensor,
        embeddings: torch.Tensor,
        tiles_spec: list[dict],
    ) -> torch.Tensor:
        """Denoise with temporal tiling."""
        frames_list = []
        
        for tile in tiles_spec:
            t_s, t_e = tile['t']
            
            # Extract temporal tile
            tile_latents = latents[:, :, :, :, t_s:t_e]
            
            # Denoise
            tile_denoised = self.denoise_fn(tile_latents, embeddings)
            frames_list.append(tile_denoised)
        
        # Blend temporal tiles
        # Get overlap from first tile
        overlap_t = tiles_spec[0]['overlap_t'] if len(tiles_spec) > 0 else 0
        
        output = TileBlendingManager.blend_temporal_tiles(
            frames_list,
            overlap_frames=overlap_t,
            window_type=self.window_type,
        )
        
        return output
    
    def _denoise_hybrid_tiles(
        self,
        latents: torch.Tensor,
        embeddings: torch.Tensor,
        tiles_spec: list[dict],
        config: TilingSizeConfig,
    ) -> torch.Tensor:
        """Denoise with hybrid (spatial + temporal) tiling."""
        B, C, H, W, T = latents.shape
        
        tiles_dict = {}
        for tile in tiles_spec:
            h_s, h_e = tile['h']
            w_s, w_e = tile['w']
            t_s, t_e = tile['t']
            
            # Extract 3D tile
            tile_latents = latents[:, :, h_s:h_e, w_s:w_e, t_s:t_e]
            
            # Denoise
            tile_denoised = self.denoise_fn(tile_latents, embeddings)
            
            # Store
            tiles_dict[(h_s, h_e, w_s, w_e, t_s, t_e)] = tile_denoised
        
        # Blend hybrid tiles
        output = TileBlendingManager.blend_hybrid_tiles(
            tiles_dict,
            latents.shape,
            overlap_h=tiles_spec[0]['overlap_h'] if tiles_spec else 0,
            overlap_w=tiles_spec[0]['overlap_w'] if tiles_spec else 0,
            overlap_t=tiles_spec[0]['overlap_t'] if tiles_spec else 0,
            window_type=self.window_type,
        )
        
        return output
