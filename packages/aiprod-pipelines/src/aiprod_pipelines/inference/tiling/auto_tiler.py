"""
Adaptive tiling engine that automatically selects best tiling strategy.

Based on:
- Available VRAM
- Model memory footprint
- Input tensor shape
- Target memory budget

Ensures inference stays within memory limits while maximizing performance.
"""

from typing import Literal, Tuple
import torch

from .strategies import (
    TilingSizeConfig,
    TilingStrategy,
    NoTiling,
    SpatialTiling,
    TemporalTiling,
    HybridTiling,
)


def get_available_vram_gb() -> float:
    """Get available GPU VRAM in GB.
    
    Returns:
        Available VRAM in GB
    """
    try:
        # Try CUDA first
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            reserved_gb = torch.cuda.memory_reserved(0) / 1e9
            allocated_gb = torch.cuda.memory_allocated(0) / 1e9
            available_gb = total_gb - reserved_gb
            return max(available_gb, 1.0)  # At least 1GB
        else:
            # CPU - assume 16GB available (not limiting)
            return 16.0
    except Exception:
        return 8.0  # Conservative default


class AdaptiveTilingEngine:
    """Automatically selects tiling strategy based on VRAM availability.
    
    Algorithm:
    1. Estimate full tensor memory requirement
    2. Check if it fits within budget
    3. If not, progressively try more aggressive tiling
    4. Select best strategy that keeps memory within limits
    """
    
    def __init__(
        self,
        target_memory_percent: float = 0.75,
        min_memory_overhead_gb: float = 2.0,
    ):
        """Initialize tiling engine.
        
        Args:
            target_memory_percent: Use this % of available VRAM (0.75 = 75%)
            min_memory_overhead_gb: Keep this much VRAM free for other ops
        """
        self.target_memory_percent = target_memory_percent
        self.min_memory_overhead_gb = min_memory_overhead_gb
    
    def select_strategy(
        self,
        latents_shape: Tuple[int, int, int, int, int],
        model_memory_gb: float = 8.0,
    ) -> Literal["no_tiling", "spatial_tiling", "temporal_tiling", "hybrid_tiling"]:
        """Select best tiling strategy for given input.
        
        Args:
            latents_shape: Shape tuple (B, C, H, W, T)
            model_memory_gb: Estimated model + other overhead in GB
        
        Returns:
            Selected strategy name
        """
        B, C, H, W, T = latents_shape
        
        # Calculate available budget
        total_vram_gb = get_available_vram_gb()
        used_vram_gb = model_memory_gb + self.min_memory_overhead_gb
        budget_gb = (total_vram_gb - used_vram_gb) * self.target_memory_percent
        budget_gb = max(budget_gb, 0.5)  # At least 0.5GB
        
        # Estimate memory for no tiling
        no_tile_memory = self._estimate_tensor_memory(B, C, H, W, T)
        
        # If fits, use no tiling (fastest)
        if no_tile_memory <= budget_gb:
            return "no_tiling"
        
        # Calculate reduction factor needed
        reduction_needed = no_tile_memory / budget_gb
        
        # Try strategies in order of memory savings
        config = TilingSizeConfig()
        
        # Spatial tiling: ~40% reduction
        spatial_tiler = SpatialTiling(config)
        spatial_memory = spatial_tiler.estimate_memory_needed(latents_shape)
        if spatial_memory * reduction_needed < 1.8:  # 40% = 0.6x memory
            return "spatial_tiling"
        
        # Temporal tiling: ~50% reduction
        temporal_tiler = TemporalTiling(config)
        temporal_memory = temporal_tiler.estimate_memory_needed(latents_shape)
        if temporal_memory * reduction_needed < 1.5:  # 50% = 0.5x memory
            return "temporal_tiling"
        
        # Hybrid tiling: ~80% reduction (maximum)
        return "hybrid_tiling"
    
    def get_strategy_instance(
        self,
        strategy_name: Literal["no_tiling", "spatial_tiling", "temporal_tiling", "hybrid_tiling"],
        config: TilingSizeConfig | None = None,
    ) -> TilingStrategy:
        """Get instantiated strategy object.
        
        Args:
            strategy_name: Strategy to instantiate
            config: Optional custom TilingSizeConfig
        
        Returns:
            Instantiated strategy
        """
        if config is None:
            config = TilingSizeConfig()
        
        strategies = {
            "no_tiling": lambda: NoTiling(),
            "spatial_tiling": lambda: SpatialTiling(config),
            "temporal_tiling": lambda: TemporalTiling(config),
            "hybrid_tiling": lambda: HybridTiling(config),
        }
        
        return strategies[strategy_name]()
    
    def get_config_for_strategy(
        self,
        strategy_name: Literal["no_tiling", "spatial_tiling", "temporal_tiling", "hybrid_tiling"],
        latents_shape: Tuple[int, int, int, int, int] | None = None,
    ) -> TilingSizeConfig | None:
        """Get recommended TilingSizeConfig for strategy.
        
        Args:
            strategy_name: Strategy name
            latents_shape: Optional shape for fine-tuning config
        
        Returns:
            TilingSizeConfig or None if no tiling
        """
        if strategy_name == "no_tiling":
            return None
        
        # Base configs by strategy
        configs = {
            "spatial_tiling": TilingSizeConfig(
                spatial_tile_h=192,
                spatial_tile_w=192,
                temporal_tile_f=9999,  # No temporal splitting
                spatial_overlap_h=64,
                spatial_overlap_w=64,
                temporal_overlap_f=0,
            ),
            "temporal_tiling": TilingSizeConfig(
                spatial_tile_h=9999,  # No spatial splitting
                spatial_tile_w=9999,
                temporal_tile_f=48,
                spatial_overlap_h=0,
                spatial_overlap_w=0,
                temporal_overlap_f=24,
            ),
            "hybrid_tiling": TilingSizeConfig(
                spatial_tile_h=128,  # Smaller tiles for balance
                spatial_tile_w=128,
                temporal_tile_f=32,
                spatial_overlap_h=32,
                spatial_overlap_w=32,
                temporal_overlap_f=8,
            ),
        }
        
        config = configs.get(strategy_name)
        
        # Fine-tune based on shape if provided
        if config and latents_shape:
            B, C, H, W, T = latents_shape
            
            # Adjust temporal tile size if input is smaller
            if T < config.temporal_tile_f:
                config.temporal_tile_f = max(8, T // 2)
                config.temporal_overlap_f = max(0, config.temporal_tile_f // 3)
            
            # Adjust spatial sizes if input is smaller
            if H < config.spatial_tile_h or W < config.spatial_tile_w:
                config.spatial_tile_h = min(config.spatial_tile_h, H)
                config.spatial_tile_w = min(config.spatial_tile_w, W)
        
        return config
    
    @staticmethod
    def _estimate_tensor_memory(B: int, C: int, H: int, W: int, T: int) -> float:
        """Estimate tensor memory in GB.
        
        Args:
            B, C, H, W, T: Tensor dimensions
        
        Returns:
            Memory estimate in GB
        """
        # Each element: 2 bytes (bfloat16)
        return (B * C * H * W * T * 2) / 1e9


def auto_select_tiling(
    latents_shape: Tuple[int, int, int, int, int],
    model_memory_gb: float = 8.0,
    target_memory_percent: float = 0.75,
) -> Tuple[Literal["no_tiling", "spatial_tiling", "temporal_tiling", "hybrid_tiling"], TilingSizeConfig | None]:
    """Convenience function for auto-selecting tiling.
    
    Args:
        latents_shape: Tensor shape (B, C, H, W, T)
        model_memory_gb: Model memory footprint
        target_memory_percent: Target memory usage percentage
    
    Returns:
        Tuple of (strategy_name, config)
    """
    engine = AdaptiveTilingEngine(target_memory_percent=target_memory_percent)
    strategy = engine.select_strategy(latents_shape, model_memory_gb)
    config = engine.get_config_for_strategy(strategy, latents_shape)
    return strategy, config
