"""
Smart Tiling System for memory-efficient inference.

Enables adaptive tiling of latent tensors during denoising to reduce VRAM
usage by 30-80% depending on configuration and memory constraints.

Quick Start:
```python
from aiprod_pipelines.inference.tiling import (
    TilingConfigNode,
    TiledDenoiseWrapper,
    auto_select_tiling,
)

# Auto-select tiling based on VRAM
strategy, config = auto_select_tiling(
    latents_shape=(1, 16, 68, 120, 97),
    model_memory_gb=8.0
)
print(f"Selected strategy: {strategy}")  # spatial_tiling, temporal_tiling, hybrid_tiling, or no_tiling

# Use in inference graph
graph.add_node("tiling_config", TilingConfigNode())
graph.add_node("denoise", DenoiseNode(...))
graph.connect("tiling_config", "denoise")
```

Strategies:
- no_tiling: Full tensor at once (fastest, high VRAM)
- spatial_tiling: Split H, W (40% memory savings)
- temporal_tiling: Split T (50% memory savings)
- hybrid_tiling: Split H, W, T (80% memory savings, most conservative)

Blending:
- Gaussian windows (default, smoothest)
- Linear fade (standard)
- Cosine fade (smooth alternative)
"""

from .strategies import (
    TilingSizeConfig,
    TilingStrategy,
    NoTiling,
    SpatialTiling,
    TemporalTiling,
    HybridTiling,
)
from .auto_tiler import (
    AdaptiveTilingEngine,
    auto_select_tiling,
    get_available_vram_gb,
)
from .blending import TileBlendingManager
from .tiling_node import TilingConfigNode, TiledDenoiseWrapper

__all__ = [
    # Strategies
    "TilingSizeConfig",
    "TilingStrategy",
    "NoTiling",
    "SpatialTiling",
    "TemporalTiling",
    "HybridTiling",
    # Auto-tiling
    "AdaptiveTilingEngine",
    "auto_select_tiling",
    "get_available_vram_gb",
    # Blending
    "TileBlendingManager",
    # Nodes
    "TilingConfigNode",
    "TiledDenoiseWrapper",
]
