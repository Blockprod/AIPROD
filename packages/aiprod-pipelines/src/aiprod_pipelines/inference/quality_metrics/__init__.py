"""Quality metrics module for comprehensive inference quality monitoring.

Provides:
- FVVR: CLIP-based prompt alignment (semantic quality)
- LPIPS: Learned perceptual image patch similarity (visual quality)
- Motion: Optical flow-based motion smoothness (temporal quality)
- Quality Aggregation: Unified quality scoring from multiple dimensions
"""

from .fvvr import (
    FVVRMetric,
    FVVRCalculator,
    FVVRTracker,
    compute_fvvr_efficient,
)

from .lpips import (
    LPIPSMetric,
    LPIPSNet,
    LPIPSCalculator,
    compute_lpips_batch_efficient,
)

from .motion import (
    MotionMetric,
    OpticalFlowEstimator,
    MotionConsistencyCalculator,
)

from .quality_monitor import (
    QualityScore,
    QualityMonitor,
    QualityAggregator,
)

__all__ = [
    # FVVR exports
    "FVVRMetric",
    "FVVRCalculator",
    "FVVRTracker",
    "compute_fvvr_efficient",
    
    # LPIPS exports
    "LPIPSMetric",
    "LPIPSNet",
    "LPIPSCalculator",
    "compute_lpips_batch_efficient",
    
    # Motion exports
    "MotionMetric",
    "OpticalFlowEstimator",
    "MotionConsistencyCalculator",
    
    # Aggregation exports
    "QualityScore",
    "QualityMonitor",
    "QualityAggregator",
]
