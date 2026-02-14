"""
Dynamic Batch Sizing Module

Provides adaptive batch sizing strategies optimized for different hardware configurations and models.
Automatically determines optimal batch sizes based on memory, compute, and throughput constraints.
"""

# Batch Sizing Strategies
from .batch_sizing_strategy import (
    BatchSizingStrategy,
    MemoryManagementMode,
    DeviceType,
    BatchSizingConstraints,
    BatchSizingConfig,
    BatchSizeDecision,
    BatchSizingOptimizer,
    LinearBatchSizer,
    ExponentialBatchSizer,
    PerformanceProfile,
)

# Memory Profiling
from .memory_profiler import (
    MemoryAllocationPattern,
    MemorySnapshot,
    MemoryProfile,
    GPUMemoryMonitor,
    MemoryFragmentationEstimator,
    ComputeMemoryTradeoffAnalyzer,
)

# Adaptive Batching
from .adaptive_batcher import (
    BatchPerformanceMetrics,
    AdaptiveBatcherConfig,
    AdaptiveIntelligence,
    AdaptiveBatcher,
    PerformancePredictor,
    DynamicBatchOptimizer,
)

# Batch Caching
from .batch_cache import (
    CacheKey,
    CacheEntry,
    BatchSizingCache,
    ModelProfileRegistry,
    BatchSizeRecommender,
)

# Performance Estimation
from .performance_estimator import (
    PerformanceModel,
    PerformanceCurveParams,
    PerformanceEstimate,
    CurveEstimator,
    PerformanceProfiler,
    ResourceEstimator,
    PerformanceSnapshot,
    PerformanceMonitor,
)

# Dynamic Batch Optimization
from .dynamic_batch_optimization import (
    DynamicBatchState,
    DynamicBatchConfig,
    DynamicBatchMetrics,
    DynamicBatchSizer,
    DynamicBatchNode,
    MultiModelBatchOptimizer,
)

__all__ = [
    # Batch Sizing Strategies (13 exports)
    "BatchSizingStrategy",
    "MemoryManagementMode",
    "DeviceType",
    "BatchSizingConstraints",
    "BatchSizingConfig",
    "BatchSizeDecision",
    "BatchSizingOptimizer",
    "LinearBatchSizer",
    "ExponentialBatchSizer",
    "PerformanceProfile",
    
    # Memory Profiling (7 exports)
    "MemoryAllocationPattern",
    "MemorySnapshot",
    "MemoryProfile",
    "GPUMemoryMonitor",
    "MemoryFragmentationEstimator",
    "ComputeMemoryTradeoffAnalyzer",
    
    # Adaptive Batching (7 exports)
    "BatchPerformanceMetrics",
    "AdaptiveBatcherConfig",
    "AdaptiveIntelligence",
    "AdaptiveBatcher",
    "PerformancePredictor",
    "DynamicBatchOptimizer",
    
    # Batch Caching (5 exports)
    "CacheKey",
    "CacheEntry",
    "BatchSizingCache",
    "ModelProfileRegistry",
    "BatchSizeRecommender",
    
    # Performance Estimation (8 exports)
    "PerformanceModel",
    "PerformanceCurveParams",
    "PerformanceEstimate",
    "CurveEstimator",
    "PerformanceProfiler",
    "ResourceEstimator",
    "PerformanceSnapshot",
    "PerformanceMonitor",
    
    # Dynamic Batch Optimization (7 exports)
    "DynamicBatchState",
    "DynamicBatchConfig",
    "DynamicBatchMetrics",
    "DynamicBatchSizer",
    "DynamicBatchNode",
    "MultiModelBatchOptimizer",
]
