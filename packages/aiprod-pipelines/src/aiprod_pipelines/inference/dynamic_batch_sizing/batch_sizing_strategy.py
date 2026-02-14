"""
Dynamic Batch Sizing Strategies

Provides flexible batch sizing strategies optimized for different hardware and model configurations.
Defines the core abstractions for batch decisions and resource constraints.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class BatchSizingStrategy(Enum):
    """Enumeration of available batch sizing strategies."""
    FIXED = "fixed"  # Static batch size
    LINEAR = "linear"  # Linear scaling with memory
    EXPONENTIAL = "exponential"  # Exponential probing
    BINARY_SEARCH = "binary_search"  # Binary search for optimal
    ADAPTIVE = "adaptive"  # Learning-based adaptation
    CONSERVATIVE = "conservative"  # Memory-safe fallback
    AGGRESSIVE = "aggressive"  # Maximum throughput focus
    POWER_AWARE = "power_aware"  # Power consumption aware


class MemoryManagementMode(Enum):
    """Memory management strategies for batch sizing."""
    MEMORY_INTENSIVE = "memory_intensive"  # Maximize memory usage
    BALANCED = "balanced"  # Balance memory and compute
    COMPUTE_INTENSIVE = "compute_intensive"  # Minimize memory, maximize compute
    ULTRA_LOW = "ultra_low"  # Minimal memory footprint


class DeviceType(Enum):
    """Device type enumeration for batch sizing."""
    GPU_HIGH_END = "gpu_high_end"  # A100, H100 (80GB+)
    GPU_MID_RANGE = "gpu_mid_range"  # RTX 4090, A6000 (24-48GB)
    GPU_CONSUMER = "gpu_consumer"  # RTX 4080, GTX 4090 (12-24GB)
    GPU_MOBILE = "gpu_mobile"  # NVIDIA Jetson (2-8GB)
    CPU = "cpu"  # CPU inference
    TPU = "tpu"  # TPU compute
    ACCELERATOR = "accelerator"  # Custom accelerator


@dataclass
class BatchSizingConstraints:
    """Constraints for batch sizing optimization."""
    min_batch_size: int = 1
    max_batch_size: int = 512
    memory_limit_mb: float = 40000.0  # 40GB default
    latency_target_ms: float = 1000.0  # 1 second default
    throughput_target_samples_per_sec: float = 100.0
    power_limit_watts: Optional[float] = None
    optimal_utilization_target: float = 0.9  # 90% GPU utilization
    prefer_multiple_of: int = 1  # Batch size should be multiple of N
    
    def validate(self) -> bool:
        """Validate constraints consistency."""
        if self.min_batch_size <= 0:
            return False
        if self.max_batch_size < self.min_batch_size:
            return False
        if self.memory_limit_mb <= 0:
            return False
        return True
    
    def adjust_for_device(self, device_type: DeviceType) -> None:
        """Adjust constraints based on device type."""
        if device_type == DeviceType.GPU_HIGH_END:
            self.memory_limit_mb = 80000.0
            self.max_batch_size = 512
        elif device_type == DeviceType.GPU_MID_RANGE:
            self.memory_limit_mb = 40000.0
            self.max_batch_size = 256
        elif device_type == DeviceType.GPU_CONSUMER:
            self.memory_limit_mb = 20000.0
            self.max_batch_size = 128
        elif device_type == DeviceType.GPU_MOBILE:
            self.memory_limit_mb = 6000.0
            self.max_batch_size = 16
        elif device_type == DeviceType.CPU:
            self.memory_limit_mb = 8000.0
            self.max_batch_size = 32


@dataclass
class BatchSizingConfig:
    """Configuration for batch sizing engine."""
    strategy: BatchSizingStrategy = BatchSizingStrategy.ADAPTIVE
    memory_mode: MemoryManagementMode = MemoryManagementMode.BALANCED
    device_type: DeviceType = DeviceType.GPU_MID_RANGE
    constraints: BatchSizingConstraints = field(default_factory=BatchSizingConstraints)
    probing_samples: int = 10  # Number of samples for strategy probing
    warmup_iterations: int = 5  # Warmup iterations before measurement
    enable_caching: bool = True  # Cache batch size decisions
    enable_prediction: bool = True  # Enable latency/memory prediction
    update_frequency: int = 100  # Update batch size every N iterations
    emergency_fallback_enabled: bool = True  # Fallback on OOM
    profile_memory_overhead: bool = True  # Account for framework overhead
    
    def __post_init__(self):
        self.constraints.adjust_for_device(self.device_type)


@dataclass
class BatchSizeDecision:
    """Result of batch sizing decision."""
    batch_size: int
    strategy_used: BatchSizingStrategy
    predicted_memory_mb: float
    predicted_latency_ms: float
    predicted_throughput: float
    confidence_score: float = 1.0
    reasoning: str = ""
    alternative_sizes: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_size": self.batch_size,
            "strategy": self.strategy_used.value,
            "memory_mb": self.predicted_memory_mb,
            "latency_ms": self.predicted_latency_ms,
            "throughput": self.predicted_throughput,
            "confidence": self.confidence_score,
            "reasoning": self.reasoning,
        }


class BatchSizingOptimizer:
    """Base class for batch sizing optimization strategies."""
    
    def __init__(self, config: BatchSizingConfig):
        self.config = config
        self.decision_history: List[BatchSizeDecision] = []
        
    def propose_batch_size(
        self,
        memory_per_sample_mb: float,
        latency_per_sample_ms: float,
        throughput_per_sample: float,
    ) -> BatchSizeDecision:
        """Propose optimal batch size based on resource profiles."""
        raise NotImplementedError
    
    def update_from_measurement(
        self,
        batch_size: int,
        actual_memory_mb: float,
        actual_latency_ms: float,
        actual_throughput: float,
        success: bool = True,
    ) -> None:
        """Update optimizer with actual measurement."""
        raise NotImplementedError


class LinearBatchSizer(BatchSizingOptimizer):
    """Linear batch sizing strategy - scale linearly with available memory."""
    
    def propose_batch_size(
        self,
        memory_per_sample_mb: float,
        latency_per_sample_ms: float,
        throughput_per_sample: float,
    ) -> BatchSizeDecision:
        """Propose batch size with linear memory scaling."""
        if memory_per_sample_mb <= 0:
            memory_per_sample_mb = 0.1
        
        # Calculate max batch based on memory constraint
        max_batch_from_memory = int(
            self.config.constraints.memory_limit_mb / memory_per_sample_mb
        )
        
        # Apply constraints
        proposed_batch = min(
            max_batch_from_memory,
            self.config.constraints.max_batch_size,
        )
        proposed_batch = max(proposed_batch, self.config.constraints.min_batch_size)
        
        # Align to preferred multiple
        if self.config.constraints.prefer_multiple_of > 1:
            proposed_batch = (proposed_batch // self.config.constraints.prefer_multiple_of) * self.config.constraints.prefer_multiple_of
        
        pred_memory = memory_per_sample_mb * proposed_batch
        pred_latency = latency_per_sample_ms * proposed_batch
        pred_throughput = throughput_per_sample / latency_per_sample_ms if latency_per_sample_ms > 0 else 0
        
        return BatchSizeDecision(
            batch_size=proposed_batch,
            strategy_used=BatchSizingStrategy.LINEAR,
            predicted_memory_mb=pred_memory,
            predicted_latency_ms=pred_latency,
            predicted_throughput=pred_throughput,
            confidence_score=0.8,
            reasoning=f"Linear scaling: memory constraint allows {proposed_batch} samples",
        )


class ExponentialBatchSizer(BatchSizingOptimizer):
    """Exponential probing strategy - binary search for optimal batch size."""
    
    def __init__(self, config: BatchSizingConfig):
        super().__init__(config)
        self.low_batch = config.constraints.min_batch_size
        self.high_batch = config.constraints.max_batch_size
        
    def propose_batch_size(
        self,
        memory_per_sample_mb: float,
        latency_per_sample_ms: float,
        throughput_per_sample: float,
    ) -> BatchSizeDecision:
        """Propose batch size using binary search."""
        mid_batch = (self.low_batch + self.high_batch) // 2
        
        pred_memory = memory_per_sample_mb * mid_batch
        within_memory = pred_memory <= self.config.constraints.memory_limit_mb
        
        decision = BatchSizeDecision(
            batch_size=mid_batch,
            strategy_used=BatchSizingStrategy.BINARY_SEARCH,
            predicted_memory_mb=pred_memory,
            predicted_latency_ms=latency_per_sample_ms * mid_batch,
            predicted_throughput=throughput_per_sample * mid_batch if latency_per_sample_ms > 0 else 0,
            confidence_score=0.85,
            reasoning=f"Binary search probe: batch {mid_batch} (memory within limit: {within_memory})",
        )
        
        return decision


class PerformanceProfile:
    """Profile of model performance characteristics."""
    
    def __init__(
        self,
        model_name: str,
        base_memory_mb: float,
        memory_per_sample_mb: float,
        base_latency_ms: float,
        latency_per_sample_ms: float,
    ):
        self.model_name = model_name
        self.base_memory_mb = base_memory_mb
        self.memory_per_sample_mb = memory_per_sample_mb
        self.base_latency_ms = base_latency_ms
        self.latency_per_sample_ms = latency_per_sample_ms
    
    def estimate_memory(self, batch_size: int) -> float:
        """Estimate total memory for batch size."""
        return self.base_memory_mb + self.memory_per_sample_mb * batch_size
    
    def estimate_latency(self, batch_size: int) -> float:
        """Estimate latency for batch size."""
        return self.base_latency_ms + self.latency_per_sample_ms * batch_size
    
    def is_feasible(self, batch_size: int, memory_limit_mb: float) -> bool:
        """Check if batch size is feasible within memory limit."""
        return self.estimate_memory(batch_size) <= memory_limit_mb
