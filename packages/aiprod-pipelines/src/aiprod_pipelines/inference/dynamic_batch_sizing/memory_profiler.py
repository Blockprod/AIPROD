"""
Memory Profiling and Tracking

Monitors and profiles GPU/CPU memory usage to inform batch sizing decisions.
Tracks memory allocation patterns, fragmentation, and predicts memory requirements.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time


class MemoryAllocationPattern(Enum):
    """Patterns of memory allocation."""
    LINEAR = "linear"  # Memory grows linearly with batch
    SUPERLINEAR = "superlinear"  # Subquadratic growth (e.g., attention)
    FRAGMENTED = "fragmented"  # Memory fragmentation issues
    FIXED_OVERHEAD = "fixed_overhead"  # Large fixed overhead


@dataclass
class MemorySnapshot:
    """Snapshot of memory state at a point in time."""
    timestamp: float
    batch_size: int
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    fragmentation_ratio: float = 0.0
    peak_mb: float = 0.0
    
    @property
    def utilization(self) -> float:
        """Memory utilization percentage."""
        if self.reserved_mb <= 0:
            return 0.0
        return (self.allocated_mb / self.reserved_mb) * 100.0


@dataclass
class MemoryProfile:
    """Complete memory profile for a model."""
    model_name: str
    device_type: str
    base_memory_mb: float = 0.0
    memory_per_sample_mb: float = 0.1
    peak_memory_multiplier: float = 1.2  # Peak can be 1.2x average
    snapshots: List[MemorySnapshot] = field(default_factory=list)
    allocation_pattern: MemoryAllocationPattern = MemoryAllocationPattern.LINEAR
    
    def add_snapshot(self, snapshot: MemorySnapshot) -> None:
        """Add memory snapshot to profile."""
        self.snapshots.append(snapshot)
    
    def estimate_memory(self, batch_size: int, include_peak: bool = True) -> float:
        """Estimate total memory for batch size."""
        estimated = self.base_memory_mb + self.memory_per_sample_mb * batch_size
        if include_peak:
            estimated *= self.peak_memory_multiplier
        return estimated
    
    def get_fragmentation_estimate(self) -> float:
        """Estimate fragmentation from snapshots."""
        if not self.snapshots:
            return 0.0
        return sum(s.fragmentation_ratio for s in self.snapshots) / len(self.snapshots)
    
    def recommend_batch_for_memory(self, target_memory_mb: float, include_peak: bool = True) -> int:
        """Recommend batch size to fit within target memory."""
        if self.memory_per_sample_mb <= 0:
            return 1
        
        divisor = self.peak_memory_multiplier if include_peak else 1.0
        budget = target_memory_mb / divisor - self.base_memory_mb
        batch_size = max(1, int(budget / self.memory_per_sample_mb))
        return batch_size


class GPUMemoryMonitor:
    """Monitors GPU memory usage and provides recommendations."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.memory_profiles: Dict[str, MemoryProfile] = {}
        self.current_snapshot: Optional[MemorySnapshot] = None
        
    def get_current_memory(self) -> Tuple[float, float, float]:
        """Get current memory state: (allocated, reserved, free) in MB.
        
        Note: In production, this would use torch.cuda.memory_stats() or similar.
        """
        # Simulated for now
        return (0.0, 0.0, 0.0)
    
    def profile_batch_sizes(
        self,
        model_forward_fn,
        batch_sizes: List[int],
        warmup_iterations: int = 5,
    ) -> MemoryProfile:
        """Profile model memory usage across different batch sizes."""
        measurements = []
        
        for batch_size in batch_sizes:
            # Warmup
            for _ in range(warmup_iterations):
                try:
                    model_forward_fn(batch_size)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        break
            
            # Measure
            alloc, reserved, free = self.get_current_memory()
            peak = reserved - free if free > 0 else alloc
            
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                batch_size=batch_size,
                allocated_mb=alloc,
                reserved_mb=reserved,
                free_mb=free,
                peak_mb=peak,
            )
            measurements.append(snapshot)
        
        # Infer profile from measurements
        profile = self._infer_profile(measurements)
        return profile
    
    def _infer_profile(self, snapshots: List[MemorySnapshot]) -> MemoryProfile:
        """Infer memory profile from snapshots."""
        profile = MemoryProfile(
            model_name="inferred",
            device_type="gpu",
        )
        
        for snapshot in snapshots:
            profile.add_snapshot(snapshot)
        
        # Estimate parameters
        if len(snapshots) >= 2:
            s1, s2 = snapshots[0], snapshots[-1]
            batch_diff = s2.batch_size - s1.batch_size
            if batch_diff > 0:
                profile.memory_per_sample_mb = (s2.allocated_mb - s1.allocated_mb) / batch_diff
                profile.base_memory_mb = s1.allocated_mb - profile.memory_per_sample_mb * s1.batch_size
        
        return profile
    
    def get_oom_safe_batch_size(
        self,
        profile: MemoryProfile,
        available_memory_mb: float,
        safety_margin: float = 0.85,
    ) -> int:
        """Get OOM-safe batch size with safety margin."""
        target_memory = available_memory_mb * safety_margin
        return profile.recommend_batch_for_memory(target_memory, include_peak=True)


class MemoryFragmentationEstimator:
    """Estimates and mitigates memory fragmentation effects."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.fragmentation_history: List[float] = []
    
    def estimate_fragmentation(self, allocated_mb: float, reserved_mb: float) -> float:
        """Estimate fragmentation ratio."""
        if reserved_mb <= 0:
            return 0.0
        # Fragmentation = 1 - (allocated / reserved)
        return 1.0 - min(1.0, allocated_mb / reserved_mb)
    
    def get_effective_memory(
        self,
        total_reserved_mb: float,
        fragmentation_ratio: float = 0.0,
    ) -> float:
        """Get effectively usable memory accounting for fragmentation."""
        return total_reserved_mb * (1.0 - fragmentation_ratio)
    
    def should_compact_memory(self, fragmentation_threshold: float = 0.3) -> bool:
        """Recommend memory compaction if fragmentation too high."""
        if not self.fragmentation_history:
            return False
        recent_avg = sum(self.fragmentation_history[-5:]) / min(5, len(self.fragmentation_history))
        return recent_avg > fragmentation_threshold


class ComputeMemoryTradeoffAnalyzer:
    """Analyzes tradeoffs between compute and memory efficiency."""
    
    @staticmethod
    def compute_latency_memory_curve(
        batch_sizes: List[int],
        latencies_ms: List[float],
        memory_mb: List[float],
    ) -> Dict[str, float]:
        """Compute metrics for latency-memory tradeoff."""
        if not batch_sizes or len(batch_sizes) != len(latencies_ms):
            return {}
        
        # Compute Pareto frontier
        pareto_points = []
        for i, (b, l, m) in enumerate(zip(batch_sizes, latencies_ms, memory_mb)):
            is_dominated = False
            for j in range(len(batch_sizes)):
                if i != j and latencies_ms[j] <= l and memory_mb[j] <= m:
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_points.append((b, l, m))
        
        return {
            "pareto_frontier_points": len(pareto_points),
            "average_efficiency": sum(b / (l * m) for b, l, m in pareto_points) / len(pareto_points) if pareto_points else 0,
        }
