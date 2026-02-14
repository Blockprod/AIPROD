"""
Adaptive Batch Sizing Engine

Dynamically adjusts batch sizes based on runtime performance and resource constraints.
Learns optimal batch sizes through measurement and prediction.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Tuple, List
from enum import Enum
import statistics


@dataclass
class BatchPerformanceMetrics:
    """Metrics collected during batch execution."""
    batch_size: int
    latency_ms: float
    throughput_samples_per_sec: float
    memory_peak_mb: float
    memory_avg_mb: float
    gpu_utilization_percent: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    @property
    def efficiency_score(self) -> float:
        """Score batch efficiency: throughput * utilization / memory."""
        if self.memory_avg_mb <= 0:
            return 0.0
        return (self.throughput_samples_per_sec * self.gpu_utilization_percent) / self.memory_avg_mb


@dataclass
class AdaptiveBatcherConfig:
    """Configuration for adaptive batcher."""
    initial_batch_size: int = 32
    learning_rate: float = 0.1
    window_size: int = 10  # Number of recent measurements to consider
    min_stable_iterations: int = 5  # Iterations before accepting as stable
    adjustment_threshold: float = 0.05  # 5% improvement needed to adjust
    probe_interval: int = 100  # Probe every N iterations
    enable_stability_tracking: bool = True
    historical_discount: float = 0.9  # Discount old measurements


class AdaptiveIntelligence(Enum):
    """Types of adaptive intelligence for batch sizing."""
    PERFORMANCE_BASED = "performance"  # Optimize for throughput/latency
    MEMORY_BASED = "memory"  # Optimize for memory efficiency
    POWER_BASED = "power"  # Optimize for power consumption
    BALANCED = "balanced"  # Multi-objective optimization


class AdaptiveBatcher:
    """Adaptively adjusts batch size based on runtime metrics."""
    
    def __init__(self, config: AdaptiveBatcherConfig):
        self.config = config
        self.current_batch_size = config.initial_batch_size
        self.metrics_history: List[BatchPerformanceMetrics] = []
        self.is_stable = False
        self.stable_batch_size: Optional[int] = None
        self.total_iterations = 0
        
    def record_measurement(self, metrics: BatchPerformanceMetrics) -> None:
        """Record batch performance metrics."""
        self.metrics_history.append(metrics)
        self.total_iterations += 1
        
        # Keep only recent measurements
        if len(self.metrics_history) > self.config.window_size:
            self.metrics_history.pop(0)
    
    def should_adjust_batch_size(self) -> bool:
        """Determine if batch size should be adjusted."""
        if len(self.metrics_history) < self.config.min_stable_iterations:
            return False
        
        if self.is_stable:
            return False
        
        # Check if recent performance is improving
        recent = self.metrics_history[-self.config.min_stable_iterations:]
        scores = [m.efficiency_score for m in recent]
        
        # Require threshold improvement
        if len(scores) < 2:
            return False
        
        improvement = (scores[-1] - scores[0]) / abs(scores[0]) if scores[0] != 0 else 0
        return improvement > self.config.adjustment_threshold
    
    def recommend_adjustment(self) -> Optional[int]:
        """Recommend batch size adjustment."""
        if not self.should_adjust_batch_size():
            return None
        
        recent_metrics = self.metrics_history[-self.config.min_stable_iterations:]
        
        # Analyze trend
        efficiency_scores = [m.efficiency_score for m in recent_metrics]
        batch_sizes = [m.batch_size for m in recent_metrics]
        
        if not batch_sizes:
            return None
        
        best_idx = efficiency_scores.index(max(efficiency_scores))
        best_batch = batch_sizes[best_idx]
        
        # Adjust towards best
        if best_batch > self.current_batch_size:
            # Try increasing
            adjustment = int(self.current_batch_size * (1 + self.config.learning_rate))
            return min(adjustment, best_batch)
        else:
            # Try decreasing
            adjustment = int(self.current_batch_size * (1 - self.config.learning_rate))
            return max(adjustment, best_batch)
    
    def apply_adjustment(self, new_batch_size: int) -> None:
        """Apply batch size adjustment."""
        self.current_batch_size = new_batch_size
    
    def mark_stable(self, batch_size: int) -> None:
        """Mark batch size as stable."""
        self.is_stable = True
        self.stable_batch_size = batch_size
    
    def get_stability_confidence(self) -> float:
        """Get confidence that current batch size is stable (0-1)."""
        if not self.metrics_history:
            return 0.0
        
        recent = self.metrics_history[-self.config.min_stable_iterations:]
        if len(recent) < self.config.min_stable_iterations:
            return 0.0
        
        scores = [m.efficiency_score for m in recent]
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0
        
        # Lower variance = higher confidence
        return max(0.0, 1.0 - (std_dev / (max(scores) if max(scores) > 0 else 1)))


class PerformancePredictor:
    """Predicts batch performance before execution."""
    
    def __init__(self):
        self.calibration_points: List[Tuple[int, float, float]] = []  # (batch_size, latency, memory)
        
    def calibrate(self, measurements: List[BatchPerformanceMetrics]) -> None:
        """Calibrate predictor with measurements."""
        self.calibration_points.clear()
        for m in measurements:
            self.calibration_points.append(
                (m.batch_size, m.latency_ms, m.memory_peak_mb)
            )
    
    def predict_latency(self, batch_size: int) -> float:
        """Predict latency for batch size."""
        if not self.calibration_points:
            return 10.0 * batch_size  # Default estimate
        
        # Linear interpolation/extrapolation
        sorted_points = sorted(self.calibration_points, key=lambda x: x[0])
        
        for i in range(len(sorted_points) - 1):
            b1, l1, _ = sorted_points[i]
            b2, l2, _ = sorted_points[i + 1]
            
            if b1 <= batch_size <= b2:
                # Interpolate
                ratio = (batch_size - b1) / (b2 - b1) if b2 != b1 else 0
                return l1 + ratio * (l2 - l1)
        
        # Extrapolate using last two points
        b_last, l_last, _ = sorted_points[-1]
        if len(sorted_points) >= 2:
            b_prev, l_prev, _ = sorted_points[-2]
            slope = (l_last - l_prev) / (b_last - b_prev) if b_last != b_prev else 0
            return l_last + slope * (batch_size - b_last)
        
        return l_last * (batch_size / b_last) if b_last > 0 else 10.0
    
    def predict_memory(self, batch_size: int) -> float:
        """Predict memory for batch size."""
        if not self.calibration_points:
            return 100.0 * batch_size  # Default estimate
        
        sorted_points = sorted(self.calibration_points, key=lambda x: x[0])
        
        for i in range(len(sorted_points) - 1):
            b1, _, m1 = sorted_points[i]
            b2, _, m2 = sorted_points[i + 1]
            
            if b1 <= batch_size <= b2:
                ratio = (batch_size - b1) / (b2 - b1) if b2 != b1 else 0
                return m1 + ratio * (m2 - m1)
        
        b_last, _, m_last = sorted_points[-1]
        if len(sorted_points) >= 2:
            b_prev, _, m_prev = sorted_points[-2]
            slope = (m_last - m_prev) / (b_last - b_prev) if b_last != b_prev else 0
            return m_last + slope * (batch_size - b_last)
        
        return m_last * (batch_size / b_last) if b_last > 0 else 100.0
    
    def predict_throughput(self, batch_size: int) -> float:
        """Predict throughput for batch size."""
        latency_ms = self.predict_latency(batch_size)
        if latency_ms <= 0:
            return 0.0
        return (batch_size * 1000) / latency_ms  # samples per second


class DynamicBatchOptimizer:
    """Main orchestrator for dynamic batch optimization."""
    
    def __init__(self, config: AdaptiveBatcherConfig):
        self.config = config
        self.adaptive_batcher = AdaptiveBatcher(config)
        self.performance_predictor = PerformancePredictor()
        self.intelligence_mode = AdaptiveIntelligence.BALANCED
        
    def optimize_batch_size(
        self,
        forward_fn: Callable[[int], BatchPerformanceMetrics],
        target_iterations: int = 1000,
    ) -> int:
        """Find optimal batch size through adaptive probing."""
        current_batch = self.config.initial_batch_size
        
        for iteration in range(target_iterations):
            # Record metrics
            metrics = forward_fn(current_batch)
            self.adaptive_batcher.record_measurement(metrics)
            
            # Check if should adjust
            if iteration % self.config.probe_interval == 0 and iteration > 0:
                recommendation = self.adaptive_batcher.recommend_adjustment()
                if recommendation is not None:
                    current_batch = recommendation
                    self.adaptive_batcher.apply_adjustment(current_batch)
            
            # Check for stability
            confidence = self.adaptive_batcher.get_stability_confidence()
            if confidence > 0.9 and iteration > self.config.min_stable_iterations:
                self.adaptive_batcher.mark_stable(current_batch)
                break
        
        return self.adaptive_batcher.current_batch_size
