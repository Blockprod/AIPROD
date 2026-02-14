"""
Performance Estimation and Prediction

Estimates latency, throughput, and memory requirements for different batch sizes.
Provides tools for performance profiling and prediction without full training.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import math


class PerformanceModel(Enum):
    """Mathematical models for performance prediction."""
    LINEAR = "linear"  # y = a*x + b
    QUADRATIC = "quadratic"  # y = a*x^2 + b*x + c
    EXPONENTIAL = "exponential"  # y = a * e^(b*x)
    POWER_LAW = "power_law"  # y = a * x^b
    SIGMOID = "sigmoid"  # Saturation model


@dataclass
class PerformanceCurveParams:
    """Parameters for performance curve fitting."""
    model_type: PerformanceModel
    coefficients: List[float]  # [a, b, c, ...]
    r_squared: float = 0.0  # Goodness of fit
    
    def predict(self, x: float) -> float:
        """Predict value at x using model."""
        if self.model_type == PerformanceModel.LINEAR:
            a, b = self.coefficients
            return a * x + b
        elif self.model_type == PerformanceModel.QUADRATIC:
            a, b, c = self.coefficients
            return a * x**2 + b * x + c
        elif self.model_type == PerformanceModel.EXPONENTIAL:
            a, b = self.coefficients
            return a * math.exp(b * x)
        elif self.model_type == PerformanceModel.POWER_LAW:
            a, b = self.coefficients
            return a * (x ** b)
        elif self.model_type == PerformanceModel.SIGMOID:
            a, b, c = self.coefficients
            return a / (1 + math.exp(-b * (x - c)))
        return 0.0


@dataclass
class PerformanceEstimate:
    """Estimated performance metrics."""
    batch_size: int
    estimated_latency_ms: float
    estimated_throughput_samples_per_sec: float
    estimated_memory_mb: float
    estimated_gpu_util_percent: float
    model_confidence: float = 0.8
    bottleneck: str = "compute"  # "compute", "memory", "communication"
    

class CurveEstimator:
    """Fits performance curves from measurements."""
    
    @staticmethod
    def fit_linear(x_values: List[float], y_values: List[float]) -> PerformanceCurveParams:
        """Fit linear model y = a*x + b."""
        if len(x_values) < 2:
            return PerformanceCurveParams(PerformanceModel.LINEAR, [1.0, 0.0])
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xx = sum(x**2 for x in x_values)
        sum_xy = sum(x*y for x, y in zip(x_values, y_values))
        
        denom = n * sum_xx - sum_x**2
        if denom == 0:
            return PerformanceCurveParams(PerformanceModel.LINEAR, [1.0, 0.0])
        
        a = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - a * sum_x) / n
        
        # Calculate R²
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean)**2 for y in y_values)
        ss_res = sum((y - (a*x + b))**2 for x, y in zip(x_values, y_values))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return PerformanceCurveParams(PerformanceModel.LINEAR, [a, b], r_squared)
    
    @staticmethod
    def fit_power_law(x_values: List[float], y_values: List[float]) -> PerformanceCurveParams:
        """Fit power law model y = a * x^b."""
        if len(x_values) < 2 or any(x <= 0 or y <= 0 for x, y in zip(x_values, y_values)):
            return PerformanceCurveParams(PerformanceModel.POWER_LAW, [1.0, 1.0])
        
        import math
        log_x = [math.log(x) for x in x_values]
        log_y = [math.log(y) for y in y_values]
        
        # Linear fit on log scale
        linear_model = CurveEstimator.fit_linear(log_x, log_y)
        a = math.exp(linear_model.coefficients[1])
        b = linear_model.coefficients[0]
        
        return PerformanceCurveParams(PerformanceModel.POWER_LAW, [a, b], linear_model.r_squared)


class PerformanceProfiler:
    """Profiles model performance across batch sizes."""
    
    def __init__(self):
        self.measurements: List[Tuple[int, float, float]] = []  # (batch, latency, memory)
        self.latency_model: Optional[PerformanceCurveParams] = None
        self.memory_model: Optional[PerformanceCurveParams] = None
        
    def add_measurement(self, batch_size: int, latency_ms: float, memory_mb: float) -> None:
        """Add performance measurement."""
        self.measurements.append((batch_size, latency_ms, memory_mb))
    
    def fit_models(self) -> None:
        """Fit performance models from measurements."""
        if len(self.measurements) < 2:
            return
        
        batches, latencies, memories = zip(*self.measurements)
        
        # Fit latency model
        self.latency_model = CurveEstimator.fit_power_law(
            list(batches), list(latencies)
        )
        
        # Fit memory model
        self.memory_model = CurveEstimator.fit_linear(
            list(batches), list(memories)
        )
    
    def estimate_latency(self, batch_size: int) -> float:
        """Estimate latency for batch size."""
        if not self.latency_model:
            return 10.0 * batch_size
        return self.latency_model.predict(batch_size)
    
    def estimate_memory(self, batch_size: int) -> float:
        """Estimate memory for batch size."""
        if not self.memory_model:
            return 100.0 * batch_size
        return self.memory_model.predict(batch_size)
    
    def estimate_throughput(self, batch_size: int) -> float:
        """Estimate throughput for batch size."""
        latency = self.estimate_latency(batch_size)
        if latency <= 0:
            return 0.0
        return (batch_size * 1000) / latency


class ResourceEstimator:
    """Estimates resource usage and bottlenecks."""
    
    def __init__(self, device_specs: Dict[str, float]):
        """Initialize with device specifications.
        
        Args:
            device_specs: {"memory_gb": 80, "bandwidth_gbps": 2000, "compute_tflops": 312}
        """
        self.device_specs = device_specs
    
    def identify_bottleneck(
        self,
        batch_size: int,
        ops_per_sample: float,
        memory_per_sample_mb: float,
        memory_bandwidth_util: float = 0.8,
    ) -> str:
        """Identify whether workload is compute or memory bound."""
        # Compute-bound if ops per byte > memory bandwidth
        bytes_per_sample = memory_per_sample_mb * 1024 * 1024
        memory_bw = self.device_specs.get("bandwidth_gbps", 2000) * 1e9 / 8  # bytes/sec
        
        ops_per_second = ops_per_sample * batch_size * (1000 / 10)  # Rough estimate
        memory_ops_per_second = (memory_per_sample_mb * batch_size) / memory_bandwidth_util
        
        if ops_per_second > memory_ops_per_second:
            return "compute"
        else:
            return "memory"
    
    def estimate_utilization(
        self,
        latency_ms: float,
        batch_size: int,
        ops_per_sample: float,
    ) -> float:
        """Estimate GPU utilization percentage."""
        if latency_ms <= 0:
            return 0.0
        
        compute_tflops = self.device_specs.get("compute_tflops", 312)
        theoretical_time = (ops_per_sample * batch_size) / (compute_tflops * 1e12)
        utilization = (theoretical_time / (latency_ms / 1000)) * 100
        
        return min(100.0, utilization)


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time."""
    timestamp: float
    batch_size: int
    latency_ms: float
    throughput_samples_per_sec: float
    memory_mb: float
    gpu_utilization_percent: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class PerformanceMonitor:
    """Real-time monitoring of inference performance."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.snapshots: List[PerformanceSnapshot] = []
        self.aggregated_stats: Dict[str, float] = {}
        
    def record_snapshot(self, snapshot: PerformanceSnapshot) -> None:
        """Record performance snapshot."""
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.window_size:
            self.snapshots.pop(0)
        
        self._update_aggregates()
    
    def _update_aggregates(self) -> None:
        """Update aggregated statistics."""
        if not self.snapshots:
            return
        
        import statistics
        latencies = [s.latency_ms for s in self.snapshots]
        throughputs = [s.throughput_samples_per_sec for s in self.snapshots]
        memories = [s.memory_mb for s in self.snapshots]
        utils = [s.gpu_utilization_percent for s in self.snapshots]
        
        self.aggregated_stats = {
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": self._percentile(latencies, 0.95),
            "avg_throughput": statistics.mean(throughputs),
            "avg_memory_mb": statistics.mean(memories),
            "avg_utilization_percent": statistics.mean(utils),
        }
    
    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]
    
    def get_performance_forecast(self, batch_size: int) -> PerformanceEstimate:
        """Forecast performance for batch size."""
        profiler = PerformanceProfiler()
        for s in self.snapshots:
            profiler.add_measurement(s.batch_size, s.latency_ms, s.memory_mb)
        profiler.fit_models()
        
        latency = profiler.estimate_latency(batch_size)
        throughput = profiler.estimate_throughput(batch_size)
        memory = profiler.estimate_memory(batch_size)
        utilization = self.aggregated_stats.get("avg_utilization_percent", 50.0)
        
        return PerformanceEstimate(
            batch_size=batch_size,
            estimated_latency_ms=latency,
            estimated_throughput_samples_per_sec=throughput,
            estimated_memory_mb=memory,
            estimated_gpu_util_percent=utilization,
        )
