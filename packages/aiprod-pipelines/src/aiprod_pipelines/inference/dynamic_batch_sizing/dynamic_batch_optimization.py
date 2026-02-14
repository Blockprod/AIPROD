"""
Dynamic Batch Optimization Engine

Main orchestrator for dynamic batch sizing across all phases of inference.
Combines profiling, prediction, caching, and adaptive strategies.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List, Tuple, Any
from enum import Enum
import threading
import time


class DynamicBatchState(Enum):
    """State of dynamic batch sizing engine."""
    INITIALIZING = "initializing"
    PROFILING = "profiling"
    STABLE = "stable"
    ADAPTING = "adapting"
    ERROR = "error"


@dataclass
class DynamicBatchConfig:
    """Complete configuration for dynamic batch sizing."""
    # Profiling settings
    enable_profiling: bool = True
    profiling_samples: int = 100
    profiling_warmup: int = 10
    
    # Adaptation settings
    enable_adaptation: bool = True
    adaptation_interval: int = 1000
    min_adaptation_confidence: float = 0.8
    
    # Caching and persistence
    enable_caching: bool = True
    cache_max_entries: int = 1000
    cache_persistence_path: Optional[str] = None
    
    # Safety and constraints
    safety_margin_memory: float = 0.85  # Only use 85% of available memory
    max_allowed_batch_size: int = 512
    min_allowed_batch_size: int = 1
    prefer_stable_over_optimal: bool = True
    
    # Monitoring and statistics
    enable_monitoring: bool = True
    monitor_window_size: int = 100
    
    # Multi-threading
    enable_background_optimization: bool = True
    optimization_thread_priority: str = "low"  # "low", "normal", "high"


@dataclass
class DynamicBatchMetrics:
    """Metrics tracking dynamic batch optimization."""
    current_batch_size: int
    state: DynamicBatchState
    iterations: int = 0
    adaptations_made: int = 0
    cache_hits: int = 0
    profiling_complete: bool = False
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    avg_throughput: float = 0.0
    avg_memory_mb: float = 0.0
    gpu_utilization_percent: float = 0.0
    
    # Quality metrics
    predictions_accuracy: float = 0.0  # 0-1, how accurate are latency/memory predictions
    adaptation_convergence: float = 0.0  # 0-1, how close to optimal
    

class DynamicBatchSizer:
    """Main dynamic batch sizing engine."""
    
    def __init__(self, config: DynamicBatchConfig):
        self.config = config
        self.state = DynamicBatchState.INITIALIZING
        self.current_batch_size = 32
        self.metrics = DynamicBatchMetrics(
            current_batch_size=32,
            state=DynamicBatchState.INITIALIZING,
        )
        self.lock = threading.Lock()
        
        # Component instances (lazy loaded)
        self._batch_cache = None
        self._adaptive_batcher = None
        self._performance_monitor = None
        self._background_thread = None
        
    def initialize(
        self,
        model_name: str,
        device_type: str,
        memory_available_mb: float,
    ) -> None:
        """Initialize batch sizer for specific model and device."""
        self.model_name = model_name
        self.device_type = device_type
        self.memory_available_mb = memory_available_mb
        
        # Safety check
        if memory_available_mb <= 0:
            raise ValueError("Invalid memory budget")
        
        # Try to load from cache first
        if self.config.enable_caching:
            cached_batch = self._try_load_from_cache()
            if cached_batch:
                self.current_batch_size = cached_batch
                self.metrics.current_batch_size = cached_batch
                self.state = DynamicBatchState.STABLE
                return
        
        # Start profiling state
        self.state = DynamicBatchState.PROFILING
        self.metrics.state = DynamicBatchState.PROFILING
    
    def suggest_batch_size(self) -> int:
        """Get current recommended batch size."""
        with self.lock:
            return self.current_batch_size
    
    def record_batch_performance(
        self,
        batch_size: int,
        latency_ms: float,
        memory_mb: float,
        throughput_samples_per_sec: float,
        gpu_utilization_percent: float = 0.0,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Record performance metrics from batch execution."""
        with self.lock:
            self.metrics.iterations += 1
            
            if not success:
                self.state = DynamicBatchState.ERROR
                self.metrics.state = DynamicBatchState.ERROR
                return
            
            # Update metrics
            self.metrics.avg_latency_ms = latency_ms
            self.metrics.avg_throughput = throughput_samples_per_sec
            self.metrics.avg_memory_mb = memory_mb
            self.metrics.gpu_utilization_percent = gpu_utilization_percent
            
            # Check if should adapt
            if self._should_adapt():
                self._adapt_batch_size()
    
    def _should_adapt(self) -> bool:
        """Determine if batch size should be adapted."""
        if not self.config.enable_adaptation:
            return False
        
        if self.state == DynamicBatchState.STABLE:
            return False
        
        if self.metrics.iterations % self.config.adaptation_interval == 0:
            return True
        
        return False
    
    def _adapt_batch_size(self) -> None:
        """Adapt batch size based on performance."""
        memory_utilization = self.metrics.avg_memory_mb / self.memory_available_mb
        
        if memory_utilization > 0.95:
            # Too close to limit, reduce
            self.current_batch_size = max(
                int(self.current_batch_size * 0.9),
                self.config.min_allowed_batch_size,
            )
        elif memory_utilization < 0.5 and self.metrics.gpu_utilization_percent < 80:
            # Underutilized, can increase
            self.current_batch_size = min(
                int(self.current_batch_size * 1.1),
                self.config.max_allowed_batch_size,
            )
        
        # Ensure alignment
        if self.current_batch_size > 0:
            self.current_batch_size = max(1, self.current_batch_size)
        
        self.metrics.current_batch_size = self.current_batch_size
        self.metrics.adaptations_made += 1
    
    def _try_load_from_cache(self) -> Optional[int]:
        """Try to load pre-computed batch size from cache."""
        # Implementation depends on cache module
        return None
    
    def get_metrics(self) -> DynamicBatchMetrics:
        """Get current metrics snapshot."""
        with self.lock:
            return DynamicBatchMetrics(
                current_batch_size=self.metrics.current_batch_size,
                state=self.metrics.state,
                iterations=self.metrics.iterations,
                adaptations_made=self.metrics.adaptations_made,
                cache_hits=self.metrics.cache_hits,
                profiling_complete=self.metrics.profiling_complete,
                avg_latency_ms=self.metrics.avg_latency_ms,
                avg_throughput=self.metrics.avg_throughput,
                avg_memory_mb=self.metrics.avg_memory_mb,
                gpu_utilization_percent=self.metrics.gpu_utilization_percent,
                predictions_accuracy=self.metrics.predictions_accuracy,
                adaptation_convergence=self.metrics.adaptation_convergence,
            )
    
    def shutdown(self) -> None:
        """Clean shutdown of batch sizer."""
        with self.lock:
            self.state = DynamicBatchState.INITIALIZING
            if self._background_thread:
                self._background_thread.join(timeout=5.0)


class DynamicBatchNode:
    """Graph node wrapper for dynamic batch sizing."""
    
    def __init__(self, name: str, batch_sizer: DynamicBatchSizer):
        self.name = name
        self.batch_sizer = batch_sizer
        self.input_type = "any"
        self.output_type = "any"
        
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute batch sizing optimization."""
        batch_size = self.batch_sizer.suggest_batch_size()
        
        return {
            "batch_size": batch_size,
            "metrics": self.batch_sizer.get_metrics(),
        }


class MultiModelBatchOptimizer:
    """Manages batch sizing for multiple models simultaneously."""
    
    def __init__(self, config: DynamicBatchConfig):
        self.config = config
        self.model_sizers: Dict[str, DynamicBatchSizer] = {}
        self.lock = threading.Lock()
    
    def register_model(
        self,
        model_name: str,
        device_type: str,
        memory_available_mb: float,
    ) -> DynamicBatchSizer:
        """Register new model for optimization."""
        with self.lock:
            if model_name in self.model_sizers:
                return self.model_sizers[model_name]
            
            sizer = DynamicBatchSizer(self.config)
            sizer.initialize(model_name, device_type, memory_available_mb)
            self.model_sizers[model_name] = sizer
            return sizer
    
    def get_sizer(self, model_name: str) -> Optional[DynamicBatchSizer]:
        """Get batch sizer for model."""
        with self.lock:
            return self.model_sizers.get(model_name)
    
    def unregister_model(self, model_name: str) -> None:
        """Unregister model."""
        with self.lock:
            if model_name in self.model_sizers:
                self.model_sizers[model_name].shutdown()
                del self.model_sizers[model_name]
    
    def get_all_metrics(self) -> Dict[str, DynamicBatchMetrics]:
        """Get metrics for all registered models."""
        with self.lock:
            return {
                name: sizer.get_metrics()
                for name, sizer in self.model_sizers.items()
            }
