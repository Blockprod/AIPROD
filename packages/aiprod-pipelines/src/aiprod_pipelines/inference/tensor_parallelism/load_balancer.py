"""
Distributed Load Balancing and Work Distribution

Implements dynamic load balancing across distributed training devices,
monitoring workload imbalance, and redistributing work.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
from collections import deque


class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    STATIC = "static"  # Fixed assignment
    DYNAMIC = "dynamic"  # Adjust based on metrics
    WORK_STEALING = "work_stealing"  # Busy devices steal from idle
    GREEDY = "greedy"  # Assign to least loaded device


@dataclass
class DeviceWorkload:
    """Current workload metrics for a device"""
    device_id: int
    rank: int
    num_samples_processing: int = 0
    compute_utilization: float = 0.0  # 0.0-1.0
    memory_utilization: float = 0.0  # 0.0-1.0
    queue_depth: int = 0
    last_updated: float = field(default_factory=time.time)
    
    @property
    def imbalance_score(self) -> float:
        """Higher = more imbalanced (idle)"""
        return (1.0 - self.compute_utilization) * self.memory_utilization
    
    @property
    def is_overloaded(self) -> bool:
        """Check if device is overloaded"""
        return self.compute_utilization > 0.95 or self.queue_depth > 10
    
    @property
    def is_idle(self) -> bool:
        """Check if device is idle"""
        return self.compute_utilization < 0.1 and self.queue_depth == 0


@dataclass
class LoadMetrics:
    """Aggregated load metrics across all devices"""
    avg_utilization: float
    max_utilization: float
    min_utilization: float
    utilization_variance: float
    imbalance_ratio: float  # max / min
    total_queue_depth: int
    idle_device_count: int
    overloaded_device_count: int


class LoadBalancer:
    """Monitors and balances workload across devices"""
    
    def __init__(self, num_devices: int, history_size: int = 100):
        self.num_devices = num_devices
        self.device_workloads: Dict[int, DeviceWorkload] = {
            i: DeviceWorkload(device_id=i, rank=i) for i in range(num_devices)
        }
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.rebalance_decisions: List[Dict[str, Any]] = []
    
    def update_device_metrics(self, device_id: int, compute_util: float,
                             memory_util: float, queue_depth: int):
        """Update metrics for a device"""
        if device_id not in self.device_workloads:
            return
        
        workload = self.device_workloads[device_id]
        workload.compute_utilization = min(1.0, max(0.0, compute_util))
        workload.memory_utilization = min(1.0, max(0.0, memory_util))
        workload.queue_depth = queue_depth
        workload.last_updated = time.time()
    
    def get_load_metrics(self) -> LoadMetrics:
        """Compute load metrics"""
        utilizations = [w.compute_utilization for w in self.device_workloads.values()]
        
        avg_util = sum(utilizations) / len(utilizations) if utilizations else 0.0
        max_util = max(utilizations) if utilizations else 0.0
        min_util = min(utilizations) if utilizations else 0.0
        
        # Compute variance
        variance = sum((u - avg_util) ** 2 for u in utilizations) / len(utilizations) if utilizations else 0.0
        
        imbalance_ratio = max_util / max(min_util, 0.01)  # Avoid division by zero
        
        idle_count = sum(1 for w in self.device_workloads.values() if w.is_idle)
        overloaded_count = sum(1 for w in self.device_workloads.values() if w.is_overloaded)
        total_queue = sum(w.queue_depth for w in self.device_workloads.values())
        
        metrics = LoadMetrics(
            avg_utilization=avg_util,
            max_utilization=max_util,
            min_utilization=min_util,
            utilization_variance=variance,
            imbalance_ratio=imbalance_ratio,
            total_queue_depth=total_queue,
            idle_device_count=idle_count,
            overloaded_device_count=overloaded_count
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def should_rebalance(self) -> bool:
        """Determine if rebalancing is needed"""
        if not self.metrics_history:
            return False
        
        latest = self.metrics_history[-1]
        
        # Rebalance if imbalance ratio is high or devices are idle/overloaded
        return (latest.imbalance_ratio > 1.5 or
                latest.idle_device_count > 0 or
                latest.overloaded_device_count > 0)
    
    def compute_rebalance_plan(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.GREEDY) -> Dict[int, List[int]]:
        """
        Compute rebalancing plan
        
        Returns:
            Mapping of source device -> list of batches to move to which target
        """
        rebalance_plan = {}
        
        if strategy == LoadBalancingStrategy.GREEDY:
            return self._greedy_rebalance()
        elif strategy == LoadBalancingStrategy.WORK_STEALING:
            return self._work_stealing_rebalance()
        else:
            return rebalance_plan
    
    def _greedy_rebalance(self) -> Dict[int, List[int]]:
        """Greedy rebalancing: assign to least loaded"""
        rebalance_plan = {}
        
        sorted_devices = sorted(
            self.device_workloads.values(),
            key=lambda w: w.compute_utilization,
            reverse=True
        )
        
        # Move batches from overloaded to underloaded
        for device in sorted_devices:
            if device.is_overloaded and device.queue_depth > 0:
                # Find least loaded device
                target = min(self.device_workloads.values(),
                           key=lambda w: w.compute_utilization)
                
                if target.device_id != device.device_id:
                    rebalance_plan[device.device_id] = [target.device_id]
        
        return rebalance_plan
    
    def _work_stealing_rebalance(self) -> Dict[int, List[int]]:
        """Work stealing: busy devices steal from idle"""
        rebalance_plan = {}
        
        idle_devices = [w for w in self.device_workloads.values() if w.is_idle]
        busy_devices = [w for w in self.device_workloads.values() if w.is_overloaded]
        
        for busy in busy_devices:
            targets = [d.device_id for d in idle_devices]
            if targets:
                rebalance_plan[busy.device_id] = targets
        
        return rebalance_plan
    
    def get_device_assignment(self, num_samples: int, 
                             strategy: LoadBalancingStrategy = LoadBalancingStrategy.GREEDY) -> Dict[int, int]:
        """
        Assign samples to devices
        
        Returns:
            Mapping of device_id -> number of samples to assign
        """
        if strategy == LoadBalancingStrategy.STATIC:
            # Equal distribution
            samples_per_device = num_samples // self.num_devices
            return {i: samples_per_device for i in range(self.num_devices)}
        
        elif strategy == LoadBalancingStrategy.GREEDY:
            # Assign to least loaded
            assignment = {i: 0 for i in range(self.num_devices)}
            
            for _ in range(num_samples):
                # Find least loaded device
                least_loaded = min(
                    self.device_workloads.values(),
                    key=lambda w: (w.compute_utilization, w.queue_depth)
                )
                assignment[least_loaded.device_id] += 1
                least_loaded.queue_depth += 1
            
            return assignment
        
        return {}
    
    def get_optimal_batch_size(self) -> Tuple[int, int]:
        """
        Get optimal batch size for balanced training
        
        Returns:
            (global_batch_size, micro_batch_per_device)
        """
        metrics = self.get_load_metrics()
        
        # Global batch size
        if metrics.max_utilization > 0.9:
            # Reduce batch size if devices are overloaded
            global_batch_size = max(32, int(256 * 0.8))
        else:
            global_batch_size = 256
        
        # Micro batch per device
        micro_batch = global_batch_size // self.num_devices
        
        return global_batch_size, micro_batch


class AdaptiveLoadBalancer(LoadBalancer):
    """Adaptive load balancer that adjusts strategy based on workload"""
    
    def __init__(self, num_devices: int):
        super().__init__(num_devices)
        self.strategy_history: List[LoadBalancingStrategy] = []
        self.performance_metrics: Dict[LoadBalancingStrategy, List[float]] = {
            strategy: [] for strategy in LoadBalancingStrategy
        }
    
    def select_best_strategy(self) -> LoadBalancingStrategy:
        """Select best strategy based on historical performance"""
        if not self.performance_metrics or not any(self.performance_metrics.values()):
            return LoadBalancingStrategy.GREEDY
        
        # Choose strategy with best (lowest) imbalance ratio history
        best_strategy = LoadBalancingStrategy.GREEDY
        best_score = float('inf')
        
        for strategy, scores in self.performance_metrics.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_strategy = strategy
        
        return best_strategy
    
    def record_strategy_performance(self, strategy: LoadBalancingStrategy, metrics: LoadMetrics):
        """Record performance of a strategy"""
        self.performance_metrics[strategy].append(metrics.imbalance_ratio)
        if len(self.performance_metrics[strategy]) > 100:
            self.performance_metrics[strategy].pop(0)
    
    def adaptive_rebalance(self) -> Dict[int, List[int]]:
        """Automatically select and apply best rebalancing strategy"""
        metrics = self.get_load_metrics()
        
        if not self.should_rebalance():
            return {}
        
        # Select best strategy
        strategy = self.select_best_strategy()
        self.strategy_history.append(strategy)
        
        # Compute rebalance plan
        rebalance_plan = self.compute_rebalance_plan(strategy)
        
        # Record performance
        self.record_strategy_performance(strategy, metrics)
        
        return rebalance_plan


@dataclass
class LoadBalancingConfig:
    """Configuration for load balancing"""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.GREEDY
    rebalance_interval_steps: int = 100
    enable_adaptive: bool = True
    imbalance_threshold: float = 1.5
    enable_work_stealing: bool = False
    
    def should_use_adaptive(self) -> bool:
        """Check if adaptive balancing should be used"""
        return self.enable_adaptive
