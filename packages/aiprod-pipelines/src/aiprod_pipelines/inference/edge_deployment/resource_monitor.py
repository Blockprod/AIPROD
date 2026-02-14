"""
Device Resource Monitoring

Monitors CPU, memory, battery, thermal, and network resource usage on edge devices.
Enables adaptive inference based on available resources.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum
import time


class ResourceLevel(Enum):
    """Resource availability levels."""
    CRITICAL = "critical"  # < 20%
    LOW = "low"  # 20-40%
    MODERATE = "moderate"  # 40-70%
    HIGH = "high"  # 70-90%
    EXCELLENT = "excellent"  # > 90%


@dataclass
class ResourceSnapshot:
    """Snapshot of device resource state."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    battery_percent: Optional[float] = None
    temperature_celsius: Optional[float] = None
    network_bandwidth_mbps: Optional[float] = None
    storage_percent: Optional[float] = None


@dataclass
class ResourceThresholds:
    """Thresholds for resource warnings."""
    cpu_high_threshold: float = 80.0
    memory_high_threshold: float = 85.0
    battery_low_threshold: float = 20.0
    temperature_high_threshold: float = 45.0
    network_low_threshold: float = 5.0


class DeviceResourceMonitor:
    """Monitors device resources."""
    
    def __init__(self):
        self.snapshots: List[ResourceSnapshot] = []
        self.thresholds = ResourceThresholds()
        self.max_history = 100
    
    def collect_snapshot(self) -> ResourceSnapshot:
        """Collect current resource snapshot."""
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=self._get_cpu_percent(),
            memory_percent=self._get_memory_percent(),
            battery_percent=self._get_battery_percent(),
            temperature_celsius=self._get_temperature(),
            storage_percent=self._get_storage_percent(),
        )
        
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_history:
            self.snapshots.pop(0)
        
        return snapshot
    
    def _get_cpu_percent(self) -> float:
        """Get CPU utilization percentage."""
        # Simulated
        return 45.0
    
    def _get_memory_percent(self) -> float:
        """Get memory utilization percentage."""
        # Simulated
        return 60.0
    
    def _get_battery_percent(self) -> float:
        """Get battery percentage."""
        # Simulated
        return 75.0
    
    def _get_temperature(self) -> float:
        """Get device temperature in Celsius."""
        # Simulated
        return 35.0
    
    def _get_storage_percent(self) -> float:
        """Get storage utilization percentage."""
        # Simulated
        return 50.0
    
    def get_resource_levels(self) -> Dict[str, ResourceLevel]:
        """Get current resource levels."""
        if not self.snapshots:
            self.collect_snapshot()
        
        latest = self.snapshots[-1]
        
        levels = {}
        
        # CPU level
        if latest.cpu_percent > self.thresholds.cpu_high_threshold:
            levels["cpu"] = ResourceLevel.CRITICAL
        elif latest.cpu_percent > 60:
            levels["cpu"] = ResourceLevel.MODERATE
        else:
            levels["cpu"] = ResourceLevel.HIGH
        
        # Memory level
        if latest.memory_percent > self.thresholds.memory_high_threshold:
            levels["memory"] = ResourceLevel.CRITICAL
        elif latest.memory_percent > 70:
            levels["memory"] = ResourceLevel.MODERATE
        else:
            levels["memory"] = ResourceLevel.HIGH
        
        # Battery level
        if latest.battery_percent:
            if latest.battery_percent < self.thresholds.battery_low_threshold:
                levels["battery"] = ResourceLevel.CRITICAL
            elif latest.battery_percent < 50:
                levels["battery"] = ResourceLevel.LOW
            else:
                levels["battery"] = ResourceLevel.HIGH
        
        return levels
    
    def get_average_resources(self, window_size: int = 10) -> Dict[str, float]:
        """Get average resource usage over window."""
        if not self.snapshots:
            return {}
        
        recent = self.snapshots[-window_size:]
        avg_cpu = sum(s.cpu_percent for s in recent) / len(recent)
        avg_memory = sum(s.memory_percent for s in recent) / len(recent)
        
        return {
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
        }
    
    def predict_resource_critical(self, time_window_seconds: int = 60) -> bool:
        """Predict if resources will become critical soon."""
        if not self.snapshots or len(self.snapshots) < 2:
            return False
        
        recent = [s for s in self.snapshots if time.time() - s.timestamp < time_window_seconds]
        if len(recent) < 2:
            return False
        
        # Check trend
        cpu_trend = recent[-1].cpu_percent - recent[0].cpu_percent
        memory_trend = recent[-1].memory_percent - recent[0].memory_percent
        
        # If strong upward trend
        return cpu_trend > 30 or memory_trend > 30


class AdaptiveInferenceController:
    """Controls inference based on available resources."""
    
    def __init__(self, monitor: DeviceResourceMonitor):
        self.monitor = monitor
    
    def get_recommended_batch_size(self) -> int:
        """Recommend batch size based on resources."""
        levels = self.monitor.get_resource_levels()
        
        # Conservative based on worst resource
        worst_level = max(levels.values(), key=lambda x: x.value)
        
        if worst_level == ResourceLevel.CRITICAL:
            return 1
        elif worst_level == ResourceLevel.LOW:
            return 2
        elif worst_level == ResourceLevel.MODERATE:
            return 4
        else:
            return 8
    
    def should_defer_inference(self) -> bool:
        """Determine if inference should be deferred."""
        if self.monitor.predict_resource_critical():
            return True
        
        levels = self.monitor.get_resource_levels()
        
        # Defer if CPU or memory critical
        return levels.get("cpu") == ResourceLevel.CRITICAL or levels.get("memory") == ResourceLevel.CRITICAL
    
    def get_maximum_model_size_mb(self) -> float:
        """Get maximum model size that fits in available memory."""
        avg_resources = self.monitor.get_average_resources()
        avg_memory_percent = avg_resources.get("avg_memory_percent", 60.0)
        
        # Assume 512MB total memory, use 60% for models
        total_memory_mb = 512
        safe_percentage = 60.0
        
        available = total_memory_mb * (safe_percentage - avg_memory_percent) / 100
        return max(10, available)  # Min 10MB
