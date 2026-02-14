"""
Monitoring and Analytics for Multi-Tenant SaaS.

Provides real-time monitoring, anomaly detection, and analytics
for SaaS platform health and usage.

Core Classes:
  - TenantMetrics: Per-tenant performance metrics
  - MetricsCollector: Metric aggregation
  - AnomalyDetector: Anomaly detection
  - AnalyticsCollector: Event analytics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict
import statistics


class HealthStatus(str, Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class MetricSnapshot:
    """Single metric snapshot."""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class TenantMetrics:
    """Tenant-level performance metrics."""
    tenant_id: str
    
    # API metrics
    total_api_calls: int = 0
    successful_api_calls: int = 0
    failed_api_calls: int = 0
    avg_api_response_time_ms: float = 0.0
    
    # Job metrics
    total_jobs_submitted: int = 0
    successful_jobs: int = 0
    failed_jobs: int = 0
    avg_job_duration_seconds: float = 0.0
    
    # Resource usage
    storage_used_gb: float = 0.0
    compute_hours: float = 0.0
    network_gb: float = 0.0
    
    # Cost metrics
    total_cost_current_month: float = 0.0
    estimated_cost_next_month: float = 0.0
    
    # Error rate
    error_rate_percentage: float = 0.0
    p99_response_time_ms: float = 0.0
    
    collected_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "total_api_calls": self.total_api_calls,
            "successful_api_calls": self.successful_api_calls,
            "failed_api_calls": self.failed_api_calls,
            "avg_api_response_time_ms": self.avg_api_response_time_ms,
            "total_jobs_submitted": self.total_jobs_submitted,
            "successful_jobs": self.successful_jobs,
            "failed_jobs": self.failed_jobs,
            "avg_job_duration_seconds": self.avg_job_duration_seconds,
            "error_rate_percentage": self.error_rate_percentage,
            "total_cost_current_month": self.total_cost_current_month,
            "collected_at": self.collected_at.isoformat(),
        }


class MetricsCollector:
    """Collects and aggregates metrics."""
    
    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector."""
        self.retention_hours = retention_hours
        self._metrics: List[MetricSnapshot] = []
        self._tenant_metrics: Dict[str, List[TenantMetrics]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric."""
        snapshot = MetricSnapshot(
            timestamp=datetime.utcnow(),
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags or {},
        )
        
        with self._lock:
            self._metrics.append(snapshot)
            self._cleanup_old_metrics()
    
    def record_tenant_metrics(self, metrics: TenantMetrics) -> None:
        """Record tenant metrics snapshot."""
        with self._lock:
            self._tenant_metrics[metrics.tenant_id].append(metrics)
            
            # Keep only recent snapshots
            if len(self._tenant_metrics[metrics.tenant_id]) > 1000:
                self._tenant_metrics[metrics.tenant_id] = \
                    self._tenant_metrics[metrics.tenant_id][-1000:]
    
    def get_tenant_metrics_history(
        self,
        tenant_id: str,
        hours: int = 24,
    ) -> List[TenantMetrics]:
        """Get tenant metrics history."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            return [
                m for m in self._tenant_metrics.get(tenant_id, [])
                if m.collected_at > cutoff_time
            ]
    
    def get_metric_statistics(
        self,
        metric_name: str,
        hours: int = 24,
    ) -> Optional[Dict[str, float]]:
        """Get statistics for metric over time period."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            values = [
                m.value for m in self._metrics
                if m.metric_name == metric_name and m.timestamp > cutoff_time
            ]
        
        if not values:
            return None
        
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }
    
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics outside retention window."""
        cutoff_time = datetime.utcnow() - timedelta(hours=self.retention_hours)
        self._metrics = [m for m in self._metrics if m.timestamp > cutoff_time]
    
    @staticmethod
    def _percentile(values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_vals = sorted(values)
        index = (percentile / 100) * len(sorted_vals)
        if index == int(index):
            return sorted_vals[int(index) - 1]
        return sorted_vals[int(index)]


class AnomalyDetector:
    """Detects anomalies in metrics."""
    
    def __init__(self, std_dev_threshold: float = 3.0):
        """Initialize anomaly detector."""
        self.std_dev_threshold = std_dev_threshold
        self._baseline_stats: Dict[str, Dict[str, float]] = {}
        self._lock = threading.RLock()
    
    def establish_baseline(
        self,
        metric_name: str,
        values: List[float],
    ) -> None:
        """Establish baseline statistics for metric."""
        if not values:
            return
        
        with self._lock:
            self._baseline_stats[metric_name] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            }
    
    def detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """Detect if value is anomalous."""
        with self._lock:
            if metric_name not in self._baseline_stats:
                return False, 0.0
            
            stats = self._baseline_stats[metric_name]
            if stats["stdev"] == 0:
                return False, 0.0
            
            # Calculate z-score
            z_score = abs((value - stats["mean"]) / stats["stdev"])
            is_anomalous = z_score > self.std_dev_threshold
            
            return is_anomalous, z_score
    
    def detect_trend_anomaly(
        self,
        metric_name: str,
        recent_values: List[float],
    ) -> Tuple[bool, str]:
        """Detect anomalous trend."""
        if len(recent_values) < 2:
            return False, ""
        
        # Calculate trend
        mean_first_half = statistics.mean(recent_values[:len(recent_values)//2])
        mean_second_half = statistics.mean(recent_values[len(recent_values)//2:])
        
        percent_change = ((mean_second_half - mean_first_half) / mean_first_half * 100) \
            if mean_first_half != 0 else 0
        
        if abs(percent_change) > 50:  # 50% change threshold
            trend = "increasing" if percent_change > 0 else "decreasing"
            return True, f"{metric_name} {trend} by {abs(percent_change):.1f}%"
        
        return False, ""


class AnalyticsCollector:
    """Collects event-based analytics."""
    
    def __init__(self):
        """Initialize analytics collector."""
        self._events: List[Dict[str, Any]] = []
        self._event_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    def track_event(
        self,
        event_name: str,
        tenant_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track an event."""
        event = {
            "name": event_name,
            "tenant_id": tenant_id,
            "timestamp": datetime.utcnow(),
            "properties": properties or {},
        }
        
        with self._lock:
            self._events.append(event)
            self._event_counts[event_name] += 1
            
            # Keep only recent events
            if len(self._events) > 100000:
                self._events = self._events[-100000:]
    
    def get_event_count(self, event_name: str) -> int:
        """Get total count of event."""
        with self._lock:
            return self._event_counts.get(event_name, 0)
    
    def get_events(
        self,
        event_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get events matching criteria."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            events = [
                e for e in self._events
                if e["timestamp"] > cutoff_time
            ]
            
            if event_name:
                events = [e for e in events if e["name"] == event_name]
            if tenant_id:
                events = [e for e in events if e["tenant_id"] == tenant_id]
            
            return events
    
    def get_tenant_analytics(self, tenant_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get analytics for tenant."""
        events = self.get_events(tenant_id=tenant_id, hours=hours)
        
        event_summary = defaultdict(int)
        for event in events:
            event_summary[event["name"]] += 1
        
        return {
            "tenant_id": tenant_id,
            "period_hours": hours,
            "total_events": len(events),
            "event_summary": dict(event_summary),
        }


class HealthMonitor:
    """Monitors overall platform health."""
    
    def __init__(self, metrics_collector: MetricsCollector, anomaly_detector: AnomalyDetector):
        """Initialize health monitor."""
        self.metrics_collector = metrics_collector
        self.anomaly_detector = anomaly_detector
    
    def get_platform_health(self) -> Dict[str, Any]:
        """Get overall platform health."""
        # Check key metrics for anomalies
        error_rate_stats = self.metrics_collector.get_metric_statistics(
            "error_rate",
            hours=1,
        )
        
        response_time_stats = self.metrics_collector.get_metric_statistics(
            "response_time_ms",
            hours=1,
        )
        
        health_status = HealthStatus.HEALTHY
        issues = []
        
        if error_rate_stats and error_rate_stats.get("mean", 0) > 5:
            health_status = HealthStatus.DEGRADED
            issues.append(f"Error rate elevated: {error_rate_stats['mean']:.2f}%")
        
        if response_time_stats and response_time_stats.get("p99", 0) > 5000:
            health_status = HealthStatus.DEGRADED
            issues.append(f"Response time high: {response_time_stats['p99']:.0f}ms")
        
        if error_rate_stats and error_rate_stats.get("mean", 0) > 20:
            health_status = HealthStatus.CRITICAL
        
        return {
            "status": health_status.value,
            "issues": issues,
            "error_rate": error_rate_stats,
            "response_time": response_time_stats,
        }
    
    def get_tenant_health(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant-specific health."""
        metrics_history = self.metrics_collector.get_tenant_metrics_history(tenant_id, hours=24)
        
        if not metrics_history:
            return {"status": "no_data"}
        
        latest_metrics = metrics_history[-1]
        health_status = HealthStatus.HEALTHY
        
        if latest_metrics.error_rate_percentage > 5:
            health_status = HealthStatus.DEGRADED
        if latest_metrics.error_rate_percentage > 20:
            health_status = HealthStatus.CRITICAL
        
        return {
            "tenant_id": tenant_id,
            "status": health_status.value,
            "latest_metrics": latest_metrics.to_dict(),
            "metrics_history_count": len(metrics_history),
        }
