"""
Metrics Collector - Advanced monitoring and analytics
Tracks performance, errors, usage patterns, and system health
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
from src.utils.monitoring import logger


class MetricType(str, Enum):
    """Types of metrics collected"""
    REQUEST_DURATION = "request_duration"
    REQUEST_COUNT = "request_count"
    ERROR_COUNT = "error_count"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    RESPONSE_SIZE = "response_size"
    DATABASE_QUERY_TIME = "db_query_time"
    CACHE_HIT_RATE = "cache_hit_rate"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ENDPOINT_LATENCY = "endpoint_latency"
    ACTIVE_CONNECTIONS = "active_connections"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Single metric measurement"""
    name: str
    value: float
    unit: str
    timestamp: str
    labels: Dict[str, str]
    metric_type: MetricType


@dataclass
class Alert:
    """Alert for monitored condition"""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    metric: str
    threshold: float
    current_value: float
    timestamp: str
    resolved: bool = False


class MetricsCollector:
    """
    Collects and aggregates metrics for advanced monitoring.
    
    Features:
    - Per-endpoint request tracking
    - Error rate calculation
    - Performance metrics (latency, throughput)
    - Anomaly detection
    - Alert generation
    - Time-series data aggregation
    """
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector
        
        Args:
            retention_hours: How long to keep metrics data
        """
        self.retention_hours = retention_hours
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.alerts: Dict[str, Alert] = {}
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "request_count": 0,
            "error_count": 0,
            "total_duration": 0,
            "min_duration": float('inf'),
            "max_duration": 0,
            "total_size": 0,
            "method": "GET",
        })
        self.start_time = time.time()
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        duration_ms: float,
        status_code: int,
        response_size: int = 0,
        error: Optional[str] = None
    ):
        """
        Record a request metric
        
        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc)
            duration_ms: Request duration in milliseconds
            status_code: HTTP response status code
            response_size: Response body size in bytes
            error: Error message if failed
        """
        timestamp = datetime.utcnow().isoformat()
        key = f"{method} {endpoint}"
        
        # Update endpoint statistics
        stats = self.endpoint_stats[key]
        stats["request_count"] += 1
        stats["method"] = method
        stats["total_duration"] += duration_ms
        stats["min_duration"] = min(stats["min_duration"], duration_ms)
        stats["max_duration"] = max(stats["max_duration"], duration_ms)
        
        if response_size > 0:
            stats["total_size"] += response_size
        
        if status_code >= 400:
            stats["error_count"] += 1
        
        # Record latency metric
        metric = Metric(
            name="endpoint_latency",
            value=duration_ms,
            unit="ms",
            timestamp=timestamp,
            labels={"endpoint": endpoint, "method": method, "status": str(status_code)},
            metric_type=MetricType.ENDPOINT_LATENCY
        )
        self.metrics["endpoint_latency"].append(metric)
        
        # Check thresholds for alerts
        if duration_ms > 1000:  # > 1 second
            self._create_alert(
                metric_name="endpoint_latency",
                severity=AlertSeverity.WARNING,
                title=f"High latency on {key}",
                message=f"Request took {duration_ms}ms (threshold: 1000ms)",
                threshold=1000,
                current_value=duration_ms,
                timestamp=timestamp
            )
        
        if error:
            error_metric = Metric(
                name="error_count",
                value=1,
                unit="count",
                timestamp=timestamp,
                labels={"endpoint": endpoint, "method": method, "error": error},
                metric_type=MetricType.ERROR_COUNT
            )
            self.metrics["error_count"].append(error_metric)
        
        logger.debug(f"Recorded metric: {key} {status_code} {duration_ms}ms")
    
    def record_cache_hit(self, hit: bool):
        """Record cache hit/miss"""
        timestamp = datetime.utcnow().isoformat()
        metric = Metric(
            name="cache_hit",
            value=1 if hit else 0,
            unit="count",
            timestamp=timestamp,
            labels={"result": "hit" if hit else "miss"},
            metric_type=MetricType.CACHE_HIT_RATE
        )
        self.metrics["cache_hit"].append(metric)
    
    def record_database_query(self, query_time_ms: float, table: str = "unknown"):
        """Record database query performance"""
        timestamp = datetime.utcnow().isoformat()
        metric = Metric(
            name="db_query_time",
            value=query_time_ms,
            unit="ms",
            timestamp=timestamp,
            labels={"table": table},
            metric_type=MetricType.DATABASE_QUERY_TIME
        )
        self.metrics["db_query_time"].append(metric)
        
        if query_time_ms > 500:
            self._create_alert(
                metric_name="db_query_time",
                severity=AlertSeverity.WARNING,
                title=f"Slow database query on {table}",
                message=f"Query took {query_time_ms}ms",
                threshold=500,
                current_value=query_time_ms,
                timestamp=timestamp
            )
    
    def get_endpoint_stats(self, endpoint: Optional[str] = None) -> Dict:
        """
        Get statistics for endpoints
        
        Args:
            endpoint: Specific endpoint or None for all
            
        Returns:
            Statistics dictionary
        """
        if endpoint:
            stats = self.endpoint_stats.get(endpoint, {})
            if stats and stats["request_count"] > 0:
                return {
                    **stats,
                    "avg_duration": stats["total_duration"] / stats["request_count"],
                    "error_rate": stats["error_count"] / stats["request_count"] if stats["request_count"] > 0 else 0,
                    "avg_size": stats["total_size"] / stats["request_count"] if stats["request_count"] > 0 else 0,
                }
            return stats
        
        # Return all endpoints
        result = {}
        for endpoint, stats in self.endpoint_stats.items():
            if stats["request_count"] > 0:
                result[endpoint] = {
                    **stats,
                    "avg_duration": stats["total_duration"] / stats["request_count"],
                    "error_rate": stats["error_count"] / stats["request_count"],
                    "avg_size": stats["total_size"] / stats["request_count"],
                }
        return result
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        total_requests = sum(s["request_count"] for s in self.endpoint_stats.values())
        total_errors = sum(s["error_count"] for s in self.endpoint_stats.values())
        error_rate = total_errors / total_requests if total_requests > 0 else 0
        
        # Determine health based on error rate
        if error_rate > 0.1:  # > 10% errors
            health = "critical"
        elif error_rate > 0.05:  # > 5% errors
            health = "warning"
        else:
            health = "healthy"
        
        uptime_seconds = time.time() - self.start_time
        
        return {
            "status": health,
            "uptime_seconds": uptime_seconds,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": round(error_rate * 100, 2),
            "endpoints_count": len(self.endpoint_stats),
            "active_alerts": len([a for a in self.alerts.values() if not a.resolved]),
        }
    
    def get_top_endpoints(self, limit: int = 10) -> List[Dict]:
        """Get endpoints with highest error rates or latencies"""
        endpoints = []
        
        for endpoint, stats in self.endpoint_stats.items():
            if stats["request_count"] == 0:
                continue
            
            endpoints.append({
                "endpoint": endpoint,
                "requests": stats["request_count"],
                "errors": stats["error_count"],
                "error_rate": round(stats["error_count"] / stats["request_count"] * 100, 2),
                "avg_latency": round(stats["total_duration"] / stats["request_count"], 2),
                "max_latency": round(stats["max_duration"], 2),
            })
        
        # Sort by error rate
        endpoints.sort(key=lambda x: x["error_rate"], reverse=True)
        return endpoints[:limit]
    
    def get_alerts(self, unresolved_only: bool = True) -> List[Dict]:
        """Get alerts"""
        alerts = []
        for alert in self.alerts.values():
            if unresolved_only and alert.resolved:
                continue
            alerts.append(asdict(alert))
        
        # Sort by timestamp descending
        alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        return alerts
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            logger.info(f"Resolved alert {alert_id}")
    
    def _create_alert(
        self,
        metric_name: str,
        severity: AlertSeverity,
        title: str,
        message: str,
        threshold: float,
        current_value: float,
        timestamp: str
    ):
        """Create an alert"""
        alert_id = f"{metric_name}_{int(time.time() * 1000)}"
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            metric=metric_name,
            threshold=threshold,
            current_value=current_value,
            timestamp=timestamp,
            resolved=False
        )
        
        self.alerts[alert_id] = alert
        logger.warning(f"Alert created: {title}")
    
    def cleanup_old_metrics(self):
        """Remove metrics older than retention period"""
        cutoff_time = (datetime.utcnow() - timedelta(hours=self.retention_hours)).isoformat()
        
        for metric_name in self.metrics:
            before_count = len(self.metrics[metric_name])
            self.metrics[metric_name] = [
                m for m in self.metrics[metric_name]
                if m.timestamp > cutoff_time
            ]
            removed = before_count - len(self.metrics[metric_name])
            if removed > 0:
                logger.debug(f"Cleaned up {removed} old {metric_name} metrics")


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector(retention_hours: int = 24) -> MetricsCollector:
    """Get or create singleton metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(retention_hours)
    return _metrics_collector
