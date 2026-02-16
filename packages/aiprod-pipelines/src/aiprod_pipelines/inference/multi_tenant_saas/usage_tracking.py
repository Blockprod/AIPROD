"""
Usage Tracking and Metering for Multi-Tenant SaaS.

Tracks resource consumption, API usage, and metering for billing purposes.

Core Classes:
  - UsageEvent: Single tracked event
  - UsageEventLogger: Event collection and aggregation
  - MeteringEngine: Usage calculation and quota enforcement
  - UsageMetrics: Aggregated usage statistics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
import threading
from collections import defaultdict


class UsageEventType(str):
    """Usage event types."""
    VIDEO_GENERATION = "video_generation"
    MODEL_INFERENCE = "model_inference"
    API_CALL = "api_call"
    STORAGE = "storage"
    BATCH_JOB = "batch_job"
    TRAINING = "training"
    TRANSCODING = "transcoding"


@dataclass
class UsageEvent:
    """Single usage event."""
    event_id: str
    tenant_id: str
    user_id: str
    event_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_seconds: float = 0.0
    resource_consumed: float = 0.0  # Generic unit (MB, seconds, etc)
    resource_unit: str = "credits"
    cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "resource_consumed": self.resource_consumed,
            "resource_unit": self.resource_unit,
            "cost": self.cost,
        }


@dataclass
class UsageMetrics:
    """Aggregated usage metrics."""
    tenant_id: str
    period_start: datetime
    period_end: datetime
    
    total_api_calls: int = 0
    total_video_seconds_generated: float = 0.0
    total_storage_gb_hours: float = 0.0
    total_cost: float = 0.0
    
    event_counts: Dict[str, int] = field(default_factory=dict)
    hourly_usage: Dict[str, float] = field(default_factory=dict)
    daily_usage: Dict[str, List[float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_api_calls": self.total_api_calls,
            "total_video_seconds_generated": self.total_video_seconds_generated,
            "total_storage_gb_hours": self.total_storage_gb_hours,
            "total_cost": self.total_cost,
            "event_types": self.event_counts,
        }


class UsageEventLogger:
    """Logs and aggregates usage events."""
    
    def __init__(self, retention_days: int = 90):
        """Initialize event logger."""
        self.retention_days = retention_days
        self._events: List[UsageEvent] = []
        self._tenant_events: Dict[str, List[UsageEvent]] = defaultdict(list)
        self._lock = threading.RLock()
    
    def log_event(self, event: UsageEvent) -> str:
        """Log a usage event."""
        with self._lock:
            self._events.append(event)
            self._tenant_events[event.tenant_id].append(event)
        
        return event.event_id
    
    def get_events(
        self,
        tenant_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_type: Optional[str] = None,
    ) -> List[UsageEvent]:
        """Get events matching criteria."""
        with self._lock:
            events = self._tenant_events.get(tenant_id, [])
            
            # Filter by time range
            if start_time:
                events = [e for e in events if e.timestamp >= start_time]
            if end_time:
                events = [e for e in events if e.timestamp <= end_time]
            
            # Filter by type
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            return events
    
    def aggregate_usage(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> UsageMetrics:
        """Aggregate usage for period."""
        events = self.get_events(tenant_id, start_time, end_time)
        
        metrics = UsageMetrics(
            tenant_id=tenant_id,
            period_start=start_time,
            period_end=end_time,
        )
        
        for event in events:
            metrics.amount += event.resource_consumed
            metrics.total_cost += event.cost
            metrics.event_counts[event.event_type] = metrics.event_counts.get(event.event_type, 0) + 1
            
            if event.event_type == UsageEventType.API_CALL:
                metrics.total_api_calls += 1
            elif event.event_type == UsageEventType.VIDEO_GENERATION:
                metrics.total_video_seconds_generated += event.resource_consumed
            elif event.event_type == UsageEventType.STORAGE:
                metrics.total_storage_gb_hours += event.resource_consumed
        
        return metrics
    
    def cleanup_old_events(self) -> int:
        """Remove events older than retention period."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        with self._lock:
            original_count = len(self._events)
            self._events = [e for e in self._events if e.timestamp > cutoff_date]
            
            for tenant_id in self._tenant_events:
                self._tenant_events[tenant_id] = [
                    e for e in self._tenant_events[tenant_id]
                    if e.timestamp > cutoff_date
                ]
            
            removed = original_count - len(self._events)
            return removed


class MeteringEngine:
    """Tracks and enforces usage quotas."""
    
    def __init__(self):
        """Initialize metering engine."""
        self._usage: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._reset_times: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        self._lock = threading.RLock()
    
    def record_usage(
        self,
        tenant_id: str,
        metric: str,
        amount: float,
        reset_period_hours: int = 24,
    ) -> float:
        """Record usage and return current total."""
        with self._lock:
            key = f"{tenant_id}_{metric}"
            reset_key = (tenant_id, metric)
            
            # Check if we need to reset
            now = datetime.utcnow()
            if reset_key in self._reset_times:
                last_reset = self._reset_times[reset_key]
                if (now - last_reset).total_seconds() > reset_period_hours * 3600:
                    self._usage[tenant_id][metric] = 0.0
                    self._reset_times[reset_key] = now
            else:
                self._reset_times[reset_key] = now
            
            self._usage[tenant_id][metric] += amount
            return self._usage[tenant_id][metric]
    
    def get_current_usage(self, tenant_id: str, metric: str) -> float:
        """Get current usage for metric."""
        with self._lock:
            return self._usage[tenant_id].get(metric, 0.0)
    
    def check_quota(
        self,
        tenant_id: str,
        metric: str,
        quota_limit: float,
    ) -> Tuple[bool, float]:
        """Check if usage is within quota."""
        current = self.get_current_usage(tenant_id, metric)
        return current <= quota_limit, current
    
    def get_all_usage(self, tenant_id: str) -> Dict[str, float]:
        """Get all usage metrics for tenant."""
        with self._lock:
            return dict(self._usage.get(tenant_id, {}))
    
    def reset_usage(self, tenant_id: str, metric: Optional[str] = None) -> None:
        """Reset usage metrics."""
        with self._lock:
            if metric:
                self._usage[tenant_id][metric] = 0.0
                self._reset_times[(tenant_id, metric)] = datetime.utcnow()
            else:
                self._usage[tenant_id] = defaultdict(float)
                for key in list(self._reset_times.keys()):
                    if key[0] == tenant_id:
                        self._reset_times[key] = datetime.utcnow()


class UsageAggregator:
    """Aggregates usage across time periods and resource types."""
    
    def __init__(self, logger: UsageEventLogger):
        """Initialize aggregator."""
        self.logger = logger
    
    def get_daily_usage(self, tenant_id: str, date: datetime) -> Dict[str, Any]:
        """Get usage for a specific day."""
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        
        return self.logger.aggregate_usage(tenant_id, start, end).to_dict()
    
    def get_monthly_usage(self, tenant_id: str, year: int, month: int) -> Dict[str, Any]:
        """Get usage for a specific month."""
        start = datetime(year, month, 1)
        
        # Calculate end of month
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)
        
        return self.logger.aggregate_usage(tenant_id, start, end).to_dict()
    
    def get_usage_trend(
        self,
        tenant_id: str,
        start_date: datetime,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get usage trend over days."""
        trend = []
        for i in range(days):
            date = start_date + timedelta(days=i)
            daily_usage = self.get_daily_usage(tenant_id, date)
            daily_usage["date"] = date.date().isoformat()
            trend.append(daily_usage)
        
        return trend


from typing import Tuple
