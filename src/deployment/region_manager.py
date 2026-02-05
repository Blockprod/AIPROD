"""
Region Manager - Multi-region coordination and orchestration
Manages deployment across multiple geographic regions
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import uuid
from src.utils.monitoring import logger


class RegionStatus(str, Enum):
    """Region operational status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class RegionTier(str, Enum):
    """Region priority tier"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"


@dataclass
class RegionMetrics:
    """Metrics for a single region"""
    region_id: str
    region_name: str
    latency_ms: float = 0
    available_capacity: float = 100.0  # percentage
    error_rate: float = 0.0  # percentage
    request_count: int = 0
    last_check: datetime = field(default_factory=datetime.utcnow)
    uptime_percentage: float = 100.0
    response_time_p95: float = 0
    response_time_p99: float = 0


@dataclass
class Region:
    """Represents a geographic region deployment"""
    region_id: str
    region_name: str
    endpoint: str
    tier: RegionTier
    max_capacity: int = 1000
    metrics: RegionMetrics = field(default_factory=lambda: RegionMetrics("", ""))
    status: RegionStatus = RegionStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    health_check_interval_seconds: int = 30
    consecutive_failures: int = 0
    enabled: bool = True


class RegionManager:
    """
    Manages multi-region deployment orchestration.
    
    Features:
    - Region registration and management
    - Regional health monitoring
    - Load distribution across regions
    - Performance comparison
    - Capacity planning
    - Regional metrics aggregation
    """
    
    def __init__(self):
        self.regions: Dict[str, Region] = {}
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.region_history: List[Dict[str, Any]] = []
        self.failover_events: List[Dict[str, Any]] = []
        self.max_consecutive_failures = 3
    
    def register_region(
        self,
        region_name: str,
        endpoint: str,
        tier: RegionTier = RegionTier.SECONDARY,
        max_capacity: int = 1000,
        health_check_interval: int = 30,
    ) -> str:
        """Register a new region"""
        region_id = str(uuid.uuid4())[:8]
        
        region = Region(
            region_id=region_id,
            region_name=region_name,
            endpoint=endpoint,
            tier=tier,
            max_capacity=max_capacity,
            metrics=RegionMetrics(region_id=region_id, region_name=region_name),
            health_check_interval_seconds=health_check_interval,
        )
        
        self.regions[region_id] = region
        
        logger.info(
            f"Region registered: {region_name} ({region_id}) - "
            f"Tier: {tier.value}, Endpoint: {endpoint}"
        )
        
        return region_id
    
    def get_region(self, region_id: str) -> Optional[Region]:
        """Get region by ID"""
        return self.regions.get(region_id)
    
    def get_region_by_name(self, region_name: str) -> Optional[Region]:
        """Get region by name"""
        for region in self.regions.values():
            if region.region_name == region_name:
                return region
        return None
    
    def get_all_regions(self) -> List[Region]:
        """Get all registered regions"""
        return list(self.regions.values())
    
    def get_healthy_regions(self) -> List[Region]:
        """Get only healthy regions"""
        return [
            r for r in self.regions.values()
            if r.status == RegionStatus.HEALTHY and r.enabled
        ]
    
    def get_primary_regions(self) -> List[Region]:
        """Get primary tier regions"""
        return [
            r for r in self.regions.values()
            if r.tier == RegionTier.PRIMARY and r.status != RegionStatus.UNHEALTHY
        ]
    
    def update_region_metrics(
        self,
        region_id: str,
        latency_ms: float,
        available_capacity: float,
        error_rate: float,
        request_count: int,
        response_time_p95: float = 0,
        response_time_p99: float = 0,
    ):
        """Update metrics for a region"""
        region = self.get_region(region_id)
        if not region:
            return
        
        region.metrics.latency_ms = latency_ms
        region.metrics.available_capacity = available_capacity
        region.metrics.error_rate = error_rate
        region.metrics.request_count = request_count
        region.metrics.response_time_p95 = response_time_p95
        region.metrics.response_time_p99 = response_time_p99
        region.metrics.last_check = datetime.utcnow()
        
        # Determine status based on metrics
        self._update_region_status(region)
    
    def _update_region_status(self, region: Region):
        """Update region status based on metrics"""
        old_status = region.status
        
        if region.metrics.error_rate > 50 and region.metrics.available_capacity < 20:
            region.status = RegionStatus.UNHEALTHY
            region.consecutive_failures += 1
        elif region.metrics.error_rate > 10 or region.metrics.available_capacity < 50 or region.metrics.latency_ms > 1000:
            region.status = RegionStatus.DEGRADED
            region.consecutive_failures = max(0, region.consecutive_failures - 1)
        else:
            region.status = RegionStatus.HEALTHY
            region.consecutive_failures = 0
        
        if old_status != region.status:
            logger.warning(
                f"Region {region.region_name} status changed: "
                f"{old_status.value} → {region.status.value}"
            )
    
    def enable_region(self, region_id: str):
        """Enable a region"""
        region = self.get_region(region_id)
        if region:
            region.enabled = True
            logger.info(f"Region {region.region_name} enabled")
    
    def disable_region(self, region_id: str):
        """Disable a region"""
        region = self.get_region(region_id)
        if region:
            region.enabled = False
            logger.info(f"Region {region.region_name} disabled")
    
    def get_recommended_region(self) -> Optional[Region]:
        """Get best region for routing based on metrics"""
        healthy = self.get_healthy_regions()
        if not healthy:
            return None
        
        # Prefer primary regions
        primaries = [r for r in healthy if r.tier == RegionTier.PRIMARY]
        candidates = primaries if primaries else healthy
        
        # Sort by lowest latency
        candidates.sort(key=lambda r: r.metrics.latency_ms)
        
        return candidates[0] if candidates else None
    
    def get_regional_comparison(self) -> Dict[str, Any]:
        """Get performance comparison across regions"""
        regions_data = []
        
        for region in self.get_all_regions():
            regions_data.append({
                "region_id": region.region_id,
                "region_name": region.region_name,
                "tier": region.tier.value,
                "status": region.status.value,
                "latency_ms": region.metrics.latency_ms,
                "available_capacity": region.metrics.available_capacity,
                "error_rate": region.metrics.error_rate,
                "request_count": region.metrics.request_count,
                "uptime_percentage": region.metrics.uptime_percentage,
                "p95_latency": region.metrics.response_time_p95,
                "p99_latency": region.metrics.response_time_p99,
            })
        
        # Sort by latency
        regions_data.sort(key=lambda x: x["latency_ms"])
        
        return {
            "total_regions": len(self.regions),
            "healthy_regions": len(self.get_healthy_regions()),
            "primary_regions": len(self.get_primary_regions()),
            "regions": regions_data,
            "best_performing": regions_data[0] if regions_data else None,
            "worst_performing": regions_data[-1] if regions_data else None,
        }
    
    def get_capacity_analysis(self) -> Dict[str, Any]:
        """Analyze total capacity across regions"""
        total_capacity = sum(r.max_capacity for r in self.regions.values())
        used_capacity = sum(
            r.max_capacity * (100 - r.metrics.available_capacity) / 100
            for r in self.regions.values()
        )
        available = sum(
            r.max_capacity * r.metrics.available_capacity / 100
            for r in self.regions.values()
        )
        
        return {
            "total_capacity": total_capacity,
            "used_capacity": used_capacity,
            "available_capacity": available,
            "utilization_percentage": (used_capacity / total_capacity * 100) if total_capacity > 0 else 0,
            "regional_breakdown": [
                {
                    "region": r.region_name,
                    "capacity": r.max_capacity,
                    "available": r.max_capacity * r.metrics.available_capacity / 100,
                    "utilization": (100 - r.metrics.available_capacity),
                }
                for r in self.regions.values()
            ],
        }
    
    def get_overview(self) -> Dict[str, Any]:
        """Get multi-region overview"""
        healthy = len(self.get_healthy_regions())
        total = len(self.regions)
        
        # Calculate average error rate
        error_rates = [r.metrics.error_rate for r in self.regions.values()]
        avg_error_rate = sum(error_rates) / len(error_rates) if error_rates else 0
        
        # Calculate average latency
        latencies = [r.metrics.latency_ms for r in self.regions.values()]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        return {
            "total_regions": total,
            "healthy_regions": healthy,
            "degraded_regions": sum(1 for r in self.regions.values() if r.status == RegionStatus.DEGRADED),
            "unhealthy_regions": sum(1 for r in self.regions.values() if r.status == RegionStatus.UNHEALTHY),
            "average_error_rate": round(avg_error_rate, 2),
            "average_latency_ms": round(avg_latency, 2),
            "health_percentage": round((healthy / total * 100) if total > 0 else 0, 2),
        }
    
    def record_failover_event(self, from_region: str, to_region: str, reason: str):
        """Record a failover event"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_region": from_region,
            "to_region": to_region,
            "reason": reason,
        }
        self.failover_events.append(event)
        logger.warning(f"Failover: {from_region} → {to_region} ({reason})")


# Global region manager instance
_region_manager = None


def get_region_manager() -> RegionManager:
    """Get or create singleton region manager"""
    global _region_manager
    if _region_manager is None:
        _region_manager = RegionManager()
    return _region_manager
