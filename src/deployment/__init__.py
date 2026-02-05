"""
Deployment module - Multi-region orchestration and failover management
"""

from src.deployment.region_manager import (
    get_region_manager,
    RegionManager,
    Region,
    RegionStatus,
    RegionTier,
    RegionMetrics,
)
from src.deployment.failover_manager import (
    get_failover_manager,
    FailoverManager,
    FailoverPolicy,
    FailoverStrategy,
    FailoverTrigger,
    FailoverEvent,
)
from src.deployment.deployment_routes import setup_deployment_routes

__all__ = [
    "get_region_manager",
    "RegionManager",
    "Region",
    "RegionStatus",
    "RegionTier",
    "RegionMetrics",
    "get_failover_manager",
    "FailoverManager",
    "FailoverPolicy",
    "FailoverStrategy",
    "FailoverTrigger",
    "FailoverEvent",
    "setup_deployment_routes",
]
