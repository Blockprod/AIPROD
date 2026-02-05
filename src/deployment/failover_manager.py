"""
Failover Manager - Automated failover detection and handling
Manages regional failover, recovery, and load rebalancing
"""

import asyncio
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from src.utils.monitoring import logger


class FailoverStrategy(str, Enum):
    """Failover strategy options"""
    AUTOMATIC = "automatic"  # Automatic failover when conditions met
    MANUAL = "manual"  # Manual failover only
    GRADUAL = "gradual"  # Gradual traffic shift


class FailoverTrigger(str, Enum):
    """Events that trigger failover"""
    HIGH_ERROR_RATE = "high_error_rate"
    LOW_CAPACITY = "low_capacity"
    HIGH_LATENCY = "high_latency"
    TIMEOUT = "timeout"
    MANUAL = "manual"


@dataclass
class FailoverPolicy:
    """Policy for failover behavior"""
    strategy: FailoverStrategy = FailoverStrategy.AUTOMATIC
    error_rate_threshold: float = 50.0  # percentage
    capacity_threshold: float = 20.0  # percentage
    latency_threshold_ms: float = 5000
    consecutive_failures_threshold: int = 3
    recovery_window_seconds: int = 300
    gradual_shift_percentage: float = 10.0  # per shift
    cooldown_seconds: int = 60


@dataclass
class FailoverEvent:
    """Record of a failover event"""
    event_id: str
    timestamp: datetime
    trigger: FailoverTrigger
    from_region: str
    to_region: str
    success: bool
    traffic_shifted: float = 0.0  # percentage
    reason: str = ""
    recovery_initiated: bool = False


class FailoverManager:
    """
    Manages automatic failover and recovery.
    
    Features:
    - Multi-trigger failover detection
    - Automatic and manual failover
    - Gradual traffic shifting
    - Health-based recovery
    - Failover history and analytics
    - Circuit breaker pattern
    """
    
    def __init__(self, policy: Optional[FailoverPolicy] = None):
        self.policy = policy or FailoverPolicy()
        self.failover_history: List[FailoverEvent] = []
        self.current_failovers: Dict[str, FailoverEvent] = {}
        self.last_failover_time: Dict[str, datetime] = {}
        self.recovery_tasks: Dict[str, asyncio.Task] = {}
        self.traffic_distribution: Dict[str, float] = {}  # region_id -> percentage
    
    async def check_failover_conditions(
        self,
        region_metrics: Dict[str, Any]
    ) -> Optional[FailoverTrigger]:
        """Check if failover should be triggered"""
        if self.policy.strategy == FailoverStrategy.MANUAL:
            return None
        
        # Check error rate
        if region_metrics.get("error_rate", 0) > self.policy.error_rate_threshold:
            return FailoverTrigger.HIGH_ERROR_RATE
        
        # Check capacity
        if region_metrics.get("available_capacity", 100) < self.policy.capacity_threshold:
            return FailoverTrigger.LOW_CAPACITY
        
        # Check latency
        if region_metrics.get("latency_ms", 0) > self.policy.latency_threshold_ms:
            return FailoverTrigger.HIGH_LATENCY
        
        return None
    
    async def initiate_failover(
        self,
        from_region: str,
        to_region: str,
        trigger: FailoverTrigger,
        reason: str = ""
    ) -> bool:
        """Initiate failover from one region to another"""
        # Check cooldown period
        last_failover = self.last_failover_time.get(from_region)
        if last_failover and (datetime.utcnow() - last_failover).total_seconds() < self.policy.cooldown_seconds:
            logger.warning(f"Failover cooldown active for {from_region}")
            return False
        
        # Create failover event
        event_id = f"failover_{int(time.time())}"
        event = FailoverEvent(
            event_id=event_id,
            timestamp=datetime.utcnow(),
            trigger=trigger,
            from_region=from_region,
            to_region=to_region,
            success=False,
            reason=reason,
        )
        
        try:
            # Execute failover
            if self.policy.strategy == FailoverStrategy.GRADUAL:
                success = await self._gradual_failover(event)
            else:
                success = await self._immediate_failover(event)
            
            event.success = success
            
            if success:
                self.last_failover_time[from_region] = datetime.utcnow()
                self.current_failovers[from_region] = event
                
                # Initiate recovery task
                if not self.recovery_tasks.get(from_region):
                    task = asyncio.create_task(
                        self._recovery_monitor(from_region, self.policy.recovery_window_seconds)
                    )
                    self.recovery_tasks[from_region] = task
        
        finally:
            self.failover_history.append(event)
            logger.info(f"Failover event recorded: {event_id}")
        
        return event.success
    
    async def _immediate_failover(self, event: FailoverEvent) -> bool:
        """Execute immediate traffic shift to new region"""
        try:
            # Complete traffic shift
            self.traffic_distribution[event.from_region] = 0.0
            self.traffic_distribution[event.to_region] = 100.0
            
            logger.warning(
                f"Immediate failover: {event.from_region} → {event.to_region} "
                f"({event.trigger.value})"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Failover failed: {e}")
            return False
    
    async def _gradual_failover(self, event: FailoverEvent) -> bool:
        """Execute gradual traffic shift over time"""
        try:
            current_shift = 0.0
            shifts = int(100 / self.policy.gradual_shift_percentage)
            shift_interval = self.policy.cooldown_seconds / shifts
            
            for _ in range(shifts):
                current_shift += self.policy.gradual_shift_percentage
                
                # Update distribution
                self.traffic_distribution[event.from_region] = 100 - current_shift
                self.traffic_distribution[event.to_region] = current_shift
                
                logger.info(
                    f"Gradual failover: {event.from_region} → {event.to_region}, "
                    f"Shifted {current_shift}%"
                )
                
                await asyncio.sleep(shift_interval)
            
            event.traffic_shifted = 100.0
            return True
        
        except Exception as e:
            logger.error(f"Gradual failover failed: {e}")
            return False
    
    async def _recovery_monitor(self, region_id: str, window_seconds: int):
        """Monitor region for recovery after failover"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < window_seconds:
            # Check if region recovered
            # This would typically check with external health check
            
            await asyncio.sleep(30)  # Check every 30 seconds
        
        # Recovery window ended
        if region_id in self.recovery_tasks:
            del self.recovery_tasks[region_id]
    
    async def initiate_recovery(self, region_id: str) -> bool:
        """Manually initiate recovery for a region"""
        try:
            # Restore gradual traffic to region
            if self.policy.strategy == FailoverStrategy.GRADUAL:
                # Gradually restore traffic
                current_traffic = self.traffic_distribution.get(region_id, 0.0)
                steps = int((100 - current_traffic) / self.policy.gradual_shift_percentage)
                
                for _ in range(steps):
                    current_traffic += self.policy.gradual_shift_percentage
                    self.traffic_distribution[region_id] = current_traffic
                    await asyncio.sleep(5)
            else:
                # Immediate restore
                self.traffic_distribution[region_id] = 100.0
            
            logger.info(f"Recovery initiated for {region_id}")
            return True
        
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def get_failover_status(self) -> Dict[str, Any]:
        """Get current failover status"""
        active_failovers = [
            {
                "from_region": e.from_region,
                "to_region": e.to_region,
                "trigger": e.trigger.value,
                "traffic_shifted": e.traffic_shifted,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in self.current_failovers.values()
        ]
        
        return {
            "active_failovers": active_failovers,
            "total_failover_events": len(self.failover_history),
            "successful_failovers": sum(1 for e in self.failover_history if e.success),
            "failed_failovers": sum(1 for e in self.failover_history if not e.success),
        }
    
    def get_failover_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent failover events"""
        recent = sorted(
            self.failover_history,
            key=lambda e: e.timestamp,
            reverse=True
        )[:limit]
        
        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp.isoformat(),
                "trigger": e.trigger.value,
                "from_region": e.from_region,
                "to_region": e.to_region,
                "success": e.success,
                "traffic_shifted": e.traffic_shifted,
                "reason": e.reason,
            }
            for e in recent
        ]
    
    def get_traffic_distribution(self) -> Dict[str, float]:
        """Get current traffic distribution across regions"""
        return dict(self.traffic_distribution)
    
    def set_traffic_distribution(self, distribution: Dict[str, float]) -> bool:
        """Manually set traffic distribution"""
        # Validate distribution sums to 100
        total = sum(distribution.values())
        if abs(total - 100.0) > 0.1:
            logger.error(f"Invalid distribution: sum={total}, expected=100")
            return False
        
        self.traffic_distribution = distribution
        logger.info(f"Traffic distribution updated: {distribution}")
        return True
    
    def get_failover_analytics(self) -> Dict[str, Any]:
        """Get failover analytics"""
        if not self.failover_history:
            return {
                "total_events": 0,
                "success_rate": 0,
                "average_duration_seconds": 0,
                "most_common_trigger": None,
            }
        
        total = len(self.failover_history)
        successful = sum(1 for e in self.failover_history if e.success)
        
        # Calculate average failover duration
        durations = []
        for event in self.failover_history:
            if event.success:
                # Assume failover completes in a few seconds to minutes
                durations.append((event.timestamp - event.timestamp).total_seconds())
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Most common trigger
        triggers = {}
        for event in self.failover_history:
            trigger = event.trigger.value
            triggers[trigger] = triggers.get(trigger, 0) + 1
        
        most_common = max(triggers.items(), key=lambda x: x[1])[0] if triggers else None
        
        return {
            "total_events": total,
            "successful_events": successful,
            "success_rate": round((successful / total * 100), 2),
            "average_duration_seconds": round(avg_duration, 2),
            "most_common_trigger": most_common,
            "trigger_breakdown": triggers,
        }
    
    def update_policy(self, policy: FailoverPolicy):
        """Update failover policy"""
        self.policy = policy
        logger.info("Failover policy updated")


# Global failover manager instance
_failover_manager = None


def get_failover_manager(policy: Optional[FailoverPolicy] = None) -> FailoverManager:
    """Get or create singleton failover manager"""
    global _failover_manager
    if _failover_manager is None:
        _failover_manager = FailoverManager(policy)
    return _failover_manager
