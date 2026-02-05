"""
Performance Profiler - Request and system performance profiling
Tracks performance bottlenecks and generates optimization insights
"""

import time
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from src.utils.monitoring import logger


@dataclass
class PerformanceProfile:
    """Performance data for a single request"""
    endpoint: str
    method: str
    duration_ms: float
    timestamp: str
    status_code: int
    db_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class PerformanceProfiler:
    """
    Profiles request performance and identifies bottlenecks.
    
    Features:
    - Request-level performance tracking
    - Bottleneck identification
    - Performance trend analysis
    - Optimization recommendations
    - Historical performance data
    """
    
    def __init__(self, slow_threshold_ms: float = 500.0):
        self.slow_threshold = slow_threshold_ms
        self.profiles: List[PerformanceProfile] = []
        self.endpoint_stats: Dict[str, Dict[str, Any]] = {}
    
    def record_request(
        self,
        endpoint: str,
        method: str,
        duration_ms: float,
        status_code: int,
        db_queries: int = 0,
        cache_hits: int = 0,
        cache_misses: int = 0,
    ):
        """Record a request's performance"""
        profile = PerformanceProfile(
            endpoint=endpoint,
            method=method,
            duration_ms=duration_ms,
            timestamp=datetime.utcnow().isoformat(),
            status_code=status_code,
            db_queries=db_queries,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )
        
        self.profiles.append(profile)
        
        # Update endpoint stats
        key = f"{method}:{endpoint}"
        if key not in self.endpoint_stats:
            self.endpoint_stats[key] = {
                "count": 0,
                "total_time": 0,
                "slow_count": 0,
                "error_count": 0,
            }
        
        stats = self.endpoint_stats[key]
        stats["count"] += 1
        stats["total_time"] += duration_ms
        
        if duration_ms > self.slow_threshold:
            stats["slow_count"] += 1
        
        if status_code >= 400:
            stats["error_count"] += 1
        
        # Log slow requests
        if duration_ms > self.slow_threshold:
            logger.warning(
                f"Slow request: {method} {endpoint} ({duration_ms}ms)"
            )
    
    def get_endpoint_performance(self, method: str, endpoint: str) -> Dict[str, Any]:
        """Get performance stats for an endpoint"""
        key = f"{method}:{endpoint}"
        
        if key not in self.endpoint_stats:
            return {}
        
        # Get recent profiles for this endpoint
        endpoint_profiles = [
            p for p in self.profiles[-1000:]
            if p.endpoint == endpoint and p.method == method
        ]
        
        if not endpoint_profiles:
            return {}
        
        times = [p.duration_ms for p in endpoint_profiles]
        
        return {
            "endpoint": endpoint,
            "method": method,
            "request_count": len(endpoint_profiles),
            "avg_duration_ms": round(statistics.mean(times), 2),
            "min_duration_ms": min(times),
            "max_duration_ms": max(times),
            "median_duration_ms": round(statistics.median(times), 2),
            "std_dev": round(statistics.stdev(times), 2) if len(times) > 1 else 0,
            "slow_request_count": sum(1 for t in times if t > self.slow_threshold),
            "slow_request_percentage": round(
                (sum(1 for t in times if t > self.slow_threshold) / len(times)) * 100,
                2
            ),
            "avg_db_queries": round(
                statistics.mean([p.db_queries for p in endpoint_profiles]),
                2
            ),
            "avg_cache_hit_rate": round(
                statistics.mean([
                    (p.cache_hits / (p.cache_hits + p.cache_misses) * 100)
                    if (p.cache_hits + p.cache_misses) > 0 else 0
                    for p in endpoint_profiles
                ]),
                2
            ),
        }
    
    def get_slowest_endpoints(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest endpoints by average duration"""
        endpoint_perf = [
            self.get_endpoint_performance(key.split(":")[0], key.split(":")[1])
            for key in self.endpoint_stats.keys()
        ]
        
        # Filter out empty results
        endpoint_perf = [p for p in endpoint_perf if p]
        
        # Sort by avg duration
        endpoint_perf.sort(key=lambda x: x["avg_duration_ms"], reverse=True)
        
        return endpoint_perf[:limit]
    
    def get_performance_insights(self) -> List[Dict[str, Any]]:
        """Generate performance insights and recommendations"""
        insights = []
        
        # Analyze slowest endpoints
        slowest = self.get_slowest_endpoints(limit=5)
        for endpoint_data in slowest:
            if endpoint_data["avg_duration_ms"] > 1000:
                insights.append({
                    "type": "slow_endpoint",
                    "severity": "critical",
                    "endpoint": f'{endpoint_data["method"]} {endpoint_data["endpoint"]}',
                    "avg_duration_ms": endpoint_data["avg_duration_ms"],
                    "recommendation": "This endpoint is slow. Consider caching, query optimization, or async processing.",
                })
            elif endpoint_data["avg_duration_ms"] > 500:
                insights.append({
                    "type": "slow_endpoint",
                    "severity": "warning",
                    "endpoint": f'{endpoint_data["method"]} {endpoint_data["endpoint"]}',
                    "avg_duration_ms": endpoint_data["avg_duration_ms"],
                    "recommendation": "Monitor performance closely. Consider optimization strategies.",
                })
            
            # Check cache hit rate
            if endpoint_data.get("avg_cache_hit_rate", 0) < 50:
                insights.append({
                    "type": "low_cache_hit_rate",
                    "severity": "info",
                    "endpoint": f'{endpoint_data["method"]} {endpoint_data["endpoint"]}',
                    "hit_rate": endpoint_data["avg_cache_hit_rate"],
                    "recommendation": "Low cache hit rate. Consider adjusting cache TTL or cache strategy.",
                })
            
            # Check database query count
            if endpoint_data.get("avg_db_queries", 0) > 5:
                insights.append({
                    "type": "high_query_count",
                    "severity": "warning",
                    "endpoint": f'{endpoint_data["method"]} {endpoint_data["endpoint"]}',
                    "query_count": endpoint_data["avg_db_queries"],
                    "recommendation": "High database query count. Consider eager loading or query optimization.",
                })
        
        return insights
    
    def get_overview(self) -> Dict[str, Any]:
        """Get performance profiling overview"""
        if not self.profiles:
            return {
                "total_requests": 0,
                "unique_endpoints": 0,
                "avg_latency_ms": 0,
            }
        
        recent_profiles = self.profiles[-10000:]
        times = [p.duration_ms for p in recent_profiles]
        
        slow_requests = sum(1 for t in times if t > self.slow_threshold)
        
        return {
            "total_requests": len(recent_profiles),
            "unique_endpoints": len(self.endpoint_stats),
            "avg_latency_ms": round(statistics.mean(times), 2),
            "p95_latency_ms": round(
                sorted(times)[int(len(times) * 0.95)],
                2
            ) if times else 0,
            "p99_latency_ms": round(
                sorted(times)[int(len(times) * 0.99)],
                2
            ) if times else 0,
            "max_latency_ms": max(times),
            "slow_requests": slow_requests,
            "slow_request_percentage": round(
                (slow_requests / len(times) * 100),
                2
            ) if times else 0,
        }
    
    def cleanup_old_profiles(self, hours: int = 24):
        """Remove profiles older than specified hours"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        initial_count = len(self.profiles)
        self.profiles = [
            p for p in self.profiles
            if datetime.fromisoformat(p.timestamp) > cutoff
        ]
        
        removed = initial_count - len(self.profiles)
        logger.info(f"Cleaned up {removed} old performance profiles")


# Global performance profiler instance
_performance_profiler = None


def get_performance_profiler(threshold_ms: float = 500.0) -> PerformanceProfiler:
    """Get or create singleton performance profiler"""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler(threshold_ms)
    return _performance_profiler
