"""
Query Optimizer - Analyze and optimize database queries
Provides query-level performance tracking and recommendations
"""

import time
import statistics
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from src.utils.monitoring import logger


@dataclass
class QueryProfile:
    """Profile of a single query execution"""
    query_id: str
    query_text: str
    execution_time_ms: float
    rows_affected: int
    timestamp: str
    indexed: bool = True
    has_joins: bool = False
    table: str = "unknown"


class QueryOptimizer:
    """
    Analyzes database queries and provides optimization recommendations.
    
    Features:
    - Query execution profiling
    - Pattern detection (N+1 queries, missing indexes, etc)
    - Performance trend analysis
    - Optimization recommendations
    - Query metrics aggregation
    """
    
    def __init__(self, slow_query_threshold_ms: float = 500.0):
        self.slow_query_threshold = slow_query_threshold_ms
        self.query_profiles: Dict[str, List[QueryProfile]] = {}
        self.optimization_suggestions: List[Dict[str, Any]] = []
    
    def record_query(
        self,
        query_id: str,
        query_text: str,
        execution_time_ms: float,
        rows_affected: int = 0,
        indexed: bool = True,
        has_joins: bool = False,
        table: str = "unknown"
    ):
        """Record a query execution"""
        profile = QueryProfile(
            query_id=query_id,
            query_text=query_text,
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            timestamp=datetime.utcnow().isoformat(),
            indexed=indexed,
            has_joins=has_joins,
            table=table,
        )
        
        if query_id not in self.query_profiles:
            self.query_profiles[query_id] = []
        
        self.query_profiles[query_id].append(profile)
        
        # Check for slow query
        if execution_time_ms > self.slow_query_threshold:
            self._analyze_slow_query(profile)
    
    def get_query_stats(self, query_id: str) -> Dict[str, Any]:
        """Get statistics for a specific query"""
        if query_id not in self.query_profiles:
            return {}
        
        profiles = self.query_profiles[query_id]
        times = [p.execution_time_ms for p in profiles]
        
        return {
            "query_id": query_id,
            "execution_count": len(profiles),
            "avg_time_ms": statistics.mean(times),
            "min_time_ms": min(times),
            "max_time_ms": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0,
            "total_rows_affected": sum(p.rows_affected for p in profiles),
            "latest_execution": profiles[-1].timestamp if profiles else None,
        }
    
    def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slowest queries"""
        all_queries = []
        
        for query_id, profiles in self.query_profiles.items():
            for profile in profiles:
                all_queries.append({
                    "query_id": query_id,
                    "execution_time_ms": profile.execution_time_ms,
                    "table": profile.table,
                    "timestamp": profile.timestamp,
                })
        
        # Sort by execution time
        all_queries.sort(key=lambda x: x["execution_time_ms"], reverse=True)
        
        return all_queries[:limit]
    
    def detect_n_plus_one_queries(self) -> List[Dict[str, Any]]:
        """Detect potential N+1 query patterns"""
        suspicious = []
        
        for query_id, profiles in self.query_profiles.items():
            if len(profiles) > 5:
                times = [p.execution_time_ms for p in profiles[-10:]]
                avg_time = statistics.mean(times)
                
                # High frequency + similar execution times = suspicious
                if avg_time < 50:  # Very fast queries
                    suspicious.append({
                        "query_id": query_id,
                        "pattern": "n_plus_one",
                        "frequency": len(profiles),
                        "avg_time_ms": round(avg_time, 2),
                        "severity": "medium",
                        "recommendation": "Consider using eager loading or batching",
                    })
        
        return suspicious
    
    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get query optimization recommendations"""
        recommendations = []
        
        # Check for slow queries
        slow = self.get_slow_queries(limit=5)
        for query in slow:
            if query["execution_time_ms"] > 2000:
                recommendations.append({
                    "type": "slow_query",
                    "severity": "critical",
                    "query_id": query["query_id"],
                    "execution_time_ms": query["execution_time_ms"],
                    "suggestion": "Query exceeds 2 seconds - add indexes or optimize joins",
                })
            elif query["execution_time_ms"] > 500:
                recommendations.append({
                    "type": "slow_query",
                    "severity": "warning",
                    "query_id": query["query_id"],
                    "execution_time_ms": query["execution_time_ms"],
                    "suggestion": "Consider adding database indexes or optimizing query",
                })
        
        # Check for N+1 patterns
        n_plus_one = self.detect_n_plus_one_queries()
        recommendations.extend(n_plus_one)
        
        # Check for missing indexes
        for query_id, profiles in list(self.query_profiles.items())[-20:]:
            for profile in profiles:
                if not profile.indexed and profile.execution_time_ms > 100:
                    recommendations.append({
                        "type": "missing_index",
                        "severity": "high",
                        "table": profile.table,
                        "suggestion": f"Consider adding index on {profile.table}",
                    })
        
        return recommendations
    
    def _analyze_slow_query(self, profile: QueryProfile):
        """Analyze a slow query"""
        logger.warning(
            f"Slow query detected: {profile.query_id} "
            f"({profile.execution_time_ms}ms, {profile.rows_affected} rows)"
        )
    
    def get_overview(self) -> Dict[str, Any]:
        """Get query optimization overview"""
        all_queries = []
        for profiles in self.query_profiles.values():
            all_queries.extend([p.execution_time_ms for p in profiles])
        
        if not all_queries:
            return {
                "total_queries": 0,
                "unique_queries": 0,
                "avg_execution_time": 0,
            }
        
        slow_queries = len([t for t in all_queries if t > self.slow_query_threshold])
        
        return {
            "total_queries": len(all_queries),
            "unique_queries": len(self.query_profiles),
            "avg_execution_time": round(statistics.mean(all_queries), 2),
            "median_execution_time": round(statistics.median(all_queries), 2),
            "max_execution_time": round(max(all_queries), 2),
            "slow_queries": slow_queries,
            "slow_query_percentage": round((slow_queries / len(all_queries) * 100), 2),
            "recommendations_count": len(self.get_recommendations()),
        }


# Global query optimizer instance
_query_optimizer = None


def get_query_optimizer(threshold_ms: float = 500.0) -> QueryOptimizer:
    """Get or create singleton query optimizer"""
    global _query_optimizer
    if _query_optimizer is None:
        _query_optimizer = QueryOptimizer(threshold_ms)
    return _query_optimizer
