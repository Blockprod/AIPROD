"""
Performance Models - Pydantic models for performance optimization endpoints
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class EndpointPerformanceData(BaseModel):
    """Performance data for a single endpoint"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "endpoint": "/api/users",
        "method": "GET",
        "request_count": 1234,
        "avg_duration_ms": 45.5,
        "min_duration_ms": 10.2,
        "max_duration_ms": 250.3,
    }})
    
    endpoint: str = Field(..., description="API endpoint path")
    method: str = Field(..., description="HTTP method")
    request_count: int = Field(..., description="Number of requests")
    avg_duration_ms: float = Field(..., ge=0, description="Average response time")
    min_duration_ms: float = Field(..., ge=0, description="Minimum response time")
    max_duration_ms: float = Field(..., ge=0, description="Maximum response time")
    median_duration_ms: Optional[float] = Field(None, ge=0, description="Median response time")
    std_dev: Optional[float] = Field(None, ge=0, description="Standard deviation")
    slow_request_count: int = Field(..., ge=0, description="Count of slow requests")
    slow_request_percentage: float = Field(..., ge=0, le=100, description="Percentage of slow requests")
    avg_db_queries: Optional[float] = Field(None, ge=0, description="Average database queries per request")
    avg_cache_hit_rate: Optional[float] = Field(None, ge=0, le=100, description="Average cache hit rate")


class CacheStatsResponse(BaseModel):
    """Cache statistics"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "hit_count": 5432,
        "miss_count": 1023,
        "hit_rate": 84.15,
        "total_entries": 456,
    }})
    
    backend: str = Field(..., description="Cache backend type")
    hit_count: int = Field(..., ge=0, description="Total cache hits")
    miss_count: int = Field(..., ge=0, description="Total cache misses")
    hit_rate: float = Field(..., ge=0, le=100, description="Cache hit rate percentage")
    total_entries: int = Field(..., ge=0, description="Current cache entries")
    max_size: int = Field(..., ge=0, description="Maximum cache size")
    expired_entries: int = Field(..., ge=0, description="Expired entries count")


class QueryOptimizationRecommendation(BaseModel):
    """Query optimization recommendation"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "type": "slow_query",
        "severity": "critical",
        "suggestion": "Query exceeds 2 seconds - add indexes or optimize joins",
    }})
    
    type: str = Field(..., description="Recommendation type")
    severity: str = Field(..., description="Severity level (info, warning, critical)")
    suggestion: str = Field(..., description="Optimization suggestion")
    query_id: Optional[str] = Field(None, description="Associated query ID")
    execution_time_ms: Optional[float] = Field(None, description="Query execution time")
    table: Optional[str] = Field(None, description="Associated table name")


class QueryOptimizationResponse(BaseModel):
    """Query optimization overview"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "total_queries": 5432,
        "unique_queries": 89,
        "avg_execution_time": 45.5,
        "slow_queries": 23,
        "recommendations": [],
    }})
    
    total_queries: int = Field(..., ge=0, description="Total query count")
    unique_queries: int = Field(..., ge=0, description="Unique queries count")
    avg_execution_time: float = Field(..., ge=0, description="Average execution time")
    median_execution_time: Optional[float] = Field(None, ge=0, description="Median execution time")
    max_execution_time: Optional[float] = Field(None, ge=0, description="Maximum execution time")
    slow_queries: int = Field(..., ge=0, description="Count of slow queries")
    slow_query_percentage: Optional[float] = Field(None, ge=0, le=100, description="Percentage of slow queries")
    recommendations: List[QueryOptimizationRecommendation] = Field(..., description="Optimization recommendations")


class AsyncTaskResponse(BaseModel):
    """Async task status"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "task_id": "abc123",
        "name": "bulk_export",
        "status": "running",
        "progress": 45.5,
    }})
    
    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Task name")
    status: str = Field(..., description="Task status")
    progress: float = Field(..., ge=0, le=100, description="Task progress percentage")
    created_at: str = Field(..., description="Task creation timestamp")
    started_at: Optional[str] = Field(None, description="Task start timestamp")
    completed_at: Optional[str] = Field(None, description="Task completion timestamp")
    execution_time_ms: Optional[float] = Field(None, ge=0, description="Execution time in milliseconds")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    error: Optional[str] = Field(None, description="Error message if failed")


class PerformanceProfileResponse(BaseModel):
    """Performance profile overview"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "total_requests": 10000,
        "avg_latency_ms": 125.5,
        "p95_latency_ms": 450.0,
        "p99_latency_ms": 850.0,
        "slow_requests": 234,
    }})
    
    total_requests: int = Field(..., ge=0, description="Total requests processed")
    unique_endpoints: int = Field(..., ge=0, description="Unique endpoints")
    avg_latency_ms: float = Field(..., ge=0, description="Average latency")
    p95_latency_ms: float = Field(..., ge=0, description="95th percentile latency")
    p99_latency_ms: float = Field(..., ge=0, description="99th percentile latency")
    max_latency_ms: float = Field(..., ge=0, description="Maximum latency")
    slow_requests: int = Field(..., ge=0, description="Count of slow requests")
    slow_request_percentage: float = Field(..., ge=0, le=100, description="Percentage of slow requests")


class PerformanceInsight(BaseModel):
    """Performance optimization insight"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "type": "slow_endpoint",
        "severity": "critical",
        "endpoint": "GET /api/users",
        "recommendation": "This endpoint is slow. Consider caching.",
    }})
    
    type: str = Field(..., description="Insight type")
    severity: str = Field(..., description="Severity level")
    endpoint: Optional[str] = Field(None, description="Associated endpoint")
    avg_duration_ms: Optional[float] = Field(None, ge=0, description="Average duration")
    query_count: Optional[int] = Field(None, ge=0, description="Database query count")
    hit_rate: Optional[float] = Field(None, ge=0, le=100, description="Hit rate")
    recommendation: str = Field(..., description="Optimization recommendation")


class CompressionStatsResponse(BaseModel):
    """Response compression statistics"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "compressed_responses": 234,
        "total_bytes_before": 50000000,
        "total_bytes_after": 12500000,
        "compression_ratio": 25.0,
    }})
    
    compressed_responses: int = Field(..., ge=0, description="Responses compressed")
    skipped_responses: int = Field(..., ge=0, description="Responses skipped")
    total_bytes_before: int = Field(..., ge=0, description="Total bytes before compression")
    total_bytes_after: int = Field(..., ge=0, description="Total bytes after compression")
    compression_ratio: float = Field(..., ge=0, le=100, description="Compression ratio percentage")
    bytes_saved: int = Field(..., ge=0, description="Total bytes saved")


class PerformanceOptimizationDashboard(BaseModel):
    """Complete performance optimization dashboard"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "status": "healthy",
        "performance_profile": {},
        "cache_stats": {},
    }})
    
    status: str = Field(..., description="Overall status (healthy/warning/critical)")
    performance_profile: PerformanceProfileResponse = Field(..., description="Performance profile")
    cache_stats: CacheStatsResponse = Field(..., description="Cache statistics")
    compression_stats: Optional[CompressionStatsResponse] = Field(None, description="Compression statistics")
    insights: List[PerformanceInsight] = Field(..., description="Performance insights")
    recommendations: List[str] = Field(..., description="Top optimization recommendations")


class PerformanceComparisonRequest(BaseModel):
    """Request for performance comparison"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "endpoints": ["/api/users", "/api/products"],
    }})
    
    endpoints: List[str] = Field(..., min_length=1, description="Endpoints to compare")
    time_period_hours: Optional[int] = Field(24, ge=1, description="Time period for comparison")


class PerformanceComparisonResponse(BaseModel):
    """Performance comparison result"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "comparison": {},
        "best_performer": "/api/users",
        "worst_performer": "/api/products",
    }})
    
    endpoints: List[EndpointPerformanceData] = Field(..., description="Endpoint performance data")
    best_performer: str = Field(..., description="Best performing endpoint")
    worst_performer: str = Field(..., description="Worst performing endpoint")
    performance_gap_ms: float = Field(..., ge=0, description="Performance gap between best and worst")
