"""
Performance Optimization Routes - API endpoints for performance features
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List, Optional
from src.performance.caching_service import get_caching_service, CacheBackend
from src.performance.query_optimizer import get_query_optimizer
from src.performance.performance_profiler import get_performance_profiler
from src.performance.async_processor import get_async_processor, Task
from src.performance.compression_middleware import CompressionMiddleware
from src.performance.performance_models import (
    EndpointPerformanceData,
    CacheStatsResponse,
    QueryOptimizationResponse,
    QueryOptimizationRecommendation,
    AsyncTaskResponse,
    PerformanceProfileResponse,
    PerformanceInsight,
    CompressionStatsResponse,
    PerformanceOptimizationDashboard,
    PerformanceComparisonResponse,
)

router = APIRouter(prefix="/performance", tags=["performance"])
limiter = Limiter(key_func=get_remote_address)


@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Get cache statistics",
    description="Get comprehensive cache performance statistics",
)
@limiter.limit("100/minute")
async def get_cache_stats(request: Request) -> CacheStatsResponse:
    """Get cache performance statistics"""
    cache_service = get_caching_service()
    stats = cache_service.get_stats()
    
    return CacheStatsResponse(
        backend=stats["backend"],
        hit_count=stats["hit_count"],
        miss_count=stats["miss_count"],
        hit_rate=stats["hit_rate"],
        total_entries=stats["cache_stats"]["total_entries"],
        max_size=stats["cache_stats"]["max_size"],
        expired_entries=stats["cache_stats"]["expired_entries"],
    )


@router.post(
    "/cache/clear",
    summary="Clear cache",
    description="Clear all cached data",
)
@limiter.limit("10/minute")
async def clear_cache(request: Request) -> dict:
    """Clear all cache"""
    cache_service = get_caching_service()
    cache_service.memory_cache.clear()
    cache_service.reset_stats()
    
    return {
        "status": "success",
        "message": "Cache cleared successfully"
    }


@router.get(
    "/query/optimization",
    response_model=QueryOptimizationResponse,
    summary="Get query optimization insights",
    description="Get database query optimization recommendations",
)
@limiter.limit("30/minute")
async def get_query_optimization(request: Request) -> QueryOptimizationResponse:
    """Get query optimization insights"""
    optimizer = get_query_optimizer()
    overview = optimizer.get_overview()
    recommendations = optimizer.get_recommendations()
    
    # Convert recommendations to models
    rec_models = [
        QueryOptimizationRecommendation(
            type=r.get("type", "unknown"),
            severity=r.get("severity", "info"),
            suggestion=r.get("suggestion", ""),
            query_id=r.get("query_id"),
            execution_time_ms=r.get("execution_time_ms"),
            table=r.get("table"),
        )
        for r in recommendations
    ]
    
    return QueryOptimizationResponse(
        total_queries=overview.get("total_queries", 0),
        unique_queries=overview.get("unique_queries", 0),
        avg_execution_time=overview.get("avg_execution_time", 0),
        median_execution_time=overview.get("median_execution_time"),
        max_execution_time=overview.get("max_execution_time"),
        slow_queries=overview.get("slow_queries", 0),
        slow_query_percentage=overview.get("slow_query_percentage"),
        recommendations=rec_models,
    )


@router.get(
    "/profile/overview",
    response_model=PerformanceProfileResponse,
    summary="Get performance profile overview",
    description="Get comprehensive performance profiling data",
)
@limiter.limit("50/minute")
async def get_profile_overview(request: Request) -> PerformanceProfileResponse:
    """Get performance profile overview"""
    profiler = get_performance_profiler()
    overview = profiler.get_overview()
    
    return PerformanceProfileResponse(
        total_requests=overview.get("total_requests", 0),
        unique_endpoints=overview.get("unique_endpoints", 0),
        avg_latency_ms=overview.get("avg_latency_ms", 0),
        p95_latency_ms=overview.get("p95_latency_ms", 0),
        p99_latency_ms=overview.get("p99_latency_ms", 0),
        max_latency_ms=overview.get("max_latency_ms", 0),
        slow_requests=overview.get("slow_requests", 0),
        slow_request_percentage=overview.get("slow_request_percentage", 0),
    )


@router.get(
    "/profile/slowest",
    response_model=List[EndpointPerformanceData],
    summary="Get slowest endpoints",
    description="Get list of slowest performing endpoints",
)
@limiter.limit("50/minute")
async def get_slowest_endpoints(
    request: Request,
    limit: int = Query(10, ge=1, le=100)
) -> List[EndpointPerformanceData]:
    """Get slowest endpoints"""
    profiler = get_performance_profiler()
    slowest = profiler.get_slowest_endpoints(limit=limit)
    
    return [
        EndpointPerformanceData(
            endpoint=e.get("endpoint", "unknown"),
            method=e.get("method", "GET"),
            request_count=e.get("request_count", 0),
            avg_duration_ms=e.get("avg_duration_ms", 0),
            min_duration_ms=e.get("min_duration_ms", 0),
            max_duration_ms=e.get("max_duration_ms", 0),
            median_duration_ms=e.get("median_duration_ms"),
            std_dev=e.get("std_dev"),
            slow_request_count=e.get("slow_request_count", 0),
            slow_request_percentage=e.get("slow_request_percentage", 0),
            avg_db_queries=e.get("avg_db_queries"),
            avg_cache_hit_rate=e.get("avg_cache_hit_rate"),
        )
        for e in slowest
    ]


@router.get(
    "/profile/insights",
    response_model=List[PerformanceInsight],
    summary="Get performance insights",
    description="Get AI-generated performance optimization insights",
)
@limiter.limit("30/minute")
async def get_performance_insights(request: Request) -> List[PerformanceInsight]:
    """Get performance insights"""
    profiler = get_performance_profiler()
    insights = profiler.get_performance_insights()
    
    return [
        PerformanceInsight(
            type=i.get("type", "unknown"),
            severity=i.get("severity", "info"),
            endpoint=i.get("endpoint"),
            avg_duration_ms=i.get("avg_duration_ms"),
            query_count=i.get("query_count"),
            hit_rate=i.get("hit_rate"),
            recommendation=i.get("recommendation", ""),
        )
        for i in insights
    ]


@router.get(
    "/tasks/{task_id}",
    response_model=AsyncTaskResponse,
    summary="Get async task status",
    description="Get status of an async task",
)
@limiter.limit("100/minute")
async def get_task_status(request: Request, task_id: str) -> AsyncTaskResponse:
    """Get task status"""
    processor = get_async_processor()
    task = processor.get_task(task_id)
    
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_dict = task.to_dict()
    
    return AsyncTaskResponse(
        task_id=task_dict["task_id"],
        name=task_dict["name"],
        status=task_dict["status"],
        progress=task_dict["progress"],
        created_at=task_dict["created_at"],
        started_at=task_dict["started_at"],
        completed_at=task_dict["completed_at"],
        execution_time_ms=task_dict["execution_time_ms"],
        result=task_dict.get("result"),
        error=task_dict.get("error"),
    )


@router.get(
    "/tasks",
    response_model=List[AsyncTaskResponse],
    summary="List active tasks",
    description="Get list of active async tasks",
)
@limiter.limit("50/minute")
async def list_active_tasks(request: Request) -> List[AsyncTaskResponse]:
    """List active tasks"""
    processor = get_async_processor()
    tasks = processor.get_active_tasks()
    
    return [
        AsyncTaskResponse(
            task_id=t["task_id"],
            name=t["name"],
            status=t["status"],
            progress=t["progress"],
            created_at=t["created_at"],
            started_at=t["started_at"],
            completed_at=t["completed_at"],
            execution_time_ms=t["execution_time_ms"],
            result=t.get("result"),
            error=t.get("error"),
        )
        for t in tasks
    ]


@router.delete(
    "/tasks/{task_id}",
    summary="Cancel async task",
    description="Cancel an async task",
)
@limiter.limit("20/minute")
async def cancel_task(request: Request, task_id: str) -> dict:
    """Cancel task"""
    processor = get_async_processor()
    success = processor.cancel_task(task_id)
    
    if not success:
        raise HTTPException(
            status_code=409,
            detail="Cannot cancel task (not pending or running)"
        )
    
    return {
        "status": "success",
        "message": "Task cancelled"
    }


@router.get(
    "/compression/stats",
    response_model=CompressionStatsResponse,
    summary="Get compression statistics",
    description="Get response compression performance statistics",
)
@limiter.limit("50/minute")
async def get_compression_stats(request: Request) -> CompressionStatsResponse:
    """Get compression statistics"""
    # This is a placeholder - compression middleware would be accessed here
    # In a real implementation, we'd track this in the middleware
    
    return CompressionStatsResponse(
        compressed_responses=0,
        skipped_responses=0,
        total_bytes_before=0,
        total_bytes_after=0,
        compression_ratio=0,
        bytes_saved=0,
    )


@router.get(
    "/dashboard",
    response_model=PerformanceOptimizationDashboard,
    summary="Get performance dashboard",
    description="Get complete performance optimization dashboard",
)
@limiter.limit("30/minute")
async def get_performance_dashboard(request: Request) -> PerformanceOptimizationDashboard:
    """Get performance optimization dashboard"""
    cache_service = get_caching_service()
    profiler = get_performance_profiler()
    optimizer = get_query_optimizer()
    
    # Get all data
    cache_stats = cache_service.get_stats()
    profile_overview = profiler.get_overview()
    insights = profiler.get_performance_insights()
    
    # Determine status
    if profile_overview.get("slow_request_percentage", 0) > 20:
        status = "critical"
    elif profile_overview.get("slow_request_percentage", 0) > 10 or cache_stats["hit_rate"] < 50:
        status = "warning"
    else:
        status = "healthy"
    
    # Get recommendations
    recommendations = []
    if cache_stats["hit_rate"] < 50:
        recommendations.append("Increase cache TTL or adjust cache strategy")
    if profile_overview.get("slow_request_percentage", 0) > 10:
        recommendations.append("Review slowest endpoints - consider optimization")
    if optimizer.get_overview().get("slow_queries", 0) > 5:
        recommendations.append("Database queries need optimization - check indexes")
    
    return PerformanceOptimizationDashboard(
        status=status,
        performance_profile=PerformanceProfileResponse(
            total_requests=profile_overview.get("total_requests", 0),
            unique_endpoints=profile_overview.get("unique_endpoints", 0),
            avg_latency_ms=profile_overview.get("avg_latency_ms", 0),
            p95_latency_ms=profile_overview.get("p95_latency_ms", 0),
            p99_latency_ms=profile_overview.get("p99_latency_ms", 0),
            max_latency_ms=profile_overview.get("max_latency_ms", 0),
            slow_requests=profile_overview.get("slow_requests", 0),
            slow_request_percentage=profile_overview.get("slow_request_percentage", 0),
        ),
        cache_stats=CacheStatsResponse(
            backend=cache_stats["backend"],
            hit_count=cache_stats["hit_count"],
            miss_count=cache_stats["miss_count"],
            hit_rate=cache_stats["hit_rate"],
            total_entries=cache_stats["cache_stats"]["total_entries"],
            max_size=cache_stats["cache_stats"]["max_size"],
            expired_entries=cache_stats["cache_stats"]["expired_entries"],
        ),
        compression_stats=None,
        insights=[
            PerformanceInsight(
                type=i.get("type", "unknown"),
                severity=i.get("severity", "info"),
                endpoint=i.get("endpoint"),
                avg_duration_ms=i.get("avg_duration_ms"),
                query_count=i.get("query_count"),
                hit_rate=i.get("hit_rate"),
                recommendation=i.get("recommendation", ""),
            )
            for i in insights
        ],
        recommendations=recommendations,
    )


def setup_performance_routes(app):
    """Setup performance optimization routes"""
    app.include_router(router)
