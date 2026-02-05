"""
Deployment Routes - API endpoints for multi-region deployment management
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List, Dict, Any
from src.deployment.region_manager import get_region_manager, RegionTier
from src.deployment.failover_manager import (
    get_failover_manager,
    FailoverStrategy,
    FailoverPolicy,
    FailoverTrigger,
)
from src.deployment.deployment_models import (
    RegionMetricsData,
    RegionComparisonResponse,
    CapacityAnalysis,
    MultiRegionOverview,
    FailoverStatus,
    FailoverAnalytics,
    MultiRegionDashboard,
    RegisterRegionRequest,
    InitiateFailoverRequest,
    SetTrafficDistributionRequest,
)

router = APIRouter(prefix="/deployment", tags=["deployment"])
limiter = Limiter(key_func=get_remote_address)


@router.post(
    "/regions/register",
    response_model=Dict[str, str],
    summary="Register new region",
    description="Register a new geographic region for deployment",
)
@limiter.limit("10/minute")
async def register_region(
    request: Request,
    region_req: RegisterRegionRequest,
) -> Dict[str, str]:
    """Register a new region"""
    manager = get_region_manager()
    
    tier = RegionTier[region_req.tier.upper()]
    region_id = manager.register_region(
        region_name=region_req.region_name,
        endpoint=region_req.endpoint,
        tier=tier,
        max_capacity=region_req.max_capacity,
    )
    
    return {
        "region_id": region_id,
        "region_name": region_req.region_name,
        "status": "registered"
    }


@router.get(
    "/regions",
    response_model=List[RegionMetricsData],
    summary="Get all regions",
    description="Get metrics for all registered regions",
)
@limiter.limit("100/minute")
async def get_all_regions(
    request: Request,
    healthy_only: bool = Query(False, description="Only return healthy regions")
) -> List[RegionMetricsData]:
    """Get all regions"""
    manager = get_region_manager()
    
    regions = manager.get_healthy_regions() if healthy_only else manager.get_all_regions()
    
    return [
        RegionMetricsData(
            region_id=r.region_id,
            region_name=r.region_name,
            tier=r.tier.value,
            status=r.status.value,
            latency_ms=r.metrics.latency_ms,
            available_capacity=r.metrics.available_capacity,
            error_rate=r.metrics.error_rate,
            request_count=r.metrics.request_count,
            uptime_percentage=r.metrics.uptime_percentage,
            p95_latency=r.metrics.response_time_p95,
            p99_latency=r.metrics.response_time_p99,
        )
        for r in regions
    ]


@router.get(
    "/regions/{region_id}",
    response_model=RegionMetricsData,
    summary="Get region details",
    description="Get detailed metrics for a specific region",
)
@limiter.limit("100/minute")
async def get_region(request: Request, region_id: str) -> RegionMetricsData:
    """Get regional details"""
    manager = get_region_manager()
    region = manager.get_region(region_id)
    
    if not region:
        raise HTTPException(status_code=404, detail="Region not found")
    
    return RegionMetricsData(
        region_id=region.region_id,
        region_name=region.region_name,
        tier=region.tier.value,
        status=region.status.value,
        latency_ms=region.metrics.latency_ms,
        available_capacity=region.metrics.available_capacity,
        error_rate=region.metrics.error_rate,
        request_count=region.metrics.request_count,
        uptime_percentage=region.metrics.uptime_percentage,
        p95_latency=region.metrics.response_time_p95,
        p99_latency=region.metrics.response_time_p99,
    )


@router.get(
    "/comparison",
    response_model=RegionComparisonResponse,
    summary="Compare regions",
    description="Get performance comparison across all regions",
)
@limiter.limit("50/minute")
async def compare_regions(request: Request) -> RegionComparisonResponse:
    """Compare regions"""
    manager = get_region_manager()
    comparison = manager.get_regional_comparison()
    
    regions_data = [
        RegionMetricsData(
            region_id=r["region_id"],
            region_name=r["region_name"],
            tier=r["tier"],
            status=r["status"],
            latency_ms=r["latency_ms"],
            available_capacity=r["available_capacity"],
            error_rate=r["error_rate"],
            request_count=r["request_count"],
            uptime_percentage=r["uptime_percentage"],
            p95_latency=r.get("p95_latency", 0),
            p99_latency=r.get("p99_latency", 0),
        )
        for r in comparison["regions"]
    ]
    
    best = RegionMetricsData(**comparison["best_performing"]) if comparison["best_performing"] else None
    worst = RegionMetricsData(**comparison["worst_performing"]) if comparison["worst_performing"] else None
    
    return RegionComparisonResponse(
        total_regions=comparison["total_regions"],
        healthy_regions=comparison["healthy_regions"],
        primary_regions=comparison["primary_regions"],
        regions=regions_data,
        best_performing=best,
        worst_performing=worst,
    )


@router.get(
    "/capacity",
    response_model=CapacityAnalysis,
    summary="Get capacity analysis",
    description="Analyze capacity utilization across regions",
)
@limiter.limit("50/minute")
async def get_capacity(request: Request) -> CapacityAnalysis:
    """Get capacity analysis"""
    manager = get_region_manager()
    capacity = manager.get_capacity_analysis()
    
    return CapacityAnalysis(
        total_capacity=int(capacity["total_capacity"]),
        used_capacity=capacity["used_capacity"],
        available_capacity=capacity["available_capacity"],
        utilization_percentage=capacity["utilization_percentage"],
    )


@router.get(
    "/overview",
    response_model=MultiRegionOverview,
    summary="Get multi-region overview",
    description="Get comprehensive multi-region status overview",
)
@limiter.limit("50/minute")
async def get_overview(request: Request) -> MultiRegionOverview:
    """Get multi-region overview"""
    manager = get_region_manager()
    overview = manager.get_overview()
    
    return MultiRegionOverview(
        total_regions=overview["total_regions"],
        healthy_regions=overview["healthy_regions"],
        degraded_regions=overview.get("degraded_regions", 0),
        unhealthy_regions=overview.get("unhealthy_regions", 0),
        average_error_rate=overview["average_error_rate"],
        average_latency_ms=overview["average_latency_ms"],
        health_percentage=overview["health_percentage"],
    )


@router.post(
    "/failover",
    response_model=Dict[str, Any],
    summary="Initiate failover",
    description="Manually initiate failover between regions",
)
@limiter.limit("10/minute")
async def initiate_failover(
    request: Request,
    failover_req: InitiateFailoverRequest,
) -> Dict[str, Any]:
    """Initiate failover"""
    failover_mgr = get_failover_manager()
    
    success = await failover_mgr.initiate_failover(
        from_region=failover_req.from_region,
        to_region=failover_req.to_region,
        trigger=FailoverTrigger.MANUAL,
        reason=failover_req.reason or "Manual failover",
    )
    
    return {
        "success": success,
        "from_region": failover_req.from_region,
        "to_region": failover_req.to_region,
    }


@router.get(
    "/failover/status",
    response_model=FailoverStatus,
    summary="Get failover status",
    description="Get current failover status",
)
@limiter.limit("100/minute")
async def get_failover_status(request: Request) -> FailoverStatus:
    """Get failover status"""
    failover_mgr = get_failover_manager()
    status = failover_mgr.get_failover_status()
    
    return FailoverStatus(
        active_failovers=status["active_failovers"],
        total_failover_events=status["total_failover_events"],
        successful_failovers=status["successful_failovers"],
        failed_failovers=status["failed_failovers"],
    )


@router.get(
    "/failover/history",
    response_model=List[Dict[str, Any]],
    summary="Get failover history",
    description="Get recent failover events",
)
@limiter.limit("50/minute")
async def get_failover_history(
    request: Request,
    limit: int = Query(20, ge=1, le=100)
) -> List[Dict[str, Any]]:
    """Get failover history"""
    failover_mgr = get_failover_manager()
    return failover_mgr.get_failover_history(limit)


@router.get(
    "/failover/analytics",
    response_model=FailoverAnalytics,
    summary="Get failover analytics",
    description="Get failover statistics and analytics",
)
@limiter.limit("30/minute")
async def get_failover_analytics(request: Request) -> FailoverAnalytics:
    """Get failover analytics"""
    failover_mgr = get_failover_manager()
    analytics = failover_mgr.get_failover_analytics()
    
    return FailoverAnalytics(
        total_events=analytics["total_events"],
        successful_events=analytics["successful_events"],
        success_rate=analytics["success_rate"],
        average_duration_seconds=analytics["average_duration_seconds"],
        most_common_trigger=analytics["most_common_trigger"],
        trigger_breakdown=analytics["trigger_breakdown"],
    )


@router.get(
    "/traffic",
    summary="Get traffic distribution",
    description="Get current traffic distribution across regions",
)
@limiter.limit("100/minute")
async def get_traffic_distribution(request: Request) -> Dict[str, float]:
    """Get traffic distribution"""
    failover_mgr = get_failover_manager()
    return failover_mgr.get_traffic_distribution()


@router.post(
    "/traffic",
    response_model=Dict[str, str],
    summary="Set traffic distribution",
    description="Manually set traffic distribution across regions",
)
@limiter.limit("10/minute")
async def set_traffic_distribution(
    request: Request,
    traffic_req: SetTrafficDistributionRequest,
) -> Dict[str, str]:
    """Set traffic distribution"""
    failover_mgr = get_failover_manager()
    
    success = failover_mgr.set_traffic_distribution(traffic_req.distribution)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Invalid traffic distribution (must sum to 100%)"
        )
    
    return {
        "status": "success",
        "message": "Traffic distribution updated"
    }


@router.get(
    "/dashboard",
    response_model=MultiRegionDashboard,
    summary="Get multi-region dashboard",
    description="Get complete multi-region deployment dashboard",
)
@limiter.limit("30/minute")
async def get_dashboard(request: Request) -> MultiRegionDashboard:
    """Get multi-region dashboard"""
    manager = get_region_manager()
    failover_mgr = get_failover_manager()
    
    # Get all components
    overview_data = manager.get_overview()
    capacity_data = manager.get_capacity_analysis()
    failover_status = failover_mgr.get_failover_status()
    comparison = manager.get_regional_comparison()
    
    # Get top performing regions
    top_regions = [
        RegionMetricsData(
            region_id=r["region_id"],
            region_name=r["region_name"],
            tier=r["tier"],
            status=r["status"],
            latency_ms=r["latency_ms"],
            available_capacity=r["available_capacity"],
            error_rate=r["error_rate"],
            request_count=r["request_count"],
            uptime_percentage=r["uptime_percentage"],
            p95_latency=r.get("p95_latency", 0),
            p99_latency=r.get("p99_latency", 0),
        )
        for r in comparison["regions"][:3]
    ]
    
    # Determine status
    health_pct = overview_data["health_percentage"]
    if health_pct >= 95:
        status = "healthy"
    elif health_pct >= 80:
        status = "warning"
    else:
        status = "critical"
    
    # Generate recommendations
    recommendations = []
    if capacity_data["utilization_percentage"] > 80:
        recommendations.append("High capacity utilization - consider scaling")
    if overview_data["average_error_rate"] > 1:
        recommendations.append("Elevated error rate detected - investigate failed requests")
    if failover_status["active_failovers"]:
        recommendations.append("Active failover in progress - monitor recovery")
    if overview_data.get("degraded_regions", 0) > 0:
        recommendations.append("Degraded regions detected - check health metrics")
    
    return MultiRegionDashboard(
        status=status,
        overview=MultiRegionOverview(
            total_regions=overview_data["total_regions"],
            healthy_regions=overview_data["healthy_regions"],
            degraded_regions=overview_data.get("degraded_regions", 0),
            unhealthy_regions=overview_data.get("unhealthy_regions", 0),
            average_error_rate=overview_data["average_error_rate"],
            average_latency_ms=overview_data["average_latency_ms"],
            health_percentage=overview_data["health_percentage"],
        ),
        capacity=CapacityAnalysis(
            total_capacity=int(capacity_data["total_capacity"]),
            used_capacity=capacity_data["used_capacity"],
            available_capacity=capacity_data["available_capacity"],
            utilization_percentage=capacity_data["utilization_percentage"],
        ),
        failover_status=FailoverStatus(
            active_failovers=failover_status["active_failovers"],
            total_failover_events=failover_status["total_failover_events"],
            successful_failovers=failover_status["successful_failovers"],
            failed_failovers=failover_status["failed_failovers"],
        ),
        top_performing_regions=top_regions,
        recommendations=recommendations,
    )


def setup_deployment_routes(app):
    """Setup deployment routes"""
    app.include_router(router)
