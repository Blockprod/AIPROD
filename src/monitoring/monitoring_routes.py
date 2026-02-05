"""
Monitoring Endpoints - API routes for monitoring and analytics
Advanced monitoring dashboard data, metrics, alerts, and analytics
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Query, Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from src.monitoring.metrics_collector import get_metrics_collector, MetricsCollector
from src.monitoring.analytics_engine import get_analytics_engine, AnalyticsEngine
from src.monitoring.monitoring_models import (
    EndpointStatsResponse,
    HealthStatusResponse,
    AlertResponse,
    DashboardMetrics,
    PerformanceInsight,
    MetricHistoryRequest,
    MetricHistoryResponse,
    AlertResolutionRequest,
    AnomalyDetectionResponse,
    AnomalyReport,
    CorrelationAnalysis,
)
from src.utils.monitoring import logger

router = APIRouter(prefix="/monitoring", tags=["monitoring"])
limiter = Limiter(key_func=get_remote_address)



@router.get(
    "/health",
    response_model=HealthStatusResponse,
    summary="Get system health status",
    description="Returns overall system health metrics and statistics",
)
@limiter.limit("100/minute")
async def get_health(request: Request) -> Dict[str, Any]:
    """
    Get current system health status
    
    Returns:
    - `status`: Health status (healthy, warning, critical)
    - `error_rate`: Percentage of requests with errors
    - `active_alerts`: Number of active alerts
    """
    try:
        collector = get_metrics_collector()
        health = collector.get_health_status()
        return health
    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving health status")


@router.get(
    "/dashboard",
    response_model=DashboardMetrics,
    summary="Get monitoring dashboard",
    description="Complete dashboard with health, top endpoints, alerts, and insights",
)
@limiter.limit("50/minute")
async def get_dashboard(request: Request) -> Dict[str, Any]:
    """
    Get complete monitoring dashboard
    
    Includes:
    - System health
    - Top endpoints by error rate/latency
    - Active alerts
    - Performance insights
    """
    try:
        collector = get_metrics_collector()
        analytics = get_analytics_engine()
        
        health = collector.get_health_status()
        top_endpoints_raw = collector.get_top_endpoints(limit=10)
        alerts = collector.get_alerts(unresolved_only=True)
        
        # Convert endpoints to response model
        top_endpoints = []
        for ep in top_endpoints_raw:
            stats = collector.get_endpoint_stats(ep["endpoint"])
            insights = analytics.get_performance_insights(stats)
            
            # Create endpoint response
            ep_response = EndpointStatsResponse(
                endpoint=ep["endpoint"],
                method=ep["endpoint"].split()[0],
                request_count=ep["requests"],
                error_count=ep["errors"],
                error_rate=ep["error_rate"] / 100,
                avg_duration=ep["avg_latency"],
                min_duration=0,
                max_duration=ep["max_latency"],
                avg_size=0,
            )
            top_endpoints.append(ep_response)
        
        # Generate insights
        insights_models = []
        stats = collector.get_endpoint_stats()
        
        for endpoint, endpoint_stats in list(stats.items())[:5]:
            insights = analytics.get_performance_insights(endpoint_stats)
            for insight in insights:
                if "Critical" in insight:
                    severity = "critical"
                elif "Warning" in insight or "HIGH" in insight or "High" in insight:
                    severity = "warning"
                elif "Excellent" in insight:
                    severity = "info"
                else:
                    severity = "info"
                
                category = "latency" if "latency" in insight.lower() else "performance"
                
                insights_models.append(PerformanceInsight(
                    category=category,
                    insight=insight,
                    severity=severity,
                ))
        
        return {
            "health": health,
            "top_endpoints": top_endpoints,
            "active_alerts": alerts,
            "insights": insights_models[:5],
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail="Error generating dashboard")


@router.get(
    "/endpoints/{endpoint_path:path}",
    summary="Get statistics for specific endpoint",
    description="Detailed statistics and metrics for a single endpoint",
)
@limiter.limit("100/minute")
async def get_endpoint_stats(
    request: Request,
    endpoint_path: str,
    method: Optional[str] = Query("GET"),
) -> Dict[str, Any]:
    """
    Get detailed statistics for a specific endpoint
    
    Args:
    - `endpoint_path`: The endpoint path
    - `method`: HTTP method (GET, POST, etc)
    
    Returns:
    - Request count, error rate, latency stats
    """
    try:
        collector = get_metrics_collector()
        key = f"{method} /{endpoint_path}" if not endpoint_path.startswith("/") else f"{method} {endpoint_path}"
        
        stats = collector.get_endpoint_stats(key)
        
        if not stats:
            raise HTTPException(status_code=404, detail=f"No stats found for {key}")
        
        return stats
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting endpoint stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving endpoint statistics")


@router.get(
    "/endpoints",
    summary="Get statistics for all endpoints",
    description="Statistics for all monitored endpoints sorted by error rate",
)
@limiter.limit("50/minute")
async def get_all_endpoints(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    sort_by: str = Query("error_rate", pattern="^(error_rate|latency|requests)$"),
) -> List[Dict[str, Any]]:
    """
    Get statistics for all endpoints
    
    Args:
    - `limit`: Maximum endpoints to return
    - `sort_by`: Sort by "error_rate", "latency", or "requests"
    
    Returns:
    - List of endpoint statistics
    """
    try:
        collector = get_metrics_collector()
        endpoints = collector.get_top_endpoints(limit=limit)
        
        # Sort by requested field
        if sort_by == "latency":
            endpoints.sort(key=lambda x: x["avg_latency"], reverse=True)
        elif sort_by == "requests":
            endpoints.sort(key=lambda x: x["requests"], reverse=True)
        
        return endpoints
    
    except Exception as e:
        logger.error(f"Error getting all endpoints: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving endpoint statistics")


@router.get(
    "/alerts",
    response_model=List[AlertResponse],
    summary="Get active alerts",
    description="List of all active alerts with details",
)
@limiter.limit("30/minute")
async def get_alerts_monitoring(
    request: Request,
    unresolved_only: bool = Query(True),
) -> List[Dict[str, Any]]:
    """
    Get list of alerts
    
    Args:
    - `unresolved_only`: Only show unresolved alerts
    
    Returns:
    - List of alert details
    """
    try:
        collector = get_metrics_collector()
        alerts = collector.get_alerts(unresolved_only=unresolved_only)
        return alerts
    
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving alerts")


@router.post(
    "/alerts/{alert_id}/resolve",
    summary="Resolve an alert",
    description="Mark an alert as resolved",
)
@limiter.limit("20/minute")
async def resolve_alert(
    request: Request,
    alert_id: str,
    body: AlertResolutionRequest,
) -> Dict[str, str]:
    """
    Resolve an alert
    
    Args:
    - `alert_id`: ID of alert to resolve
    - `body`: Resolution details
    
    Returns:
    - Confirmation message
    """
    try:
        collector = get_metrics_collector()
        collector.resolve_alert(alert_id)
        return {"message": f"Alert {alert_id} resolved"}
    
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        raise HTTPException(status_code=500, detail="Error resolving alert")


@router.post(
    "/anomalies/detect",
    response_model=AnomalyDetectionResponse,
    summary="Detect anomalies in metrics",
    description="Run anomaly detection on recent metrics",
)
@limiter.limit("10/minute")
async def detect_anomalies(
    request: Request,
    time_period: str = Query("last_hour", pattern="^(last_hour|last_24h|last_week)$"),
) -> Dict[str, Any]:
    """
    Detect anomalies in metrics
    
    Args:
    - `time_period`: Analysis period
    
    Returns:
    - List of detected anomalies and recommendations
    """
    try:
        collector = get_metrics_collector()
        analytics = get_analytics_engine()
        
        # Get recent latency metrics
        latency_metrics = collector.metrics.get("endpoint_latency", [])
        
        if not latency_metrics:
            return {
                "anomalies_detected": [],
                "timestamp": datetime.utcnow().isoformat(),
                "analysis_period": time_period,
                "recommendations": ["Not enough data collected yet"],
            }
        
        # Extract values
        values = [m.value for m in latency_metrics[-100:]]
        
        # Detect anomalies
        anomalies = analytics.detect_latency_anomalies(values)
        
        # Get recommendations
        health = collector.get_health_status()
        stats = collector.get_endpoint_stats()
        recommendations = analytics.get_recommendations(stats, health)
        
        anomaly_reports = [
            AnomalyReport(
                type=a["type"],
                value=a["value"],
                expected_range=a["expected_range"],
                deviation=a["deviation"],
            )
            for a in anomalies
        ]
        
        return {
            "anomalies_detected": anomaly_reports,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis_period": time_period,
            "recommendations": recommendations,
        }
    
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail="Error detecting anomalies")


@router.get(
    "/trends/{metric_name}",
    summary="Get trend analysis for a metric",
    description="Analyze trends in a specific metric over time",
)
@limiter.limit("30/minute")
async def get_trend(
    request: Request,
    metric_name: str,
    window: int = Query(5, ge=2, le=50),
) -> Dict[str, Any]:
    """
    Get trend analysis for a metric
    
    Args:
    - `metric_name`: Name of metric to analyze
    - `window`: Window size for moving average
    
    Returns:
    - Trend analysis with direction and change percentage
    """
    try:
        collector = get_metrics_collector()
        analytics = get_analytics_engine()
        
        # Get metric values
        metrics = collector.metrics.get(metric_name, [])
        
        if not metrics:
            raise HTTPException(status_code=404, detail=f"No metrics found for {metric_name}")
        
        values = [m.value for m in metrics[-100:]]
        
        if len(values) < 2:
            raise HTTPException(status_code=400, detail="Not enough data for trend analysis")
        
        # Calculate trend
        trend = analytics.calculate_trend(values, window=window)
        
        return trend
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing trend: {e}")
        raise HTTPException(status_code=500, detail="Error analyzing trend")


@router.post(
    "/cleanup",
    summary="Clean up old metrics",
    description="Remove metrics older than retention period",
)
@limiter.limit("5/minute")
async def cleanup_metrics(request: Request) -> Dict[str, str]:
    """
    Trigger cleanup of old metrics data
    
    Returns:
    - Confirmation message
    """
    try:
        collector = get_metrics_collector()
        before_count = sum(len(m) for m in collector.metrics.values())
        
        collector.cleanup_old_metrics()
        
        after_count = sum(len(m) for m in collector.metrics.values())
        removed = before_count - after_count
        
        return {
            "message": "Cleanup completed",
            "metrics_removed": str(removed),
        }
    
    except Exception as e:
        logger.error(f"Error cleaning up metrics: {e}")
        raise HTTPException(status_code=500, detail="Error cleaning up metrics")


@router.get(
    "/recommendations",
    summary="Get optimization recommendations",
    description="AI-generated recommendations to improve system performance",
)
@limiter.limit("20/minute")
async def get_recommendations(request: Request) -> Dict[str, Any]:
    """
    Get systems recommendations for optimization
    
    Returns:
    - List of personalized recommendations
    """
    try:
        collector = get_metrics_collector()
        analytics = get_analytics_engine()
        
        health = collector.get_health_status()
        stats = collector.get_endpoint_stats()
        recommendations = analytics.get_recommendations(stats, health)
        
        return {
            "recommendations": recommendations,
            "health_status": health["status"],
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Error generating recommendations")



# Include router in main app
def setup_monitoring_routes(app):
    """Setup monitoring routes on main app"""
    app.include_router(router)
