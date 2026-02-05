"""
Advanced analytics API endpoints
"""
import logging
from fastapi import APIRouter, Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import Optional

from src.analytics.anomaly_detector import AnomalyDetectionEngine
from src.analytics.forecaster import PerformanceForecaster
from src.analytics.cost_optimizer import CostOptimizer, OptimizationPriority
from src.analytics.analytics_models import (
    ForecastResponse,
    ForecastPointResponse,
    AnomalyReportResponse,
    AnomalyResponse,
    CostOptimizationPlanResponse,
    CostOpportunityResponse,
    RegionCostAnalysisResponse,
    CostSummaryResponse,
    AdvancedAnalyticsDashboardResponse,
    AddMetricRequest,
    AnalyticsHealthCheckResponse,
    PredictiveFailoverAnalysisResponse,
    PerformanceTrendResponse,
    MLRecommendationResponse,
)
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["advanced_analytics"])
limiter = Limiter(key_func=get_remote_address)

# Global instances
_anomaly_engine: Optional[AnomalyDetectionEngine] = None
_forecaster: Optional[PerformanceForecaster] = None
_cost_optimizer: Optional[CostOptimizer] = None


def get_anomaly_engine() -> AnomalyDetectionEngine:
    """Get or create anomaly detection engine"""
    global _anomaly_engine
    if _anomaly_engine is None:
        _anomaly_engine = AnomalyDetectionEngine()
        logger.info("Initialized anomaly detection engine")
    return _anomaly_engine


def get_forecaster() -> PerformanceForecaster:
    """Get or create performance forecaster"""
    global _forecaster
    if _forecaster is None:
        _forecaster = PerformanceForecaster()
        logger.info("Initialized performance forecaster")
    return _forecaster


def get_cost_optimizer() -> CostOptimizer:
    """Get or create cost optimizer"""
    global _cost_optimizer
    if _cost_optimizer is None:
        _cost_optimizer = CostOptimizer()
        logger.info("Initialized cost optimizer")
    return _cost_optimizer


@router.post(
    "/metrics/add",
    tags=["metrics"],
    summary="Add metric data point",
    description="Add new data point to metric for analysis",
    response_model=dict,
)
@limiter.limit("100/minute")
async def add_metric(request: Request, data: AddMetricRequest) -> dict:
    """Add data point to metric for ML analysis"""
    try:
        engine = get_anomaly_engine()
        forecaster = get_forecaster()

        engine.add_data_point(data.metric_name, data.value, data.timestamp)
        forecaster.add_data_point(data.metric_name, data.value, data.timestamp)

        return {
            "status": "success",
            "metric_name": data.metric_name,
            "value": data.value,
            "timestamp": data.timestamp or datetime.utcnow(),
        }
    except Exception as e:
        logger.error(f"Error adding metric: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding metric: {str(e)}")


@router.get(
    "/forecast/{metric_name}",
    tags=["forecasting"],
    summary="Forecast metric performance",
    description="Generate performance forecast using ensemble ML methods",
    response_model=ForecastResponse,
)
@limiter.limit("50/minute")
async def forecast_metric(
    request: Request,
    metric_name: str,
    periods_ahead: int = 10,
    method: str = "ensemble",
) -> ForecastResponse:
    """Forecast metric using ML models"""
    try:
        forecaster = get_forecaster()

        if method == "linear":
            result = forecaster.forecast_linear(metric_name, periods_ahead)
        elif method == "exponential":
            result = forecaster.forecast_exponential_smoothing(metric_name, periods_ahead)
        elif method == "seasonal":
            result = forecaster.forecast_seasonal(metric_name, periods_ahead)
        else:
            result = forecaster.ensemble_forecast(metric_name, periods_ahead)

        if not result.forecasts:
            raise HTTPException(status_code=404, detail=f"Insufficient data for {metric_name}")

        return ForecastResponse(
            metric_name=result.metric_name,
            forecast_method=result.forecast_method,
            last_known_value=result.last_known_value,
            forecast_horizon=result.forecast_horizon,
            forecasts=[
                ForecastPointResponse(
                    timestamp=f.timestamp,
                    predicted_value=f.predicted_value,
                    confidence_interval_lower=f.confidence_interval_lower,
                    confidence_interval_upper=f.confidence_interval_upper,
                )
                for f in result.forecasts
            ],
            accuracy_score=result.accuracy_score,
            trend_direction=result.trend_direction,
            trend_strength=result.trend_strength,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forecasting {metric_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")


@router.get(
    "/anomalies/{metric_name}",
    tags=["anomaly_detection"],
    summary="Detect anomalies",
    description="Detect anomalies in metric using statistical and ML methods",
    response_model=AnomalyReportResponse,
)
@limiter.limit("50/minute")
async def detect_anomalies(request: Request, metric_name: str, lookback_points: int = 100) -> AnomalyReportResponse:
    """Detect anomalies in metric"""
    try:
        engine = get_anomaly_engine()
        report = engine.detect_anomalies(metric_name, lookback_points)

        return AnomalyReportResponse(
            metric_name=report.metric_name,
            time_period=report.time_period,
            anomalies=[
                AnomalyResponse(
                    timestamp=a.timestamp,
                    value=a.value,
                    expected_value=a.expected_value,
                    severity=a.severity,
                    type=a.type,
                    confidence=a.confidence,
                    description=a.description,
                )
                for a in report.anomalies
            ],
            total_points_analyzed=report.total_points_analyzed,
            anomaly_percentage=report.anomaly_percentage,
            dominant_anomaly_type=report.dominant_anomaly_type,
        )
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail=f"Error detecting anomalies: {str(e)}")


@router.get(
    "/anomalies/{metric_name}/summary",
    tags=["anomaly_detection"],
    summary="Get anomaly summary",
    description="Get summary of detected anomalies",
    response_model=dict,
)
@limiter.limit("50/minute")
async def get_anomaly_summary(request: Request, metric_name: str) -> dict:
    """Get anomaly summary for metric"""
    try:
        engine = get_anomaly_engine()
        summary = engine.get_anomaly_summary(metric_name)
        return summary
    except Exception as e:
        logger.error(f"Error getting anomaly summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")


@router.get(
    "/costs/analyze",
    tags=["cost_optimization"],
    summary="Analyze costs",
    description="Analyze current costs and generate optimization plan",
    response_model=CostOptimizationPlanResponse,
)
@limiter.limit("30/minute")
async def analyze_costs(request: Request, priority: str = "balanced") -> CostOptimizationPlanResponse:
    """Generate cost optimization plan"""
    try:
        optimizer = get_cost_optimizer()

        # Convert priority string to enum
        priority_enum = {
            "cost": OptimizationPriority.COST,
            "performance": OptimizationPriority.PERFORMANCE,
            "balanced": OptimizationPriority.BALANCED,
        }.get(priority, OptimizationPriority.BALANCED)

        plan = optimizer.generate_cost_plan(priority_enum)

        return CostOptimizationPlanResponse(
            total_current_monthly_cost=plan.total_current_monthly_cost,
            total_potential_monthly_savings=plan.total_potential_monthly_savings,
            savings_percentage=plan.savings_percentage,
            opportunities=[
                CostOpportunityResponse(
                    title=o.title,
                    description=o.description,
                    potential_savings_percentage=o.potential_savings_percentage,
                    implementation_effort=o.implementation_effort,
                    risk_level=o.risk_level,
                    priority_score=o.priority_score,
                    estimated_monthly_savings=o.estimated_monthly_savings,
                )
                for o in plan.opportunities
            ],
            regional_analysis=[
                RegionCostAnalysisResponse(
                    region_id=r.region_id,
                    region_name=r.region_name,
                    monthly_cost=r.monthly_cost,
                    monthly_requests=r.monthly_requests,
                    cost_per_request=r.cost_per_request,
                    capacity_utilization=r.capacity_utilization,
                    error_rate=r.error_rate,
                    avg_latency_ms=r.avg_latency_ms,
                    efficiency_score=r.efficiency_score,
                )
                for r in plan.regional_analysis
            ],
            recommendations=plan.recommendations,
            quick_wins=[
                CostOpportunityResponse(
                    title=w.title,
                    description=w.description,
                    potential_savings_percentage=w.potential_savings_percentage,
                    implementation_effort=w.implementation_effort,
                    risk_level=w.risk_level,
                    priority_score=w.priority_score,
                    estimated_monthly_savings=w.estimated_monthly_savings,
                )
                for w in plan.quick_wins
            ],
        )
    except Exception as e:
        logger.error(f"Error analyzing costs: {e}")
        raise HTTPException(status_code=500, detail=f"Error analyzing costs: {str(e)}")


@router.get(
    "/costs/summary",
    tags=["cost_optimization"],
    summary="Cost optimization summary",
    description="Get quick summary of cost optimization status",
    response_model=CostSummaryResponse,
)
@limiter.limit("50/minute")
async def get_cost_summary(request: Request) -> CostSummaryResponse:
    """Get cost optimization summary"""
    try:
        optimizer = get_cost_optimizer()
        summary = optimizer.get_cost_summary()

        return CostSummaryResponse(
            total_monthly_cost=summary["total_monthly_cost"],
            potential_monthly_savings=summary["potential_monthly_savings"],
            savings_percentage=summary["savings_percentage"],
            number_of_opportunities=summary["number_of_opportunities"],
            quick_wins_available=summary["quick_wins_available"],
            regions_analyzed=summary["regions_analyzed"],
            average_region_efficiency=summary["average_region_efficiency"],
            top_recommendation=summary["top_recommendation"],
        )
    except Exception as e:
        logger.error(f"Error getting cost summary: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")


@router.get(
    "/dashboard",
    tags=["dashboard"],
    summary="Advanced analytics dashboard",
    description="Get complete advanced analytics dashboard",
    response_model=dict,
)
@limiter.limit("30/minute")
async def get_analytics_dashboard(request: Request) -> dict:
    """Get complete analytics dashboard"""
    try:
        anomaly_engine = get_anomaly_engine()
        forecaster = get_forecaster()
        optimizer = get_cost_optimizer()

        # Collect data from all sources
        forecast_summary = {"status": "ready", "metrics_tracked": 0}
        anomaly_summary = {"total_detected": 0, "critical_anomalies": 0}
        cost_summary = optimizer.get_cost_summary()

        return {
            "timestamp": datetime.utcnow(),
            "forecast_summary": forecast_summary,
            "anomaly_summary": anomaly_summary,
            "cost_optimization_summary": cost_summary,
            "predictive_insights": [],
            "performance_trends": [],
            "system_health": {"ml_engine": "operational", "data_sources": "connected"},
        }
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting dashboard: {str(e)}")


@router.get(
    "/health",
    tags=["system"],
    summary="Analytics engine health check",
    description="Check health and status of analytics engines",
    response_model=AnalyticsHealthCheckResponse,
)
@limiter.limit("100/minute")
async def health_check(request: Request) -> AnalyticsHealthCheckResponse:
    """Check analytics engine health"""
    try:
        anomaly_engine = get_anomaly_engine()
        forecaster = get_forecaster()
        optimizer = get_cost_optimizer()

        return AnalyticsHealthCheckResponse(
            status="operational",
            total_metrics_tracked=len(anomaly_engine.history),
            anomalies_detected=sum(len(anomaly_engine.detect_anomalies(m).anomalies) for m in anomaly_engine.history.keys()) if anomaly_engine.history else 0,
            forecasts_generated=len(forecaster.history),
            cost_analyses_run=len(optimizer.region_costs),
            ml_models_ready=4,
        )
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=f"Error in health check: {str(e)}")


def setup_analytics_routes(app):
    """Setup advanced analytics routes"""
    app.include_router(router)
    logger.info("Advanced analytics routes configured")
