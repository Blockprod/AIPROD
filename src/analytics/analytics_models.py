"""
Pydantic models for advanced analytics API
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ForecastPointResponse(BaseModel):
    """Single forecast point"""
    timestamp: datetime
    predicted_value: float
    confidence_interval_lower: float
    confidence_interval_upper: float

    model_config = ConfigDict(from_attributes=True)


class ForecastResponse(BaseModel):
    """Forecast response"""
    metric_name: str
    forecast_method: str
    last_known_value: float
    forecast_horizon: int
    forecasts: List[ForecastPointResponse]
    accuracy_score: float = Field(..., ge=0, le=100)
    trend_direction: str
    trend_strength: str

    model_config = ConfigDict(from_attributes=True)


class AnomalyResponse(BaseModel):
    """Single anomaly"""
    timestamp: datetime
    value: float
    expected_value: float
    severity: str
    type: str
    confidence: float = Field(..., ge=0, le=100)
    description: str

    model_config = ConfigDict(from_attributes=True)


class AnomalyReportResponse(BaseModel):
    """Anomaly detection report"""
    metric_name: str
    time_period: str
    anomalies: List[AnomalyResponse]
    total_points_analyzed: int
    anomaly_percentage: float
    dominant_anomaly_type: str

    model_config = ConfigDict(from_attributes=True)


class AnomalySummaryResponse(BaseModel):
    """Anomaly summary for metric"""
    metric_name: str
    total_anomalies: int
    anomaly_percentage: float
    anomaly_types: Dict[str, int]
    severity_breakdown: Dict[str, int]
    dominant_type: str
    recent_anomalies: List[Dict[str, Any]]

    model_config = ConfigDict(from_attributes=True)


class RegionCostAnalysisResponse(BaseModel):
    """Cost analysis for a region"""
    region_id: str
    region_name: str
    monthly_cost: float = Field(..., ge=0)
    monthly_requests: int = Field(..., ge=0)
    cost_per_request: float
    capacity_utilization: float = Field(..., ge=0, le=100)
    error_rate: float = Field(..., ge=0, le=100)
    avg_latency_ms: float = Field(..., ge=0)
    efficiency_score: float = Field(..., ge=0, le=100)

    model_config = ConfigDict(from_attributes=True)


class CostOpportunityResponse(BaseModel):
    """Cost optimization opportunity"""
    title: str
    description: str
    potential_savings_percentage: float = Field(..., ge=0, le=100)
    implementation_effort: str
    risk_level: str
    priority_score: float = Field(..., ge=0, le=100)
    estimated_monthly_savings: float = Field(..., ge=0)

    model_config = ConfigDict(from_attributes=True)


class CostOptimizationPlanResponse(BaseModel):
    """Complete cost optimization plan"""
    total_current_monthly_cost: float = Field(..., ge=0)
    total_potential_monthly_savings: float = Field(..., ge=0)
    savings_percentage: float = Field(..., ge=-100, le=100)
    opportunities: List[CostOpportunityResponse]
    regional_analysis: List[RegionCostAnalysisResponse]
    recommendations: List[str]
    quick_wins: List[CostOpportunityResponse]

    model_config = ConfigDict(from_attributes=True)


class CostSummaryResponse(BaseModel):
    """Cost optimization summary"""
    total_monthly_cost: float = Field(..., ge=0)
    potential_monthly_savings: float = Field(..., ge=0)
    savings_percentage: float = Field(..., ge=-100, le=100)
    number_of_opportunities: int = Field(..., ge=0)
    quick_wins_available: int = Field(..., ge=0)
    regions_analyzed: int = Field(..., ge=0)
    average_region_efficiency: float = Field(..., ge=0, le=100)
    top_recommendation: str

    model_config = ConfigDict(from_attributes=True)


class PredictiveFailoverAnalysisResponse(BaseModel):
    """Predictive failover analysis"""
    region_id: str
    region_name: str
    failure_probability: float = Field(..., ge=0, le=100)
    predicted_failure_time: Optional[datetime] = None
    current_health_score: float = Field(..., ge=0, le=100)
    recommended_action: str
    confidence: float = Field(..., ge=0, le=100)

    model_config = ConfigDict(from_attributes=True)


class PerformanceTrendResponse(BaseModel):
    """Performance trend analysis"""
    metric_name: str
    current_value: float
    previous_value: float
    trend_direction: str
    percent_change: float
    period: str
    recommendation: str

    model_config = ConfigDict(from_attributes=True)


class MLRecommendationResponse(BaseModel):
    """ML-based recommendation"""
    category: str
    priority: int = Field(..., ge=1, le=10)
    title: str
    description: str
    expected_impact: str
    implementation_complexity: str
    estimated_time_to_value: str

    model_config = ConfigDict(from_attributes=True)


class AdvancedAnalyticsDashboardResponse(BaseModel):
    """Complete advanced analytics dashboard"""
    timestamp: datetime
    forecast_summary: Dict[str, Any]
    anomaly_summary: Dict[str, Any]
    cost_optimization_summary: Dict[str, Any]
    predictive_insights: List[MLRecommendationResponse]
    performance_trends: List[PerformanceTrendResponse]
    system_health: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class AddMetricRequest(BaseModel):
    """Request to add metric data point"""
    metric_name: str = Field(..., min_length=1)
    value: float
    timestamp: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class AnalyticsHealthCheckResponse(BaseModel):
    """Analytics engine health check"""
    status: str
    total_metrics_tracked: int
    anomalies_detected: int
    forecasts_generated: int
    cost_analyses_run: int
    ml_models_ready: int

    model_config = ConfigDict(from_attributes=True)
