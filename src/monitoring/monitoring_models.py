"""
Monitoring Models - Pydantic data models for monitoring endpoints
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


class EndpointStatsResponse(BaseModel):
    """Statistics for an endpoint"""
    endpoint: str
    method: str
    request_count: int = Field(..., ge=0)
    error_count: int = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=1)
    avg_duration: float = Field(..., ge=0)
    min_duration: float = Field(..., ge=0)
    max_duration: float = Field(..., ge=0)
    avg_size: float = Field(..., ge=0)
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "endpoint": "GET /api/jobs",
            "method": "GET",
            "request_count": 1250,
            "error_count": 5,
            "error_rate": 0.004,
            "avg_duration": 45.3,
            "min_duration": 10.2,
            "max_duration": 250.8,
            "avg_size": 5120,
        }
    })


class HealthStatusResponse(BaseModel):
    """System health status"""
    status: str = Field(..., description="Health status: healthy, warning, critical")
    uptime_seconds: float = Field(..., ge=0)
    total_requests: int = Field(..., ge=0)
    total_errors: int = Field(..., ge=0)
    error_rate: float = Field(..., ge=0, le=100)
    endpoints_count: int = Field(..., ge=0)
    active_alerts: int = Field(..., ge=0)
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "healthy",
            "uptime_seconds": 3600.5,
            "total_requests": 10000,
            "total_errors": 50,
            "error_rate": 0.5,
            "endpoints_count": 25,
            "active_alerts": 0,
        }
    })


class AlertResponse(BaseModel):
    """Alert response model"""
    id: str
    severity: str = Field(..., description="Alert severity: info, warning, critical")
    title: str
    message: str
    metric: str
    threshold: float
    current_value: float
    timestamp: str
    resolved: bool = False
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "id": "endpoint_latency_1707148800000",
            "severity": "warning",
            "title": "High latency on GET /api/jobs",
            "message": "Request took 1250ms (threshold: 1000ms)",
            "metric": "endpoint_latency",
            "threshold": 1000,
            "current_value": 1250,
            "timestamp": "2026-02-05T12:00:00",
            "resolved": False,
        }
    })


class AnomalyReport(BaseModel):
    """Anomaly detection report"""
    type: str = Field(..., description="Type of anomaly detected")
    value: float = Field(..., description="Anomalous value")
    expected_range: str = Field(..., description="Expected value range")
    deviation: float = Field(..., description="Standard deviations from mean")
    timestamp: Optional[str] = None
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "type": "latency_spike",
            "value": 2500.5,
            "expected_range": "45.3±50.2",
            "deviation": 44.3,
            "timestamp": "2026-02-05T12:05:00",
        }
    })


class TrendAnalysis(BaseModel):
    """Trend analysis for a metric"""
    trend: str = Field(..., description="Trend direction: increasing, decreasing, unknown")
    direction: Optional[str] = Field(..., description="Direction: up, down, None")
    change_percent: float = Field(..., description="Percentage change")
    current: float = Field(..., description="Current value")
    average: float = Field(..., description="Average value")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "trend": "increasing",
            "direction": "up",
            "change_percent": 15.5,
            "current": 52.3,
            "average": 45.2,
        }
    })


class PerformanceInsight(BaseModel):
    """Performance insight message"""
    category: str = Field(..., description="Category: error, latency, performance, etc")
    insight: str = Field(..., description="Insight message")
    severity: str = Field(..., description="Severity: info, warning, critical")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "category": "latency",
            "insight": "High average latency: 250ms",
            "severity": "warning",
        }
    })


class DashboardMetrics(BaseModel):
    """Complete dashboard metrics"""
    health: HealthStatusResponse
    top_endpoints: List[EndpointStatsResponse]
    active_alerts: List[AlertResponse]
    insights: List[PerformanceInsight]
    timestamp: str
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "health": {
                "status": "healthy",
                "uptime_seconds": 3600.5,
                "total_requests": 10000,
                "total_errors": 50,
                "error_rate": 0.5,
                "endpoints_count": 25,
                "active_alerts": 0,
            },
            "top_endpoints": [],
            "active_alerts": [],
            "insights": [],
            "timestamp": "2026-02-05T12:00:00",
        }
    })


class MetricHistoryRequest(BaseModel):
    """Request for metric history"""
    metric_name: str = Field(..., description="Name of metric to retrieve")
    start_time: Optional[str] = Field(None, description="Start time (ISO format)")
    end_time: Optional[str] = Field(None, description="End time (ISO format)")
    endpoint: Optional[str] = Field(None, description="Filter by endpoint")
    limit: int = Field(100, ge=1, le=1000, description="Limit results")


class MetricHistoryResponse(BaseModel):
    """Response with metric history"""
    metric_name: str
    total_count: int
    values: List[Dict[str, Any]]
    average: float
    min_value: float
    max_value: float
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "metric_name": "endpoint_latency",
            "total_count": 100,
            "values": [
                {
                    "value": 45.3,
                    "timestamp": "2026-02-05T12:00:00",
                    "endpoint": "GET /api/jobs",
                }
            ],
            "average": 50.2,
            "min_value": 10.5,
            "max_value": 250.8,
        }
    })


class AlertResolutionRequest(BaseModel):
    """Request to resolve an alert"""
    alert_id: str = Field(..., description="ID of alert to resolve")
    resolution_notes: Optional[str] = Field(None, description="Notes on resolution")


class CorrelationAnalysis(BaseModel):
    """Correlation between two metrics"""
    metric1: str
    metric2: str
    correlation_coefficient: float = Field(..., ge=-1, le=1)
    interpretation: str = Field(..., description="Interpretation of correlation")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "metric1": "endpoint_latency",
            "metric2": "error_count",
            "correlation_coefficient": 0.78,
            "interpretation": "Strong positive correlation - errors increase with latency",
        }
    })


class AnomalyDetectionResponse(BaseModel):
    """Response from anomaly detection"""
    anomalies_detected: List[AnomalyReport]
    timestamp: str
    analysis_period: str = Field(..., description="Period analyzed")
    recommendations: List[str] = []
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "anomalies_detected": [],
            "timestamp": "2026-02-05T12:00:00",
            "analysis_period": "last_hour",
            "recommendations": ["System performing normally"],
        }
    })

