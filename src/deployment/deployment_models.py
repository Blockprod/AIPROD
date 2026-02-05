"""
Deployment Models - Pydantic models for multi-region endpoints
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class RegionMetricsData(BaseModel):
    """Region metrics response model"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "region_id": "us-east-1",
        "region_name": "US East",
        "tier": "primary",
        "status": "healthy",
        "latency_ms": 25.5,
    }})
    
    region_id: str = Field(..., description="Unique region identifier")
    region_name: str = Field(..., description="Human-readable region name")
    tier: str = Field(..., description="Region tier (primary, secondary, tertiary)")
    status: str = Field(..., description="Region status (healthy, degraded, unhealthy)")
    latency_ms: float = Field(..., ge=0, description="Average latency in milliseconds")
    available_capacity: float = Field(..., ge=0, le=100, description="Available capacity percentage")
    error_rate: float = Field(..., ge=0, le=100, description="Error rate percentage")
    request_count: int = Field(..., ge=0, description="Total requests processed")
    uptime_percentage: float = Field(..., ge=0, le=100, description="Uptime percentage")
    p95_latency: float = Field(..., ge=0, description="95th percentile latency")
    p99_latency: float = Field(..., ge=0, description="99th percentile latency")


class RegionComparisonResponse(BaseModel):
    """Multi-region performance comparison"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "total_regions": 3,
        "healthy_regions": 3,
    }})
    
    total_regions: int = Field(..., ge=0, description="Total registered regions")
    healthy_regions: int = Field(..., ge=0, description="Currently healthy regions")
    primary_regions: int = Field(..., ge=0, description="Primary tier regions")
    regions: List[RegionMetricsData] = Field(..., description="All regions data")
    best_performing: Optional[RegionMetricsData] = Field(None, description="Best performing region")
    worst_performing: Optional[RegionMetricsData] = Field(None, description="Worst performing region")


class CapacityAnalysis(BaseModel):
    """Capacity analysis across regions"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "total_capacity": 10000,
        "used_capacity": 7500,
        "utilization_percentage": 75.0,
    }})
    
    total_capacity: int = Field(..., ge=0, description="Total capacity across all regions")
    used_capacity: float = Field(..., ge=0, description="Currently used capacity")
    available_capacity: float = Field(..., ge=0, description="Available capacity")
    utilization_percentage: float = Field(..., ge=0, le=100, description="Overall utilization percentage")


class MultiRegionOverview(BaseModel):
    """Multi-region deployment overview"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "total_regions": 3,
        "healthy_regions": 3,
        "average_error_rate": 0.5,
    }})
    
    total_regions: int = Field(..., ge=0, description="Total regions")
    healthy_regions: int = Field(..., ge=0, description="Healthy regions")
    degraded_regions: int = Field(..., ge=0, description="Degraded regions")
    unhealthy_regions: int = Field(..., ge=0, description="Unhealthy regions")
    average_error_rate: float = Field(..., ge=0, le=100, description="Average error rate")
    average_latency_ms: float = Field(..., ge=0, description="Average latency")
    health_percentage: float = Field(..., ge=0, le=100, description="Overall health percentage")


class FailoverEvent(BaseModel):
    """Failover event record"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "event_id": "failover_1234567890",
        "timestamp": "2026-02-05T10:30:00",
        "trigger": "high_error_rate",
        "from_region": "us-east-1",
        "to_region": "us-west-2",
        "success": True,
    }})
    
    event_id: str = Field(..., description="Unique failover event ID")
    timestamp: str = Field(..., description="Event timestamp")
    trigger: str = Field(..., description="Failover trigger reason")
    from_region: str = Field(..., description="Source region")
    to_region: str = Field(..., description="Target region")
    success: bool = Field(..., description="Whether failover succeeded")
    traffic_shifted: float = Field(..., ge=0, le=100, description="Traffic shifted percentage")
    reason: Optional[str] = Field(None, description="Detailed reason")


class FailoverStatus(BaseModel):
    """Current failover status"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "active_failovers": [],
        "total_failover_events": 0,
        "successful_failovers": 0,
    }})
    
    active_failovers: List[Dict[str, Any]] = Field(..., description="Currently active failovers")
    total_failover_events: int = Field(..., ge=0, description="Total failover events")
    successful_failovers: int = Field(..., ge=0, description="Successful failovers")
    failed_failovers: int = Field(..., ge=0, description="Failed failovers")


class FailoverAnalytics(BaseModel):
    """Failover analytics and statistics"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "total_events": 5,
        "success_rate": 100.0,
        "most_common_trigger": "high_error_rate",
    }})
    
    total_events: int = Field(..., ge=0, description="Total failover events")
    successful_events: int = Field(..., ge=0, description="Successful failover events")
    success_rate: float = Field(..., ge=0, le=100, description="Success rate percentage")
    average_duration_seconds: float = Field(..., ge=0, description="Average failover duration")
    most_common_trigger: Optional[str] = Field(None, description="Most common trigger type")
    trigger_breakdown: Dict[str, int] = Field(..., description="Breakdown by trigger type")


class TrafficDistribution(BaseModel):
    """Traffic distribution across regions"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "us-east-1": 50.0,
        "us-west-2": 40.0,
        "eu-west-1": 10.0,
    }})
    
    pass  # Dict representation - customized in routes


class MultiRegionDashboard(BaseModel):
    """Complete multi-region dashboard"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "status": "healthy",
        "overview": {},
        "capacity": {},
    }})
    
    status: str = Field(..., description="Overall status (healthy/warning/critical)")
    overview: MultiRegionOverview = Field(..., description="Multi-region overview")
    capacity: CapacityAnalysis = Field(..., description="Capacity analysis")
    failover_status: FailoverStatus = Field(..., description="Failover status")
    top_performing_regions: List[RegionMetricsData] = Field(..., description="Top 3 performing regions")
    recommendations: List[str] = Field(..., description="Optimization recommendations")


class RegisterRegionRequest(BaseModel):
    """Request to register a new region"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "region_name": "US East",
        "endpoint": "https://api-us-east-1.example.com",
        "tier": "primary",
    }})
    
    region_name: str = Field(..., description="Human-readable region name")
    endpoint: str = Field(..., description="Regional API endpoint URL")
    tier: str = Field(..., description="Region tier to assign")
    max_capacity: int = Field(1000, ge=1, description="Maximum capacity for region")


class InitiateFailoverRequest(BaseModel):
    """Request to initiate failover"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "from_region": "us-east-1",
        "to_region": "us-west-2",
        "reason": "Manual failover test",
    }})
    
    from_region: str = Field(..., description="Source region ID")
    to_region: str = Field(..., description="Target region ID")
    reason: Optional[str] = Field(None, description="Failover reason")


class SetTrafficDistributionRequest(BaseModel):
    """Request to set traffic distribution"""
    model_config = ConfigDict(json_schema_extra={"example": {
        "distribution": {
            "us-east-1": 50.0,
            "us-west-2": 50.0,
        }
    }})
    
    distribution: Dict[str, float] = Field(..., description="Traffic distribution by region")
