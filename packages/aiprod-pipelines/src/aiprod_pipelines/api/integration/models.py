"""
Pydantic Models for API Requests and Responses
==============================================

Defines data models for HTTP API interface.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineRequestModel(BaseModel):
    """
    Pipeline execution request.
    
    Example:
        {
            "request_id": "job-001",
            "prompt": "A cat playing in a sunny garden",
            "duration_sec": 60,
            "budget": 2.0,
            "complexity": 0.5,
            "preferences": {
                "style": "cinematic",
                "mood": "cheerful"
            },
            "fallback_enabled": true
        }
    """
    request_id: str = Field(..., description="Unique job identifier")
    prompt: str = Field(..., description="Video generation prompt")
    duration_sec: int = Field(default=60, ge=10, le=300, description="Video duration in seconds")
    budget: float = Field(default=1.0, ge=0.1, le=10.0, description="Budget in USD")
    complexity: float = Field(default=0.5, ge=0.0, le=1.0, description="Complexity score")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    fallback_enabled: bool = Field(default=True, description="Enable backend fallback")
    
    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "job-001",
                "prompt": "A cat playing in a sunny garden",
                "duration_sec": 60,
                "budget": 2.0,
                "complexity": 0.5,
                "preferences": {"style": "cinematic"},
                "fallback_enabled": True
            }
        }


class VideoAssetModel(BaseModel):
    """Generated video asset."""
    id: str
    url: str
    duration_sec: float
    resolution: str
    codec: str
    bitrate: int
    file_size_bytes: int
    thumbnail_url: str


class CostBreakdownModel(BaseModel):
    """Cost estimation breakdown."""
    base_cost: float
    quantization_factor: float
    gpu_cost_factor: float
    batch_efficiency: float
    orchestration_overhead: float
    total_estimated: float
    cost_per_minute: float
    selected_backend: str
    confidence: float


class QAReportModel(BaseModel):
    """Quality assurance report."""
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: List[Dict[str, Any]]
    videos_analyzed: int


class PipelineResponseModel(BaseModel):
    """
    Pipeline execution response.
    
    Contains execution results, generated assets, and metadata.
    """
    job_id: str = Field(..., description="Job identifier")
    status: JobStatus = Field(..., description="Execution status")
    delivery_manifest: Dict[str, Any] = Field(..., description="Complete delivery manifest")
    cost: float = Field(..., description="Total cost in USD")
    quality_score: float = Field(..., description="Quality score (0-1)")
    execution_time_sec: float = Field(..., description="Execution time in seconds")
    checkpoints_created: int = Field(..., description="Number of checkpoints created")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-001",
                "status": "completed",
                "delivery_manifest": {},
                "cost": 1.5,
                "quality_score": 0.85,
                "execution_time_sec": 120.5,
                "checkpoints_created": 7,
                "errors": []
            }
        }


class CheckpointInfo(BaseModel):
    """Checkpoint metadata."""
    checkpoint_id: str
    job_id: str
    state: str
    timestamp: float
    used_successfully: bool = False


class ResumeJobRequest(BaseModel):
    """Request to resume a job from checkpoint."""
    job_id: str = Field(..., description="Job identifier to resume")
    checkpoint_id: Optional[str] = Field(None, description="Specific checkpoint (or latest if None)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job-001",
                "checkpoint_id": "ckpt-12345"
            }
        }


class JobStatusResponse(BaseModel):
    """Job status query response."""
    job_id: str
    status: JobStatus
    current_state: str
    progress_percent: float
    started_at: float
    completed_at: Optional[float] = None
    checkpoints: List[CheckpointInfo]
