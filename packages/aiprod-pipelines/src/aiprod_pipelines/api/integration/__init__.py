"""
Integration Layer - FastAPI Models and Endpoints
================================================

Provides HTTP API interface to the production pipeline orchestrator.
"""

from .models import (
    PipelineRequestModel,
    PipelineResponseModel,
    CheckpointInfo,
    JobStatus
)

__all__ = [
    "PipelineRequestModel",
    "PipelineResponseModel",
    "CheckpointInfo",
    "JobStatus",
]
