"""
Core Schema Definitions - AIPROD Internal Format
================================================

TypedDict definitions for internal AIPROD execution context and data structures.
"""

from typing import TypedDict, Dict, Any, List, Optional


class PipelineRequest(TypedDict, total=False):
    """Input request to pipeline orchestrator."""
    request_id: str
    prompt: str
    duration_sec: int
    budget: float
    complexity: float
    preferences: Dict[str, Any]
    fallback_enabled: bool


class Context(TypedDict, total=False):
    """Execution context passed through state machine."""
    request_id: str
    state: str
    memory: Dict[str, Any]
    config: Dict[str, Any]
    error: Optional[str]


class PipelineResponse(TypedDict, total=False):
    """Output response from pipeline execution."""
    job_id: str
    status: str
    delivery_manifest: Dict[str, Any]
    cost: float
    quality_score: float
    execution_time_sec: float
    checkpoints_created: int
    errors: List[str]


class VideoAsset(TypedDict, total=False):
    """Generated video asset metadata."""
    id: str
    url: str
    duration_sec: float
    resolution: str
    codec: str
    bitrate: int
    file_size_bytes: int
    thumbnail_url: str


class QAReport(TypedDict, total=False):
    """Quality assurance validation report."""
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: List[Dict[str, Any]]
    videos_analyzed: int
    technical_score: float
    semantic_score: float
