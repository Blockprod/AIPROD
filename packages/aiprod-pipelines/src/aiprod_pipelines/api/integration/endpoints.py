"""
FastAPI HTTP Endpoints for Pipeline Orchestration
=================================================

Provides REST API interface to execute and manage production pipelines.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any
import asyncio

from .models import (
    PipelineRequestModel,
    PipelineResponseModel,
    ResumeJobRequest,
    JobStatusResponse,
    JobStatus
)
from ..orchestrator import Orchestrator
from ..checkpoint.manager import CheckpointManager
from ..schema.schemas import PipelineRequest


# Initialize FastAPI app
app = FastAPI(
    title="AIPROD Production Pipeline API",
    description="Production pipeline orchestration with checkpoint/resume capabilities",
    version="2.0.0"
)

# Global state (in production, would use proper dependency injection)
orchestrator: Orchestrator = None
checkpoint_manager: CheckpointManager = None
active_jobs: Dict[str, Dict[str, Any]] = {}


def initialize_orchestrator(adapters: Dict[str, Any]):
    """
    Initialize the orchestrator with adapters.
    
    Should be called during application startup.
    """
    global orchestrator, checkpoint_manager
    
    checkpoint_manager = CheckpointManager()
    orchestrator = Orchestrator(
        adapters=adapters,
        checkpoint_manager=checkpoint_manager,
        max_retries=3
    )


@app.post("/api/pipeline/execute", response_model=PipelineResponseModel)
async def execute_pipeline(request: PipelineRequestModel):
    """
    Execute production pipeline.
    
    Processes a video generation request through the full production pipeline
    with automatic checkpoint creation and error recovery.
    
    Args:
        request: Pipeline execution request
        
    Returns:
        Pipeline execution response with results
        
    Raises:
        503: Service not initialized
        500: Execution failed
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        # Convert Pydantic model to dict
        pipeline_request: PipelineRequest = {
            "request_id": request.request_id,
            "prompt": request.prompt,
            "duration_sec": request.duration_sec,
            "budget": request.budget,
            "complexity": request.complexity,
            "preferences": request.preferences,
            "fallback_enabled": request.fallback_enabled
        }
        
        # Track job
        active_jobs[request.request_id] = {
            "status": JobStatus.RUNNING,
            "started_at": asyncio.get_event_loop().time(),
            "request": pipeline_request
        }
        
        # Execute pipeline
        response = await orchestrator.execute(pipeline_request)
        
        # Update job status
        active_jobs[request.request_id]["status"] = (
            JobStatus.COMPLETED if response["status"] == "completed" else JobStatus.FAILED
        )
        active_jobs[request.request_id]["completed_at"] = asyncio.get_event_loop().time()
        
        return PipelineResponseModel(**response)
    
    except Exception as e:
        # Update job status
        if request.request_id in active_jobs:
            active_jobs[request.request_id]["status"] = JobStatus.FAILED
        
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@app.post("/api/pipeline/resume", response_model=PipelineResponseModel)
async def resume_pipeline(request: ResumeJobRequest):
    """
    Resume a failed pipeline job from checkpoint.
    
    Args:
        request: Resume request with job_id and optional checkpoint_id
        
    Returns:
        Pipeline execution response
        
    Raises:
        404: Job or checkpoint not found
        503: Service not initialized
        500: Resume failed
    """
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        # Resume job
        response = await orchestrator.resume_job(
            job_id=request.job_id,
            checkpoint_id=request.checkpoint_id
        )
        
        # Update job status
        if request.job_id in active_jobs:
            active_jobs[request.job_id]["status"] = (
                JobStatus.COMPLETED if response["status"] == "completed" else JobStatus.FAILED
            )
            active_jobs[request.job_id]["completed_at"] = asyncio.get_event_loop().time()
        
        return PipelineResponseModel(**response)
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume failed: {str(e)}")


@app.get("/api/pipeline/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Query job execution status.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Job status information
        
    Raises:
        404: Job not found
    """
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    job = active_jobs[job_id]
    
    # Get checkpoints
    checkpoints = []
    if checkpoint_manager and job_id in checkpoint_manager.checkpoints_index:
        for ckpt in checkpoint_manager.checkpoints_index[job_id]:
            checkpoints.append({
                "checkpoint_id": ckpt["checkpoint_id"],
                "job_id": job_id,
                "state": ckpt["state"],
                "timestamp": ckpt["timestamp"],
                "used_successfully": False  # Would check actual file
            })
    
    response = JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        current_state=job.get("current_state", "unknown"),
        progress_percent=_calculate_progress(job),
        started_at=job["started_at"],
        completed_at=job.get("completed_at"),
        checkpoints=checkpoints
    )
    
    return response


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "orchestrator_initialized": orchestrator is not None,
        "active_jobs": len(active_jobs)
    }


@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics."""
    return {
        "total_jobs": len(active_jobs),
        "running_jobs": sum(1 for j in active_jobs.values() if j["status"] == JobStatus.RUNNING),
        "completed_jobs": sum(1 for j in active_jobs.values() if j["status"] == JobStatus.COMPLETED),
        "failed_jobs": sum(1 for j in active_jobs.values() if j["status"] == JobStatus.FAILED),
        "total_checkpoints": sum(
            len(checkpoint_manager.checkpoints_index.get(job_id, []))
            for job_id in active_jobs.keys()
        ) if checkpoint_manager else 0
    }


def _calculate_progress(job: Dict[str, Any]) -> float:
    """
    Calculate job progress percentage.
    
    Based on current state in the 11-state pipeline.
    """
    state_progress = {
        "INIT": 0.0,
        "ANALYSIS": 0.1,
        "CREATIVE_DIRECTION": 0.2,
        "VISUAL_TRANSLATION": 0.3,
        "FAST_TRACK": 0.3,
        "FINANCIAL_OPTIMIZATION": 0.4,
        "RENDER_EXECUTION": 0.6,
        "QA_TECHNICAL": 0.8,
        "QA_SEMANTIC": 0.9,
        "FINALIZE": 1.0,
        "ERROR": 0.0
    }
    
    current_state = job.get("current_state", "INIT")
    return state_progress.get(current_state, 0.0)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # In production, would initialize adapters here
    # initialize_orchestrator(adapters={})
    pass


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    # Save any pending checkpoints, cleanup resources
    pass
