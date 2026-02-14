"""
Batch Job Management for Multi-Tenant SaaS.

Handles long-running job scheduling, progress tracking,
and result management.

Core Classes:
  - BatchJob: Single job definition
  - JobScheduler: Job queue and execution
  - JobProgressTracker: Real-time job monitoring
  - JobResult: Job completion result
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
from enum import Enum
import threading
import uuid
from collections import deque


class JobStatus(str, Enum):
    """Job execution status."""
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class JobPriority(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BatchJob:
    """Batch job definition."""
    job_id: str
    tenant_id: str
    user_id: str
    job_type: str  # e.g., "video_generation", "model_training"
    
    status: JobStatus = JobStatus.QUEUED
    priority: JobPriority = JobPriority.NORMAL
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    progress_percentage: float = 0.0
    estimated_duration_seconds: Optional[float] = None
    timeout_seconds: float = 3600.0  # 1 hour default
    
    retry_count: int = 0
    max_retries: int = 3
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == JobStatus.RUNNING
    
    def is_completed(self) -> bool:
        """Check if job is completed."""
        return self.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
    
    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.status == JobStatus.COMPLETED
    
    def get_duration_seconds(self) -> Optional[float]:
        """Get job duration in seconds."""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.retry_count < self.max_retries and self.status == JobStatus.FAILED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "job_type": self.job_type,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress_percentage": self.progress_percentage,
            "duration_seconds": self.get_duration_seconds(),
            "error_message": self.error_message,
        }


@dataclass
class JobResult:
    """Job completion result."""
    job_id: str
    status: JobStatus
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    completed_at: datetime = field(default_factory=datetime.utcnow)


class JobProgressTracker:
    """Tracks progress of running jobs."""
    
    def __init__(self):
        """Initialize tracker."""
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._subscribers: Dict[str, List[Callable]] = {}  # job_id -> callbacks
        self._lock = threading.RLock()
    
    def update_progress(
        self,
        job_id: str,
        percentage: float,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update job progress."""
        with self._lock:
            self._progress[job_id] = {
                "percentage": min(100.0, max(0.0, percentage)),
                "message": message,
                "timestamp": datetime.utcnow(),
                "details": details or {},
            }
            
            # Notify subscribers
            if job_id in self._subscribers:
                for callback in self._subscribers[job_id]:
                    try:
                        callback(self._progress[job_id])
                    except Exception:
                        pass
    
    def get_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job progress."""
        with self._lock:
            return self._progress.get(job_id)
    
    def subscribe_to_progress(self, job_id: str, callback: Callable) -> None:
        """Subscribe to progress updates."""
        with self._lock:
            if job_id not in self._subscribers:
                self._subscribers[job_id] = []
            self._subscribers[job_id].append(callback)
    
    def unsubscribe_from_progress(self, job_id: str, callback: Callable) -> None:
        """Unsubscribe from progress updates."""
        with self._lock:
            if job_id in self._subscribers:
                try:
                    self._subscribers[job_id].remove(callback)
                except ValueError:
                    pass
    
    def clear_progress(self, job_id: str) -> None:
        """Clear progress tracking for job."""
        with self._lock:
            self._progress.pop(job_id, None)
            self._subscribers.pop(job_id, None)


class JobScheduler:
    """Schedules and manages job execution."""
    
    def __init__(self, max_concurrent_jobs: int = 10):
        """Initialize scheduler."""
        self.max_concurrent_jobs = max_concurrent_jobs
        self._job_queue: deque = deque()  # Prioritized queue
        self._running_jobs: Dict[str, BatchJob] = {}
        self._completed_jobs: Dict[str, BatchJob] = {}
        self._job_handlers: Dict[str, Callable] = {}  # job_type -> handler
        self._progress_tracker = JobProgressTracker()
        self._tenant_job_limits: Dict[str, int] = {}  # tenant -> max concurrent
        self._lock = threading.RLock()
        self._worker_thread = None
        self._stop_worker = False
    
    def register_job_handler(self, job_type: str, handler: Callable) -> None:
        """Register handler for job type."""
        with self._lock:
            self._job_handlers[job_type] = handler
    
    def set_tenant_job_limit(self, tenant_id: str, max_jobs: int) -> None:
        """Set maximum concurrent jobs for tenant."""
        with self._lock:
            self._tenant_job_limits[tenant_id] = max_jobs
    
    def submit_job(self, job: BatchJob) -> str:
        """Submit job to scheduler."""
        with self._lock:
            job.status = JobStatus.QUEUED
            self._job_queue.append(job)
            self._sort_queue()
        
        return job.job_id
    
    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get current job status."""
        with self._lock:
            if job_id in self._running_jobs:
                return self._running_jobs[job_id].status
            if job_id in self._completed_jobs:
                return self._completed_jobs[job_id].status
        
        return None
    
    def get_job(self, job_id: str) -> Optional[BatchJob]:
        """Get job by ID."""
        with self._lock:
            if job_id in self._running_jobs:
                return self._running_jobs[job_id]
            if job_id in self._completed_jobs:
                return self._completed_jobs[job_id]
        
        return None
    
    def get_tenant_jobs(
        self,
        tenant_id: str,
        status: Optional[JobStatus] = None,
    ) -> List[BatchJob]:
        """Get jobs for tenant."""
        with self._lock:
            jobs = list(self._running_jobs.values()) + list(self._completed_jobs.values())
            jobs = [j for j in jobs if j.tenant_id == tenant_id]
            
            if status:
                jobs = [j for j in jobs if j.status == status]
            
            return jobs
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        with self._lock:
            if job_id in self._running_jobs:
                job = self._running_jobs[job_id]
                job.status = JobStatus.CANCELLED
                return True
            
            # Remove from queue
            self._job_queue = deque(j for j in self._job_queue if j.job_id != job_id)
        
        return False
    
    def get_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job progress."""
        return self._progress_tracker.get_progress(job_id)
    
    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self._lock:
            return {
                "queued_jobs": len(self._job_queue),
                "running_jobs": len(self._running_jobs),
                "completed_jobs": len(self._completed_jobs),
                "max_concurrent": self.max_concurrent_jobs,
            }
    
    def _sort_queue(self) -> None:
        """Sort queue by priority."""
        priority_order = {
            JobPriority.CRITICAL: 0,
            JobPriority.HIGH: 1,
            JobPriority.NORMAL: 2,
            JobPriority.LOW: 3,
        }
        self._job_queue = deque(
            sorted(
                self._job_queue,
                key=lambda j: (priority_order.get(j.priority, 999), j.created_at),
            )
        )
    
    def _get_next_runnable_job(self) -> Optional[BatchJob]:
        """Get next job that can run."""
        with self._lock:
            # Check if we have capacity
            if len(self._running_jobs) >= self.max_concurrent_jobs:
                return None
            
            # Check tenant job limits
            for job in list(self._job_queue):
                tenant_limit = self._tenant_job_limits.get(
                    job.tenant_id,
                    self.max_concurrent_jobs,
                )
                tenant_running = len(
                    [j for j in self._running_jobs.values() if j.tenant_id == job.tenant_id]
                )
                
                if tenant_running < tenant_limit:
                    self._job_queue.remove(job)
                    return job
            
            return None
    
    def _mark_job_complete(self, job: BatchJob, success: bool, result: Dict[str, Any] = None) -> None:
        """Mark job as completed."""
        with self._lock:
            job.status = JobStatus.COMPLETED if success else JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.result = result or {}
            
            self._running_jobs.pop(job.job_id, None)
            self._completed_jobs[job.job_id] = job
            self._progress_tracker.clear_progress(job.job_id)


class JobManagementPortal:
    """Portal for job management and monitoring."""
    
    def __init__(self, scheduler: JobScheduler):
        """Initialize portal."""
        self.scheduler = scheduler
    
    def get_jobs_dashboard(self, tenant_id: str) -> Dict[str, Any]:
        """Get job dashboard for tenant."""
        all_jobs = self.scheduler.get_tenant_jobs(tenant_id)
        
        return {
            "total_jobs": len(all_jobs),
            "running": len([j for j in all_jobs if j.status == JobStatus.RUNNING]),
            "completed": len([j for j in all_jobs if j.status == JobStatus.COMPLETED]),
            "failed": len([j for j in all_jobs if j.status == JobStatus.FAILED]),
            "queued": len([j for j in all_jobs if j.status == JobStatus.QUEUED]),
            "recent_jobs": [j.to_dict() for j in sorted(all_jobs, key=lambda j: j.created_at, reverse=True)[:10]],
        }
    
    def get_job_details(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed job information."""
        job = self.scheduler.get_job(job_id)
        if not job:
            return None
        
        details = job.to_dict()
        progress = self.scheduler.get_progress(job_id)
        if progress:
            details["progress"] = progress
        
        return details
