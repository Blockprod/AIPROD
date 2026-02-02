"""PostgreSQL-based Job Repository for persistent job storage."""

from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from src.db.models import (
    Job,
    JobState,
    JobStateRecord,
    JobResult,
    JobState as JobStateEnum,
)


class JobRepository:
    """Repository pattern for Job persistence using PostgreSQL."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session

    def create_job(
        self,
        content: str,
        preset: str,
        user_id: str,
        job_metadata: Optional[Dict[str, Any]] = None,
    ) -> Job:
        """Create a new job in the database."""
        job_id = str(uuid4())
        job = Job(
            id=job_id,
            user_id=user_id,
            content=content,
            preset=preset,
            state=JobStateEnum.PENDING,
            job_metadata=job_metadata or {},
        )
        self.session.add(job)
        self.session.commit()
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self.session.query(Job).filter(Job.id == job_id).first()

    def get_job_state(self, job_id: str) -> Optional[str]:
        """Get current state of a job."""
        job = self.get_job(job_id)
        if job:
            state_val = getattr(job, "state", None)
            if state_val is None:
                return None
            return state_val.value if isinstance(state_val, JobStateEnum) else state_val  # type: ignore[return-value]
        return None

    def update_job_state(
        self,
        job_id: str,
        new_state: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update job state and create audit trail."""
        job = self.get_job(job_id)
        if not job:
            return False

        # Convert string to enum if needed
        if isinstance(new_state, str):
            new_state_enum = JobStateEnum[new_state.upper()]
        else:
            new_state_enum = new_state

        previous_state = job.state

        # Update job state
        job.state = new_state_enum  # type: ignore[assignment]
        job.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]

        # Handle state-specific timestamps
        if new_state_enum == JobStateEnum.PROCESSING:
            job.started_at = datetime.now(timezone.utc)  # type: ignore[assignment]
        elif new_state_enum in [JobStateEnum.COMPLETED, JobStateEnum.FAILED]:
            job.completed_at = datetime.now(timezone.utc)  # type: ignore[assignment]

        # Create state history record
        state_record = JobStateRecord(
            job_id=job_id,
            previous_state=previous_state,
            new_state=new_state_enum,
            reason=reason,
            metadata=metadata or {},
        )

        self.session.add(state_record)
        self.session.commit()
        return True

    def set_job_result(
        self,
        job_id: str,
        status: str,  # success, error, timeout
        output: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
    ) -> bool:
        """Store job results."""
        job = self.get_job(job_id)
        if not job:
            return False

        # Check if result already exists (update or insert)
        result = (
            self.session.query(JobResult).filter(JobResult.job_id == job_id).first()
        )
        if result:
            result.status = status  # type: ignore[assignment]
            result.output = output  # type: ignore[assignment]
            result.error_message = error_message  # type: ignore[assignment]
            result.processing_time_ms = processing_time_ms  # type: ignore[assignment]
        else:
            result = JobResult(
                job_id=job_id,
                status=status,
                output=output,
                error_message=error_message,
                processing_time_ms=processing_time_ms,
            )
            self.session.add(result)

        self.session.commit()
        return True

    def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job results."""
        result = (
            self.session.query(JobResult).filter(JobResult.job_id == job_id).first()
        )
        if result:
            return result.to_dict()
        return None

    def get_job_state_history(self, job_id: str) -> List[Dict[str, Any]]:
        """Get state change history for a job."""
        records = (
            self.session.query(JobStateRecord)
            .filter(JobStateRecord.job_id == job_id)
            .order_by(JobStateRecord.created_at)
            .all()
        )

        return [
            {
                "from": getattr(r.previous_state, "value", r.previous_state) if r.previous_state else None,  # type: ignore
                "to": (
                    r.new_state.value
                    if isinstance(r.new_state, JobStateEnum)
                    else r.new_state
                ),
                "reason": r.reason,
                "metadata": r.state_metadata,
                "timestamp": r.created_at.isoformat(),
            }
            for r in records
        ]

    def list_jobs(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        state_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List jobs for a user with optional state filter."""
        query = self.session.query(Job).filter(Job.user_id == user_id)

        if state_filter:
            state_enum = JobStateEnum[state_filter.upper()]
            query = query.filter(Job.state == state_enum)

        jobs = query.order_by(desc(Job.created_at)).limit(limit).offset(offset).all()

        return [job.to_dict() for job in jobs]

    def delete_job(self, job_id: str) -> bool:
        """Soft delete a job (mark as cancelled)."""
        job = self.get_job(job_id)
        if not job:
            return False

        # Soft delete by marking as cancelled
        self.update_job_state(job_id, "CANCELLED", reason="User deletion")
        return True

    def get_job_count(self, user_id: str, state: Optional[str] = None) -> int:
        """Get count of jobs for user."""
        query = self.session.query(Job).filter(Job.user_id == user_id)

        if state:
            state_enum = JobStateEnum[state.upper()]
            query = query.filter(Job.state == state_enum)

        return query.count()

    def get_stuck_jobs(self, processing_threshold_seconds: int = 3600) -> List[Job]:
        """Get jobs stuck in PROCESSING state."""
        threshold = datetime.now(timezone.utc) - timedelta(
            seconds=processing_threshold_seconds
        )
        return (
            self.session.query(Job)
            .filter(
                and_(Job.state == JobStateEnum.PROCESSING, Job.started_at < threshold)
            )
            .all()
        )

    def cleanup_old_jobs(self, days_old: int = 30) -> int:
        """Soft delete old completed jobs (privacy/storage cleanup)."""
        threshold = datetime.now(timezone.utc) - timedelta(days=days_old)
        jobs_to_delete = (
            self.session.query(Job)
            .filter(
                and_(
                    Job.completed_at < threshold,
                    Job.state.in_(
                        [
                            JobStateEnum.COMPLETED,
                            JobStateEnum.FAILED,
                            JobStateEnum.CANCELLED,
                        ]
                    ),
                )
            )
            .all()
        )

        count = 0
        for job in jobs_to_delete:
            self.update_job_state(
                str(job.id), "CANCELLED", reason=f"Cleanup - {days_old} days old"  # type: ignore[arg-type]
            )
            count += 1

        return count

    def close(self):
        """Close the session."""
        self.session.close()
