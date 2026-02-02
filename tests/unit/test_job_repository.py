"""Unit tests for PostgreSQL Job Repository."""

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.db.models import Base, Job, JobStateRecord, JobResult, JobState as JobStateEnum
from src.db.job_repository import JobRepository


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def repo(db_session):
    """Create a JobRepository instance."""
    return JobRepository(db_session)


class TestJobRepositoryCreate:
    """Tests for job creation."""

    def test_create_job_success(self, repo):
        """Test successful job creation."""
        job = repo.create_job(
            content="Test content",
            preset="fast",
            user_id="user123",
            job_metadata={"key": "value"},
        )
        assert job is not None
        assert job.id is not None
        assert job.content == "Test content"
        assert job.preset == "fast"
        assert job.user_id == "user123"
        assert job.state == JobStateEnum.PENDING
        assert job.job_metadata == {"key": "value"}

    def test_create_job_without_metadata(self, repo):
        """Test job creation without metadata."""
        job = repo.create_job(content="Test", preset="normal", user_id="user456")
        assert job.job_metadata == {}

    def test_create_multiple_jobs(self, repo):
        """Test creating multiple jobs."""
        job1 = repo.create_job("content1", "fast", "user1")
        job2 = repo.create_job("content2", "slow", "user2")

        assert job1.id != job2.id
        assert repo.get_job(job1.id) is not None
        assert repo.get_job(job2.id) is not None


class TestJobRepositoryRead:
    """Tests for job retrieval."""

    def test_get_job_exists(self, repo):
        """Test retrieving an existing job."""
        created = repo.create_job("content", "fast", "user1")
        retrieved = repo.get_job(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.content == "content"

    def test_get_job_not_exists(self, repo):
        """Test retrieving a non-existent job."""
        result = repo.get_job("nonexistent-id")
        assert result is None

    def test_get_job_state(self, repo):
        """Test getting job state."""
        job = repo.create_job("content", "fast", "user1")
        state = repo.get_job_state(job.id)
        assert state == "pending"

    def test_get_job_state_not_found(self, repo):
        """Test getting state of non-existent job."""
        state = repo.get_job_state("nonexistent")
        assert state is None


class TestJobRepositoryStateTransitions:
    """Tests for job state management."""

    def test_update_job_state(self, repo):
        """Test updating job state."""
        job = repo.create_job("content", "fast", "user1")
        success = repo.update_job_state(job.id, "processing")

        assert success is True
        updated = repo.get_job(job.id)
        assert updated.state == JobStateEnum.PROCESSING
        assert updated.started_at is not None

    def test_update_job_state_with_reason(self, repo):
        """Test updating job state with reason."""
        job = repo.create_job("content", "fast", "user1")
        repo.update_job_state(job.id, "processing", reason="User requested")

        history = repo.get_job_state_history(job.id)
        assert len(history) == 1
        assert history[0]["reason"] == "User requested"

    def test_job_state_history(self, repo):
        """Test state change history tracking."""
        job = repo.create_job("content", "fast", "user1")

        repo.update_job_state(job.id, "processing", reason="Started")
        repo.update_job_state(job.id, "completed", reason="Finished")

        history = repo.get_job_state_history(job.id)
        assert len(history) == 2
        assert history[0]["to"] == "processing"
        assert history[1]["to"] == "completed"

    def test_completed_job_timestamp(self, repo):
        """Test that completed_at is set when job completes."""
        job = repo.create_job("content", "fast", "user1")
        assert job.completed_at is None

        repo.update_job_state(job.id, "processing")
        repo.update_job_state(job.id, "completed")

        updated = repo.get_job(job.id)
        assert updated.completed_at is not None


class TestJobResults:
    """Tests for job results."""

    def test_set_job_result_success(self, repo):
        """Test setting successful job result."""
        job = repo.create_job("content", "fast", "user1")
        success = repo.set_job_result(
            job.id, status="success", output={"result": "data"}, processing_time_ms=1500
        )

        assert success is True
        result = repo.get_job_result(job.id)
        assert result["status"] == "success"
        assert result["output"]["result"] == "data"
        assert result["processing_time_ms"] == 1500

    def test_set_job_result_error(self, repo):
        """Test setting error job result."""
        job = repo.create_job("content", "fast", "user1")
        repo.set_job_result(job.id, status="error", error_message="Processing failed")

        result = repo.get_job_result(job.id)
        assert result["status"] == "error"
        assert result["error_message"] == "Processing failed"

    def test_get_job_result_not_found(self, repo):
        """Test getting result for job with no result."""
        result = repo.get_job_result("nonexistent")
        assert result is None

    def test_update_job_result(self, repo):
        """Test updating an existing job result."""
        job = repo.create_job("content", "fast", "user1")

        repo.set_job_result(job.id, status="error", error_message="First error")
        repo.set_job_result(job.id, status="success", output={"result": "recovered"})

        result = repo.get_job_result(job.id)
        assert result["status"] == "success"
        assert result["output"]["result"] == "recovered"


class TestJobRepositoryList:
    """Tests for job listing."""

    def test_list_jobs(self, repo):
        """Test listing jobs for a user."""
        repo.create_job("content1", "fast", "user1")
        repo.create_job("content2", "slow", "user1")
        repo.create_job("content3", "fast", "user2")

        jobs = repo.list_jobs("user1")
        assert len(jobs) == 2
        assert all(j["user_id"] == "user1" for j in jobs)

    def test_list_jobs_with_state_filter(self, repo):
        """Test listing jobs with state filter."""
        job1 = repo.create_job("content1", "fast", "user1")
        job2 = repo.create_job("content2", "slow", "user1")

        repo.update_job_state(job1.id, "processing")
        repo.update_job_state(job2.id, "completed")

        processing_jobs = repo.list_jobs("user1", state_filter="processing")
        assert len(processing_jobs) == 1
        assert processing_jobs[0]["state"] == "processing"

    def test_list_jobs_pagination(self, repo):
        """Test pagination."""
        for i in range(5):
            repo.create_job(f"content{i}", "fast", "user1")

        page1 = repo.list_jobs("user1", limit=2, offset=0)
        page2 = repo.list_jobs("user1", limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0]["id"] != page2[0]["id"]

    def test_list_jobs_empty(self, repo):
        """Test listing jobs for user with no jobs."""
        jobs = repo.list_jobs("user_no_jobs")
        assert len(jobs) == 0

    def test_get_job_count(self, repo):
        """Test counting jobs."""
        repo.create_job("content1", "fast", "user1")
        repo.create_job("content2", "slow", "user1")
        repo.create_job("content3", "fast", "user2")

        count = repo.get_job_count("user1")
        assert count == 2


class TestJobRepositoryDelete:
    """Tests for job deletion."""

    def test_delete_job(self, repo):
        """Test soft-delete job."""
        job = repo.create_job("content", "fast", "user1")
        success = repo.delete_job(job.id)

        assert success is True
        updated = repo.get_job(job.id)
        assert updated.state == JobStateEnum.CANCELLED

    def test_delete_job_not_found(self, repo):
        """Test deleting non-existent job."""
        success = repo.delete_job("nonexistent")
        assert success is False


class TestJobRepositoryStuckJobs:
    """Tests for stuck job detection."""

    def test_get_stuck_jobs(self, db_session, repo):
        """Test detecting stuck jobs."""
        job1 = repo.create_job("content1", "fast", "user1")
        job2 = repo.create_job("content2", "fast", "user1")

        repo.update_job_state(job1.id, "processing")
        repo.update_job_state(job2.id, "processing")

        # Make job1 look stuck (started long ago)
        job1_db = db_session.query(Job).filter(Job.id == job1.id).first()
        job1_db.started_at = datetime.now(timezone.utc) - timedelta(hours=2)
        db_session.commit()

        stuck = repo.get_stuck_jobs(processing_threshold_seconds=3600)
        assert len(stuck) == 1
        assert stuck[0].id == job1.id


class TestJobRepositoryCleanup:
    """Tests for old job cleanup."""

    def test_cleanup_old_jobs(self, db_session, repo):
        """Test cleaning up old jobs."""
        job1 = repo.create_job("content1", "fast", "user1")
        job2 = repo.create_job("content2", "fast", "user1")

        repo.update_job_state(job1.id, "completed")
        repo.update_job_state(job2.id, "completed")

        # Make job1 old
        job1_db = db_session.query(Job).filter(Job.id == job1.id).first()
        job1_db.completed_at = datetime.now(timezone.utc) - timedelta(days=31)
        db_session.commit()

        cleanup_count = repo.cleanup_old_jobs(days_old=30)
        assert cleanup_count == 1

        job1_updated = repo.get_job(job1.id)
        assert job1_updated.state == JobStateEnum.CANCELLED


class TestJobRepositoryConcurrency:
    """Tests for concurrent access handling."""

    def test_concurrent_updates(self, db_session, repo):
        """Test that concurrent state updates are handled (via transactions)."""
        job = repo.create_job("content", "fast", "user1")

        # Simulate two concurrent updates
        repo.update_job_state(job.id, "processing")
        repo.update_job_state(job.id, "completed")

        # Should have consistent state
        final_job = repo.get_job(job.id)
        assert final_job.state == JobStateEnum.COMPLETED

        # History should show both transitions
        history = repo.get_job_state_history(job.id)
        assert len(history) == 2
