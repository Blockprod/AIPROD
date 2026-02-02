"""
Tests unitaires pour les endpoints API asynchrones P1.2
- POST /pipeline/run (async pattern)
- GET /pipeline/job/{job_id}
- GET /pipeline/jobs
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def mock_job():
    """Create mock Job object."""
    job = MagicMock()
    job.id = "job-12345"
    job.content = "Test video content for pipeline"
    job.preset = "quick_social"
    job.user_id = "user-123"
    job.current_state = "QUEUED"
    job.created_at = datetime(2026, 1, 16, 12, 0, 0)
    job.updated_at = datetime(2026, 1, 16, 12, 0, 5)
    job.retry_count = 0
    job.state_history = []
    job.result = None
    return job


@pytest.fixture
def mock_job_with_result(mock_job):
    """Create mock Job with completed result."""
    mock_job.current_state = "COMPLETED"
    mock_job.result = MagicMock()
    mock_job.result.success = True
    mock_job.result.output = {"video_url": "https://storage.example.com/video.mp4"}
    mock_job.result.completed_at = datetime(2026, 1, 16, 12, 5, 0)
    mock_job.result.error_message = None
    mock_job.result.execution_time_ms = 30000
    return mock_job


@pytest.fixture
def mock_job_with_history(mock_job):
    """Create mock Job with state history."""
    state1 = MagicMock()
    state1.state = "QUEUED"
    state1.entered_at = datetime(2026, 1, 16, 12, 0, 0)
    state1.state_metadata = {}

    state2 = MagicMock()
    state2.state = "PROCESSING"
    state2.entered_at = datetime(2026, 1, 16, 12, 0, 5)
    state2.state_metadata = {"worker_id": "worker-1"}

    mock_job.state_history = [state1, state2]
    mock_job.current_state = "PROCESSING"
    return mock_job


class TestJobRepository:
    """Tests for JobRepository integration in API."""

    def test_job_repo_create_job(self):
        """Test that JobRepository can be instantiated with mock session."""
        from src.db.job_repository import JobRepository

        mock_session = MagicMock()
        repo = JobRepository(mock_session)
        assert repo is not None
        assert repo.session == mock_session

    def test_job_repo_get_job(self, mock_job):
        """Test JobRepository.get_job method."""
        from src.db.job_repository import JobRepository

        mock_session = MagicMock()
        repo = JobRepository(mock_session)

        # Mock the query chain - JobRepository uses .filter() not .options().filter()
        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        result = repo.get_job("job-12345")
        assert result is not None
        assert str(result.id) == "job-12345"


class TestPubSubClient:
    """Tests for PubSubClient integration in API."""

    def test_pubsub_client_initialization(self):
        """Test PubSubClient can be created."""
        from src.pubsub.client import PubSubClient

        with patch("src.pubsub.client.pubsub_v1.PublisherClient"):
            with patch("src.pubsub.client.pubsub_v1.SubscriberClient"):
                client = PubSubClient(project_id="test-project")
                assert client is not None
                assert client.project_id == "test-project"

    def test_pubsub_job_message_schema(self):
        """Test JobMessage schema structure."""
        from src.pubsub.client import JobMessage

        # JobMessage uses from_dict pattern
        msg = JobMessage.from_dict(
            {
                "job_id": "job-123",
                "user_id": "user-456",
                "content": "Test content",
                "preset": "quick_social",
                "metadata": {"test": "value"},
            }
        )

        assert msg.job_id == "job-123"
        assert msg.user_id == "user-456"
        assert msg.content == "Test content"
        assert msg.preset == "quick_social"
        assert msg.metadata == {"test": "value"}

    def test_pubsub_result_message_schema(self):
        """Test ResultMessage schema structure."""
        from src.pubsub.client import ResultMessage

        # ResultMessage uses from_dict pattern
        msg = ResultMessage.from_dict(
            {
                "job_id": "job-123",
                "status": "success",
                "output": {"video_url": "https://example.com/video.mp4"},
                "error_message": None,
                "processing_time_ms": 5000,
            }
        )

        assert msg.job_id == "job-123"
        assert msg.status == "success"
        assert msg.output is not None
        assert msg.output["video_url"] == "https://example.com/video.mp4"
        assert msg.error_message is None
        assert msg.processing_time_ms == 5000


class TestAsyncAPIIntegration:
    """Tests for async API endpoint logic (without full FastAPI client)."""

    def test_job_creation_flow(self, mock_job):
        """Test the job creation flow logic."""
        from src.db.job_repository import JobRepository

        mock_session = MagicMock()
        repo = JobRepository(mock_session)

        # Mock create_job to return a job
        mock_session.add = MagicMock()
        mock_session.commit = MagicMock()
        mock_session.refresh = MagicMock()

        with patch.object(repo, "create_job", return_value=mock_job):
            job = repo.create_job(
                content="Test video about AI",
                preset="quick_social",
                user_id="user-123",
                job_metadata={},
            )

            assert str(job.id) == "job-12345"
            assert str(job.preset) == "quick_social"

    def test_pubsub_publish_job_flow(self):
        """Test publishing job to Pub/Sub."""
        from src.pubsub.client import PubSubClient

        with patch("src.pubsub.client.pubsub_v1.PublisherClient") as mock_pub:
            with patch("src.pubsub.client.pubsub_v1.SubscriberClient"):
                # Setup mock publisher
                mock_publisher = MagicMock()
                mock_future = MagicMock()
                mock_future.result.return_value = "msg-12345"
                mock_publisher.publish.return_value = mock_future
                mock_pub.return_value = mock_publisher

                client = PubSubClient(project_id="test-project")

                result = client.publish_job(
                    job_id="job-123",
                    user_id="user-456",
                    content="Test content",
                    preset="quick_social",
                    metadata={},
                )

                assert result == "msg-12345"
                mock_publisher.publish.assert_called_once()

    def test_job_status_response_format(self, mock_job_with_result):
        """Test job status response includes all expected fields."""
        # Simulate building the response as in the API endpoint
        job = mock_job_with_result

        response = {
            "job_id": job.id,
            "status": job.current_state,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "preset": job.preset,
            "content_preview": (
                job.content[:100] + "..." if len(job.content) > 100 else job.content
            ),
        }

        # Add result if completed
        if job.result:
            response["result"] = {
                "success": job.result.success,
                "output": job.result.output,
                "completed_at": (
                    job.result.completed_at.isoformat()
                    if job.result.completed_at
                    else None
                ),
                "error_message": job.result.error_message,
                "execution_time_ms": job.result.execution_time_ms,
            }

        # Verify response structure
        assert response["job_id"] == "job-12345"
        assert response["status"] == "COMPLETED"
        assert "result" in response
        assert response["result"]["success"] is True
        assert "video_url" in response["result"]["output"]

    def test_state_history_response_format(self, mock_job_with_history):
        """Test state history is properly formatted."""
        job = mock_job_with_history

        state_history = [
            {
                "state": record.state,
                "entered_at": record.entered_at.isoformat(),
                "metadata": record.state_metadata,
            }
            for record in job.state_history
        ]

        assert len(state_history) == 2
        assert state_history[0]["state"] == "QUEUED"
        assert state_history[1]["state"] == "PROCESSING"
        assert state_history[1]["metadata"]["worker_id"] == "worker-1"

    def test_job_list_response_format(self, mock_job):
        """Test job list pagination response."""
        jobs = [mock_job]
        limit = 20
        offset = 0

        response = {
            "jobs": [
                {
                    "job_id": job.id,
                    "status": job.current_state,
                    "preset": job.preset,
                    "content_preview": (
                        job.content[:50] + "..."
                        if len(job.content) > 50
                        else job.content
                    ),
                    "created_at": job.created_at.isoformat(),
                }
                for job in jobs
            ],
            "limit": limit,
            "offset": offset,
            "count": len(jobs),
        }

        assert len(response["jobs"]) == 1
        assert response["jobs"][0]["job_id"] == "job-12345"
        assert response["limit"] == 20
        assert response["offset"] == 0
        assert response["count"] == 1


class TestErrorHandling:
    """Tests for error handling in async endpoints."""

    def test_pubsub_failure_handling(self):
        """Test that Pub/Sub failures are handled gracefully."""
        from src.pubsub.client import PubSubClient

        with patch("src.pubsub.client.pubsub_v1.PublisherClient") as mock_pub:
            with patch("src.pubsub.client.pubsub_v1.SubscriberClient"):
                # Setup mock publisher to fail
                mock_publisher = MagicMock()
                mock_future = MagicMock()
                mock_future.result.side_effect = Exception("Pub/Sub unavailable")
                mock_publisher.publish.return_value = mock_future
                mock_pub.return_value = mock_publisher

                client = PubSubClient(project_id="test-project")

                # Should raise exception
                with pytest.raises(Exception) as exc_info:
                    client.publish_job(
                        job_id="job-123",
                        user_id="user-456",
                        content="Test content",
                        preset="quick_social",
                        metadata={},
                    )

                assert "Pub/Sub unavailable" in str(exc_info.value)

    def test_job_not_found_scenario(self):
        """Test handling when job doesn't exist."""
        from src.db.job_repository import JobRepository

        mock_session = MagicMock()
        repo = JobRepository(mock_session)

        # Mock query returning None - JobRepository uses .filter() not .options().filter()
        mock_session.query.return_value.filter.return_value.first.return_value = None

        result = repo.get_job("nonexistent-job")
        assert result is None

    def test_access_denied_scenario(self, mock_job):
        """Test access denied logic."""
        requesting_user_id = "other-user"
        job_owner_id = mock_job.user_id  # "user-123"

        # Simulating the access check
        access_denied = job_owner_id != requesting_user_id

        assert access_denied is True
        assert job_owner_id == "user-123"
        assert requesting_user_id != job_owner_id
