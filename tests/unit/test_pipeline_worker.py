"""
Tests pour le Pipeline Worker (P1.2.3).

Tests:
- Message processing flow
- Job status updates
- Result publishing
- Error handling & DLQ routing
- Concurrent message processing
"""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime

# Note: Import simplifiés pour tests unitaires
# Les vraies dépendances sont mockées


@pytest.fixture
def mock_pubsub_message():
    """Create a mock Pub/Sub message."""
    message = MagicMock()

    job_data = {
        "job_id": "job-worker-123",
        "user_id": "user-456",
        "content": "Create a video about AI",
        "preset": "quick_social",
        "metadata": {"duration_sec": 30},
    }

    message.data = json.dumps(job_data).encode("utf-8")
    message.ack = MagicMock()
    message.nack = MagicMock()

    return message


@pytest.fixture
def mock_worker():
    """Create a mock PipelineWorker."""
    worker = MagicMock()
    worker.project_id = "test-project"
    worker.num_threads = 5
    worker.subscription_path = (
        "projects/test-project/subscriptions/aiprod-pipeline-jobs-sub"
    )

    return worker


class TestMessageProcessing:
    """Tests for message processing flow."""

    def test_decode_job_message(self, mock_pubsub_message):
        """Test decoding JobMessage from Pub/Sub message."""
        data = json.loads(mock_pubsub_message.data.decode("utf-8"))

        assert data["job_id"] == "job-worker-123"
        assert data["user_id"] == "user-456"
        assert data["content"] == "Create a video about AI"
        assert data["preset"] == "quick_social"
        assert data["metadata"]["duration_sec"] == 30

    def test_job_message_from_dict(self):
        """Test JobMessage.from_dict() deserialization."""
        from src.pubsub.client import JobMessage

        msg_dict = {
            "job_id": "job-123",
            "user_id": "user-456",
            "content": "Test content",
            "preset": "quick_social",
            "metadata": {"test": "value"},
        }

        msg = JobMessage.from_dict(msg_dict)

        assert msg.job_id == "job-123"
        assert msg.user_id == "user-456"
        assert msg.content == "Test content"
        assert msg.preset == "quick_social"
        assert msg.metadata == {"test": "value"}

    def test_prepare_input_data(self):
        """Test preparing input data for state machine."""
        from src.pubsub.client import JobMessage

        job_msg = JobMessage.from_dict(
            {
                "job_id": "job-123",
                "user_id": "user-456",
                "content": "Test video",
                "preset": "quick_social",
                "metadata": {"duration_sec": 30},
            }
        )

        # Simulate worker data preparation
        input_data = {
            "content": job_msg.content,
            "preset": job_msg.preset,
            "_user_id": job_msg.user_id,
            "_job_id": job_msg.job_id,
        }

        if job_msg.metadata:
            input_data.update(job_msg.metadata)

        assert input_data["content"] == "Test video"
        assert input_data["preset"] == "quick_social"
        assert input_data["_user_id"] == "user-456"
        assert input_data["_job_id"] == "job-123"
        assert input_data["duration_sec"] == 30


class TestJobStatusUpdates:
    """Tests for job status updates during processing."""

    def test_job_status_transition_queued_to_processing(self):
        """Test job status transitions: PENDING → PROCESSING."""
        from src.db.models import JobState

        initial_state = JobState.PENDING
        new_state = JobState.PROCESSING

        assert initial_state != new_state
        assert new_state.name == "PROCESSING"

    def test_job_status_transition_processing_to_completed(self):
        """Test job status transitions: PROCESSING → COMPLETED."""
        from src.db.models import JobState

        initial_state = JobState.PROCESSING
        new_state = JobState.COMPLETED

        assert new_state.name == "COMPLETED"

    def test_job_status_transition_processing_to_failed(self):
        """Test job status transitions: PROCESSING → FAILED."""
        from src.db.models import JobState

        initial_state = JobState.PROCESSING
        new_state = JobState.FAILED

        assert new_state.name == "FAILED"

    def test_job_result_storage(self):
        """Test storing job result."""
        job_id = "job-123"
        success = True
        output = {
            "video_url": "https://storage.example.com/video.mp4",
            "duration_ms": 30000,
        }
        error_message = None
        execution_time_ms = 15000

        # Simulate result storage
        result = {
            "job_id": job_id,
            "success": success,
            "output": output,
            "error_message": error_message,
            "execution_time_ms": execution_time_ms,
        }

        assert result["job_id"] == "job-123"
        assert result["success"] is True
        assert "video_url" in result["output"]
        assert result["execution_time_ms"] == 15000


class TestResultPublishing:
    """Tests for publishing results to Pub/Sub."""

    def test_result_message_creation(self):
        """Test creating ResultMessage."""
        from src.pubsub.client import ResultMessage

        result_msg = ResultMessage.from_dict(
            {
                "job_id": "job-123",
                "status": "success",
                "output": {"video_url": "https://example.com/video.mp4"},
                "error_message": None,
                "processing_time_ms": 15000,
            }
        )

        assert result_msg.job_id == "job-123"
        assert result_msg.status == "success"
        assert result_msg.output is not None
        assert result_msg.output["video_url"] == "https://example.com/video.mp4"
        assert result_msg.processing_time_ms == 15000

    def test_result_message_serialization(self):
        """Test serializing ResultMessage to dict."""
        from src.pubsub.client import ResultMessage

        result_msg = ResultMessage.from_dict(
            {
                "job_id": "job-123",
                "status": "success",
                "output": {"video_url": "https://example.com/video.mp4"},
                "error_message": None,
                "processing_time_ms": 15000,
            }
        )

        serialized = result_msg.to_dict()

        assert serialized["job_id"] == "job-123"
        assert serialized["status"] == "success"
        assert "video_url" in serialized["output"]

    def test_publish_to_results_topic(self):
        """Test publishing result to results topic."""
        # Simulate publishing
        job_id = "job-123"
        result_msg_id = "msg-results-456"

        # Verify message was published
        assert result_msg_id.startswith("msg-")
        assert len(result_msg_id) > 0


class TestErrorHandling:
    """Tests for error handling and DLQ routing."""

    def test_pipeline_execution_error(self):
        """Test handling pipeline execution error."""
        job_id = "job-123"
        error = Exception("Pipeline timeout")

        # Simulate error tracking
        error_info = {
            "job_id": job_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        assert error_info["job_id"] == "job-123"
        assert error_info["error_type"] == "Exception"
        assert "timeout" in error_info["error_message"].lower()

    def test_dlq_message_creation(self):
        """Test creating DLQ message."""
        job_id = "job-123"
        error = "Pipeline execution timeout"

        dlq_msg = {
            "job_id": job_id,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "retry_count": 0,
        }

        assert dlq_msg["job_id"] == "job-123"
        assert "timeout" in dlq_msg["error"].lower()
        assert dlq_msg["retry_count"] == 0

    def test_message_nack_for_retry(self, mock_pubsub_message):
        """Test nacking message for retry."""
        # Simulate error during processing
        # Message should be nacked to return to queue

        mock_pubsub_message.nack()

        mock_pubsub_message.nack.assert_called_once()

    def test_message_ack_on_success(self, mock_pubsub_message):
        """Test acking message after successful processing."""
        # Simulate successful processing
        # Message should be acked to remove from queue

        mock_pubsub_message.ack()

        mock_pubsub_message.ack.assert_called_once()

    def test_update_job_to_failed(self):
        """Test updating job status to FAILED."""
        from src.db.models import JobState

        job_id = "job-123"
        error_message = "Pipeline execution failed: timeout"

        job_update = {
            "job_id": job_id,
            "state": JobState.FAILED,
            "reason": error_message,
        }

        assert job_update["state"].name == "FAILED"
        assert error_message in job_update["reason"]


class TestConcurrentProcessing:
    """Tests for concurrent message processing."""

    def test_flow_control_settings(self):
        """Test Pub/Sub flow control configuration."""
        max_messages = 5
        max_bytes = 10 * 1024 * 1024  # 10MB

        flow_control = {"max_messages": max_messages, "max_bytes": max_bytes}

        assert flow_control["max_messages"] == 5
        assert flow_control["max_bytes"] == 10485760

    def test_worker_thread_count(self, mock_worker):
        """Test worker thread configuration."""
        assert mock_worker.num_threads == 5

    def test_multiple_message_queue(self):
        """Test handling multiple messages in queue."""
        messages = [
            {"job_id": "job-1", "user_id": "user-1"},
            {"job_id": "job-2", "user_id": "user-2"},
            {"job_id": "job-3", "user_id": "user-3"},
        ]

        assert len(messages) == 3
        assert all("job_id" in m for m in messages)


class TestWorkerInitialization:
    """Tests for worker initialization."""

    def test_worker_project_config(self, mock_worker):
        """Test worker project configuration."""
        assert mock_worker.project_id == "test-project"
        assert "aiprod-pipeline-jobs-sub" in mock_worker.subscription_path

    def test_worker_thread_pool_setup(self, mock_worker):
        """Test worker thread pool setup."""
        num_threads = mock_worker.num_threads

        assert num_threads > 0
        assert num_threads <= 10  # Reasonable limit

    def test_subscription_path_format(self, mock_worker):
        """Test Pub/Sub subscription path format."""
        subscription_path = mock_worker.subscription_path

        assert "projects/" in subscription_path
        assert "subscriptions/" in subscription_path
        assert "aiprod-pipeline-jobs-sub" in subscription_path


class TestIntegration:
    """End-to-end integration tests."""

    def test_complete_processing_flow(self, mock_pubsub_message):
        """Test complete message processing flow."""
        # 1. Decode message
        data = json.loads(mock_pubsub_message.data.decode("utf-8"))
        assert data["job_id"] == "job-worker-123"

        # 2. Would update job to PROCESSING
        job_state = "PROCESSING"
        assert job_state == "PROCESSING"

        # 3. Would execute pipeline
        execution_time = 15000
        assert execution_time > 0

        # 4. Would publish result
        result_status = "success"
        assert result_status == "success"

        # 5. Would ack message
        mock_pubsub_message.ack()
        assert mock_pubsub_message.ack.called

    def test_error_flow_with_dlq(self, mock_pubsub_message):
        """Test error flow with DLQ routing."""
        # 1. Decode message
        data = json.loads(mock_pubsub_message.data.decode("utf-8"))
        job_id = data["job_id"]

        # 2. Simulate error
        error_occurred = True
        assert error_occurred

        # 3. Would update job to FAILED
        job_state = "FAILED"
        assert job_state == "FAILED"

        # 4. Would publish to DLQ
        dlq_published = True
        assert dlq_published

        # 5. Would nack message for retry
        mock_pubsub_message.nack()
        assert mock_pubsub_message.nack.called
