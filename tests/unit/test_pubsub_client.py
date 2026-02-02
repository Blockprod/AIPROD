"""Tests for Pub/Sub client and messaging."""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from src.pubsub.client import PubSubClient, JobMessage, ResultMessage, get_pubsub_client


class TestPubSubClientInit:
    """Tests for PubSubClient initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default project ID."""
        with patch("src.pubsub.client.pubsub_v1.PublisherClient"):
            with patch("src.pubsub.client.pubsub_v1.SubscriberClient"):
                client = PubSubClient()
                assert client.project_id == "aiprod-484120"

    def test_init_with_custom_project(self):
        """Test initialization with custom project ID."""
        with patch("src.pubsub.client.pubsub_v1.PublisherClient"):
            with patch("src.pubsub.client.pubsub_v1.SubscriberClient"):
                client = PubSubClient("custom-project")
                assert client.project_id == "custom-project"

    def test_topic_paths_created(self):
        """Test that topic paths are created correctly."""
        with patch("src.pubsub.client.pubsub_v1.PublisherClient") as mock_publisher:
            with patch("src.pubsub.client.pubsub_v1.SubscriberClient"):
                mock_pub_instance = MagicMock()
                mock_publisher.return_value = mock_pub_instance
                mock_pub_instance.topic_path = Mock(
                    side_effect=lambda proj, topic: f"projects/{proj}/topics/{topic}"
                )

                client = PubSubClient()

                # Verify topic paths were created
                assert "aiprod-pipeline-jobs" in client.jobs_topic_path
                assert "aiprod-pipeline-results" in client.results_topic_path
                assert "aiprod-pipeline-dlq" in client.dlq_topic_path


class TestPublishJob:
    """Tests for publishing job messages."""

    @patch("src.pubsub.client.pubsub_v1.PublisherClient")
    @patch("src.pubsub.client.pubsub_v1.SubscriberClient")
    def test_publish_job_success(self, mock_sub, mock_pub):
        """Test successful job publication."""
        mock_pub_instance = MagicMock()
        mock_pub.return_value = mock_pub_instance
        mock_pub_instance.topic_path = Mock(
            return_value="projects/aiprod-484120/topics/aiprod-pipeline-jobs"
        )

        mock_future = MagicMock()
        mock_future.result.return_value = "message-id-123"
        mock_pub_instance.publish.return_value = mock_future

        client = PubSubClient()
        msg_id = client.publish_job(
            job_id="job-123",
            user_id="user-1",
            content="test content",
            preset="fast",
            metadata={"key": "value"},
        )

        assert msg_id == "message-id-123"
        mock_pub_instance.publish.assert_called_once()

    @patch("src.pubsub.client.pubsub_v1.PublisherClient")
    @patch("src.pubsub.client.pubsub_v1.SubscriberClient")
    def test_publish_job_without_metadata(self, mock_sub, mock_pub):
        """Test publishing job without metadata."""
        mock_pub_instance = MagicMock()
        mock_pub.return_value = mock_pub_instance
        mock_pub_instance.topic_path = Mock(
            return_value="projects/aiprod-484120/topics/aiprod-pipeline-jobs"
        )

        mock_future = MagicMock()
        mock_future.result.return_value = "message-id-456"
        mock_pub_instance.publish.return_value = mock_future

        client = PubSubClient()
        msg_id = client.publish_job(
            job_id="job-456", user_id="user-2", content="content", preset="slow"
        )

        assert msg_id == "message-id-456"

    @patch("src.pubsub.client.pubsub_v1.PublisherClient")
    @patch("src.pubsub.client.pubsub_v1.SubscriberClient")
    def test_publish_job_ordering_key(self, mock_sub, mock_pub):
        """Test that user_id is used as ordering key."""
        mock_pub_instance = MagicMock()
        mock_pub.return_value = mock_pub_instance
        mock_pub_instance.topic_path = Mock(
            return_value="projects/aiprod-484120/topics/aiprod-pipeline-jobs"
        )

        mock_future = MagicMock()
        mock_future.result.return_value = "msg-id"
        mock_pub_instance.publish.return_value = mock_future

        client = PubSubClient()
        client.publish_job(job_id="job-1", user_id="user-xyz", content="c", preset="f")

        # Check that publish was called with ordering_key=user_id
        call_args = mock_pub_instance.publish.call_args
        assert call_args[1]["ordering_key"] == "user-xyz"


class TestPublishResult:
    """Tests for publishing result messages."""

    @patch("src.pubsub.client.pubsub_v1.PublisherClient")
    @patch("src.pubsub.client.pubsub_v1.SubscriberClient")
    def test_publish_result_success(self, mock_sub, mock_pub):
        """Test publishing successful result."""
        mock_pub_instance = MagicMock()
        mock_pub.return_value = mock_pub_instance
        mock_pub_instance.topic_path = Mock(
            return_value="projects/aiprod-484120/topics/aiprod-pipeline-results"
        )

        mock_future = MagicMock()
        mock_future.result.return_value = "result-msg-123"
        mock_pub_instance.publish.return_value = mock_future

        client = PubSubClient()
        msg_id = client.publish_result(
            job_id="job-123",
            status="success",
            output={"video": "url"},
            processing_time_ms=5000,
        )

        assert msg_id == "result-msg-123"

    @patch("src.pubsub.client.pubsub_v1.PublisherClient")
    @patch("src.pubsub.client.pubsub_v1.SubscriberClient")
    def test_publish_result_error(self, mock_sub, mock_pub):
        """Test publishing error result."""
        mock_pub_instance = MagicMock()
        mock_pub.return_value = mock_pub_instance
        mock_pub_instance.topic_path = Mock(
            return_value="projects/aiprod-484120/topics/aiprod-pipeline-results"
        )

        mock_future = MagicMock()
        mock_future.result.return_value = "error-msg-456"
        mock_pub_instance.publish.return_value = mock_future

        client = PubSubClient()
        msg_id = client.publish_result(
            job_id="job-456", status="error", error_message="Processing failed"
        )

        assert msg_id == "error-msg-456"


class TestPublishDLQ:
    """Tests for publishing to dead-letter queue."""

    @patch("src.pubsub.client.pubsub_v1.PublisherClient")
    @patch("src.pubsub.client.pubsub_v1.SubscriberClient")
    def test_publish_dlq_message(self, mock_sub, mock_pub):
        """Test publishing to DLQ."""
        mock_pub_instance = MagicMock()
        mock_pub.return_value = mock_pub_instance
        mock_pub_instance.topic_path = Mock(
            return_value="projects/aiprod-484120/topics/aiprod-pipeline-dlq"
        )

        mock_future = MagicMock()
        mock_future.result.return_value = "dlq-msg-789"
        mock_pub_instance.publish.return_value = mock_future

        client = PubSubClient()
        msg_id = client.publish_dlq_message(
            job_id="job-789", reason="max_retries_exceeded", error="Last error: timeout"
        )

        assert msg_id == "dlq-msg-789"


class TestMessageSchemas:
    """Tests for message schema classes."""

    def test_job_message_from_dict(self):
        """Test JobMessage creation from dictionary."""
        data = {
            "job_id": "j1",
            "user_id": "u1",
            "content": "test",
            "preset": "fast",
            "metadata": {"key": "val"},
        }
        msg = JobMessage.from_dict(data)

        assert msg.job_id == "j1"
        assert msg.user_id == "u1"
        assert msg.content == "test"
        assert msg.preset == "fast"
        assert msg.metadata == {"key": "val"}

    def test_job_message_to_dict(self):
        """Test JobMessage serialization."""
        msg = JobMessage()
        msg.job_id = "j2"
        msg.user_id = "u2"
        msg.content = "c"
        msg.preset = "slow"
        msg.metadata = {}

        data = msg.to_dict()
        assert data["job_id"] == "j2"
        assert data["preset"] == "slow"

    def test_result_message_from_dict(self):
        """Test ResultMessage creation."""
        data = {
            "job_id": "j3",
            "status": "success",
            "output": {"result": "data"},
            "processing_time_ms": 1500,
        }
        msg = ResultMessage.from_dict(data)

        assert msg.job_id == "j3"
        assert msg.status == "success"
        assert msg.processing_time_ms == 1500

    def test_result_message_to_dict(self):
        """Test ResultMessage serialization."""
        msg = ResultMessage()
        msg.job_id = "j4"
        msg.status = "error"
        msg.output = None
        msg.error_message = "Failed"
        msg.processing_time_ms = None

        data = msg.to_dict()
        assert data["status"] == "error"
        assert data["error_message"] == "Failed"


class TestSingleton:
    """Tests for singleton pattern."""

    @patch("src.pubsub.client.pubsub_v1.PublisherClient")
    @patch("src.pubsub.client.pubsub_v1.SubscriberClient")
    def test_get_pubsub_client_singleton(self, mock_sub, mock_pub):
        """Test that get_pubsub_client returns same instance."""
        # Reset global
        import src.pubsub.client as pubsub_module

        pubsub_module._pubsub_client = None

        client1 = get_pubsub_client()
        client2 = get_pubsub_client()

        assert client1 is client2
