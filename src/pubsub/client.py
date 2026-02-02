"""Google Cloud Pub/Sub client and utilities for AIPROD_V33."""

import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from google.cloud import pubsub_v1
from google.api_core.exceptions import GoogleAPICallError
import os

logger = logging.getLogger(__name__)


class PubSubClient:
    """Wrapper for Google Cloud Pub/Sub operations."""

    def __init__(self, project_id: Optional[str] = None):
        """Initialize Pub/Sub client."""
        self.project_id = project_id or os.getenv(
            "GOOGLE_CLOUD_PROJECT", "aiprod-484120"
        )
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()

        # Topic paths
        self.jobs_topic_path = self.publisher.topic_path(
            self.project_id, "aiprod-pipeline-jobs"
        )
        self.results_topic_path = self.publisher.topic_path(
            self.project_id, "aiprod-pipeline-results"
        )
        self.dlq_topic_path = self.publisher.topic_path(
            self.project_id, "aiprod-pipeline-dlq"
        )

        # Subscription paths
        self.jobs_subscription_path = self.subscriber.subscription_path(
            self.project_id, "aiprod-pipeline-jobs-sub"
        )
        self.results_subscription_path = self.subscriber.subscription_path(
            self.project_id, "aiprod-pipeline-results-sub"
        )

    def publish_job(
        self,
        job_id: str,
        user_id: str,
        content: str,
        preset: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Publish a job execution request to Pub/Sub."""
        message_data = {
            "job_id": job_id,
            "user_id": user_id,
            "content": content,
            "preset": preset,
            "metadata": metadata or {},
        }

        try:
            message_json = json.dumps(message_data)
            message_bytes = message_json.encode("utf-8")

            # Publish with job_id as ordering key (ensures order per user)
            future = self.publisher.publish(
                self.jobs_topic_path,
                message_bytes,
                ordering_key=user_id,  # Ensures messages from same user are processed in order
            )

            message_id = future.result(timeout=10)
            logger.info(f"Published job {job_id} to Pub/Sub (msg_id={message_id})")
            return message_id

        except GoogleAPICallError as e:
            logger.error(f"Failed to publish job {job_id}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error publishing job {job_id}: {str(e)}")
            raise

    def publish_result(
        self,
        job_id: str,
        status: str,
        output: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
    ) -> str:
        """Publish job result to results topic."""
        message_data = {
            "job_id": job_id,
            "status": status,
            "output": output,
            "error_message": error_message,
            "processing_time_ms": processing_time_ms,
        }

        try:
            message_json = json.dumps(message_data, default=str)
            message_bytes = message_json.encode("utf-8")

            future = self.publisher.publish(
                self.results_topic_path,
                message_bytes,
                ordering_key=job_id,
            )

            message_id = future.result(timeout=10)
            logger.info(f"Published result for job {job_id} (msg_id={message_id})")
            return message_id

        except GoogleAPICallError as e:
            logger.error(f"Failed to publish result for {job_id}: {str(e)}")
            raise

    def publish_dlq_message(
        self,
        job_id: str,
        reason: str,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Publish message to dead-letter queue."""
        message_data = {
            "job_id": job_id,
            "reason": reason,
            "error": error,
            "metadata": metadata or {},
        }

        try:
            message_json = json.dumps(message_data)
            message_bytes = message_json.encode("utf-8")

            future = self.publisher.publish(
                self.dlq_topic_path,
                message_bytes,
                ordering_key=job_id,
            )

            message_id = future.result(timeout=10)
            logger.warning(f"Published DLQ message for job {job_id}: {reason}")
            return message_id

        except GoogleAPICallError as e:
            logger.error(f"Failed to publish DLQ message for {job_id}: {str(e)}")
            raise

    def pull_messages(self, subscription_path: str, max_messages: int = 1) -> List[Any]:
        """Pull messages from subscription."""
        try:
            response = self.subscriber.pull(
                request={
                    "subscription": subscription_path,
                    "max_messages": max_messages,
                },
                timeout=10,
            )
            return list(response.received_messages)
        except GoogleAPICallError as e:
            logger.error(f"Failed to pull messages: {str(e)}")
            raise

    def acknowledge_message(self, subscription_path: str, ack_ids: List[str]):
        """Acknowledge received messages."""
        try:
            self.subscriber.acknowledge(
                request={"subscription": subscription_path, "ack_ids": ack_ids}
            )
        except GoogleAPICallError as e:
            logger.error(f"Failed to acknowledge messages: {str(e)}")
            raise


@dataclass
class JobMessage:
    """Schema for job execution messages."""

    job_id: Optional[str] = None
    user_id: Optional[str] = None
    content: Optional[str] = None
    preset: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "JobMessage":
        """Create JobMessage from dictionary."""
        return JobMessage(
            job_id=data.get("job_id"),
            user_id=data.get("user_id"),
            content=data.get("content"),
            preset=data.get("preset"),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "content": self.content,
            "preset": self.preset,
            "metadata": self.metadata,
        }


@dataclass
class ResultMessage:
    """Schema for result messages."""

    job_id: Optional[str] = None
    status: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[int] = None

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ResultMessage":
        """Create ResultMessage from dictionary."""
        return ResultMessage(
            job_id=data.get("job_id"),
            status=data.get("status"),  # success, error, timeout
            output=data.get("output"),
            error_message=data.get("error_message"),
            processing_time_ms=data.get("processing_time_ms"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "output": self.output,
            "error_message": self.error_message,
            "processing_time_ms": self.processing_time_ms,
        }


# Singleton instance
_pubsub_client: Optional[PubSubClient] = None


def get_pubsub_client() -> PubSubClient:
    """Get or create Pub/Sub client singleton."""
    global _pubsub_client
    if _pubsub_client is None:
        _pubsub_client = PubSubClient()
    return _pubsub_client
