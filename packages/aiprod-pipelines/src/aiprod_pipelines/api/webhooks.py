"""
AIPROD Webhook Notification System
====================================

Sends async HTTP callbacks to client-configured endpoints when job events
occur (queued, processing, completed, failed).

Features:
- Per-tenant webhook URL registration
- HMAC-SHA256 payload signing for verification
- Exponential-backoff retry (3 attempts)
- Event filtering (subscribe to specific events)
- Delivery audit trail
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class WebhookEvent(str, Enum):
    JOB_QUEUED = "job.queued"
    JOB_PROCESSING = "job.processing"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    QUOTA_WARNING = "quota.warning"
    QUOTA_EXCEEDED = "quota.exceeded"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class WebhookEndpoint:
    """A registered webhook endpoint for a tenant."""

    endpoint_id: str = ""
    tenant_id: str = ""
    url: str = ""
    secret: str = ""  # shared secret for signature verification
    events: List[WebhookEvent] = field(default_factory=lambda: list(WebhookEvent))
    active: bool = True
    created_at: float = 0.0

    def __post_init__(self):
        if not self.endpoint_id:
            self.endpoint_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()
        if not self.secret:
            self.secret = uuid.uuid4().hex


@dataclass
class WebhookPayload:
    """Payload sent to webhook endpoints."""

    event: WebhookEvent
    job_id: str
    tenant_id: str
    timestamp: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_json(self) -> str:
        return json.dumps({
            "event": self.event.value,
            "job_id": self.job_id,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }, default=str)


@dataclass
class DeliveryRecord:
    """Audit record of a webhook delivery attempt."""

    delivery_id: str
    endpoint_id: str
    event: WebhookEvent
    job_id: str
    status_code: int = 0
    success: bool = False
    attempt: int = 1
    timestamp: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


# ---------------------------------------------------------------------------
# Signature computation
# ---------------------------------------------------------------------------

def compute_signature(payload_json: str, secret: str) -> str:
    """Compute HMAC-SHA256 signature for webhook payload verification."""
    return hmac.new(
        secret.encode(), payload_json.encode(), hashlib.sha256
    ).hexdigest()


def verify_signature(payload_json: str, secret: str, signature: str) -> bool:
    """Verify a webhook signature."""
    expected = compute_signature(payload_json, secret)
    return hmac.compare_digest(expected, signature)


# ---------------------------------------------------------------------------
# Webhook manager
# ---------------------------------------------------------------------------

class WebhookManager:
    """
    Manages webhook endpoint registration and event delivery.

    In production, delivery would be async (Celery task / Cloud Tasks).
    Here we provide synchronous delivery with retry for structural
    correctness.
    """

    MAX_RETRIES = 3
    RETRY_DELAYS = [1.0, 5.0, 30.0]  # exponential backoff seconds

    def __init__(self):
        self._endpoints: Dict[str, WebhookEndpoint] = {}  # endpoint_id → endpoint
        self._tenant_endpoints: Dict[str, List[str]] = {}  # tenant_id → [endpoint_ids]
        self._delivery_log: List[DeliveryRecord] = []

    # ---- Registration -------------------------------------------------------

    def register(
        self,
        tenant_id: str,
        url: str,
        events: Optional[List[WebhookEvent]] = None,
        secret: Optional[str] = None,
    ) -> WebhookEndpoint:
        """Register a webhook endpoint for a tenant."""
        ep = WebhookEndpoint(
            tenant_id=tenant_id,
            url=url,
            events=events or list(WebhookEvent),
            secret=secret or uuid.uuid4().hex,
        )
        self._endpoints[ep.endpoint_id] = ep
        self._tenant_endpoints.setdefault(tenant_id, []).append(ep.endpoint_id)
        return ep

    def unregister(self, endpoint_id: str) -> bool:
        """Remove a webhook endpoint."""
        ep = self._endpoints.pop(endpoint_id, None)
        if ep is None:
            return False
        tenant_eps = self._tenant_endpoints.get(ep.tenant_id, [])
        if endpoint_id in tenant_eps:
            tenant_eps.remove(endpoint_id)
        return True

    def list_endpoints(self, tenant_id: str) -> List[WebhookEndpoint]:
        """List all endpoints for a tenant."""
        ids = self._tenant_endpoints.get(tenant_id, [])
        return [self._endpoints[eid] for eid in ids if eid in self._endpoints]

    # ---- Delivery -----------------------------------------------------------

    def dispatch(self, payload: WebhookPayload) -> List[DeliveryRecord]:
        """
        Dispatch a webhook event to all matching endpoints for the tenant.

        Returns list of delivery records.
        """
        records: List[DeliveryRecord] = []
        endpoint_ids = self._tenant_endpoints.get(payload.tenant_id, [])

        for eid in endpoint_ids:
            ep = self._endpoints.get(eid)
            if ep is None or not ep.active:
                continue
            if payload.event not in ep.events:
                continue

            record = self._deliver(ep, payload)
            records.append(record)
            self._delivery_log.append(record)

        return records

    def _deliver(self, endpoint: WebhookEndpoint, payload: WebhookPayload) -> DeliveryRecord:
        """Attempt delivery with retries."""
        payload_json = payload.to_json()
        signature = compute_signature(payload_json, endpoint.secret)

        for attempt in range(1, self.MAX_RETRIES + 1):
            record = DeliveryRecord(
                delivery_id=str(uuid.uuid4()),
                endpoint_id=endpoint.endpoint_id,
                event=payload.event,
                job_id=payload.job_id,
                attempt=attempt,
            )
            try:
                req = Request(
                    endpoint.url,
                    data=payload_json.encode(),
                    headers={
                        "Content-Type": "application/json",
                        "X-AIPROD-Signature": signature,
                        "X-AIPROD-Event": payload.event.value,
                        "X-AIPROD-Delivery": record.delivery_id,
                    },
                    method="POST",
                )
                resp = urlopen(req, timeout=10)
                record.status_code = resp.status
                record.success = 200 <= resp.status < 300
                if record.success:
                    return record
            except (URLError, OSError) as e:
                record.error = str(e)
                record.success = False

            # Retry with backoff (skip sleep on last attempt)
            if attempt < self.MAX_RETRIES:
                delay = self.RETRY_DELAYS[min(attempt - 1, len(self.RETRY_DELAYS) - 1)]
                # In production: schedule retry via task queue instead of sleep
                # time.sleep(delay)  # disabled for sync operation

        return record

    # ---- Audit trail --------------------------------------------------------

    @property
    def delivery_log(self) -> List[DeliveryRecord]:
        return list(self._delivery_log)

    def delivery_stats(self, tenant_id: Optional[str] = None) -> Dict[str, int]:
        """Get delivery statistics."""
        records = self._delivery_log
        if tenant_id:
            tenant_eps = set(self._tenant_endpoints.get(tenant_id, []))
            records = [r for r in records if r.endpoint_id in tenant_eps]
        return {
            "total": len(records),
            "success": sum(1 for r in records if r.success),
            "failed": sum(1 for r in records if not r.success),
        }
