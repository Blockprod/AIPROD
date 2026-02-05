"""Webhook management for event-driven architecture"""

import logging
import json
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from enum import Enum
import uuid
from dataclasses import dataclass, asdict
import httpx
from asyncio import sleep

logger = logging.getLogger(__name__)

# Webhook configuration
WEBHOOK_TIMEOUT = 30  # seconds
WEBHOOK_MAX_RETRIES = 5
WEBHOOK_RETRY_DELAYS = [1, 2, 5, 10, 30]  # seconds
WEBHOOK_SECRET_HEADER = "X-Webhook-Signature"
WEBHOOK_ID_HEADER = "X-Webhook-ID"
WEBHOOK_TIMESTAMP_HEADER = "X-Webhook-Timestamp"


class WebhookEventType(str, Enum):
    """Supported webhook event types"""
    JOB_CREATED = "job.created"
    JOB_STARTED = "job.started"
    JOB_PROGRESS = "job.progress"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"
    BATCH_CREATED = "batch.created"
    BATCH_COMPLETED = "batch.completed"


class WebhookStatus(str, Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    FAILED_PERMANENTLY = "failed_permanently"


@dataclass
class WebhookEvent:
    """Webhook event payload"""
    event_id: str
    event_type: WebhookEventType
    timestamp: datetime
    data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data
        }


@dataclass
class WebhookDelivery:
    """Webhook delivery attempt record"""
    id: str
    webhook_id: str
    event_id: str
    attempt: int
    status: WebhookStatus
    response_status: Optional[int]
    response_body: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    sent_at: Optional[datetime]
    next_retry: Optional[datetime]


class WebhookManager:
    """Manager for webhook operations"""
    
    def __init__(self, db_client=None, cache_client=None):
        self.db = db_client
        self.cache = cache_client
    
    def generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook payload"""
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_signature(self, payload: str, signature: str, secret: str) -> bool:
        """Verify webhook signature"""
        expected_signature = self.generate_signature(payload, secret)
        return hmac.compare_digest(signature, expected_signature)
    
    async def register_webhook(
        self,
        url: str,
        events: List[WebhookEventType],
        secret: str,
        active: bool = True,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Register new webhook endpoint"""
        
        webhook_id = str(uuid.uuid4())
        
        webhook_data = {
            "id": webhook_id,
            "url": url,
            "events": [e.value for e in events],
            "secret": secret,
            "active": active,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_event_at": None,
            "failed_attempts": 0,
            "success_count": 0
        }
        
        # Store in database
        if self.db:
            await self.db.create_webhook(webhook_data)
        
        # Cache webhook metadata
        if self.cache:
            await self.cache.set(f"webhook:{webhook_id}", json.dumps(webhook_data), ttl=3600)
        
        logger.info(f"Webhook registered: {webhook_id} -> {url}")
        return webhook_data
    
    async def deliver_event(
        self,
        webhook_id: str,
        event: WebhookEvent,
        max_retries: int = WEBHOOK_MAX_RETRIES
    ) -> bool:
        """Deliver webhook event with automatic retries"""
        
        # Get webhook details
        webhook = await self._get_webhook(webhook_id)
        if not webhook or not webhook.get("active"):
            return False
        
        # Check if event type is subscribed
        if event.event_type.value not in webhook.get("events", []):
            return False
        
        payload = json.dumps(event.to_dict())
        signature = self.generate_signature(payload, webhook["secret"])
        
        headers = {
            "Content-Type": "application/json",
            WEBHOOK_SECRET_HEADER: signature,
            WEBHOOK_ID_HEADER: webhook_id,
            WEBHOOK_TIMESTAMP_HEADER: datetime.utcnow().isoformat()
        }
        
        # Attempt delivery with retries
        for attempt in range(1, max_retries + 1):
            try:
                delivery_record = WebhookDelivery(
                    id=str(uuid.uuid4()),
                    webhook_id=webhook_id,
                    event_id=event.event_id,
                    attempt=attempt,
                    status=WebhookStatus.SENT,
                    response_status=None,
                    response_body=None,
                    error_message=None,
                    created_at=datetime.utcnow(),
                    sent_at=None,
                    next_retry=None
                )
                
                async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT) as client:
                    response = await client.post(
                        webhook["url"],
                        content=payload,
                        headers=headers
                    )
                    
                    delivery_record.sent_at = datetime.utcnow()
                    delivery_record.response_status = response.status_code
                    delivery_record.response_body = response.text[:1000]  # Limit size
                    
                    if 200 <= response.status_code < 300:
                        delivery_record.status = WebhookStatus.DELIVERED
                        await self._record_delivery(delivery_record)
                        await self._update_webhook_success(webhook_id)
                        logger.info(f"Webhook delivered: {webhook_id} (attempt {attempt})")
                        return True
                    
                    elif response.status_code >= 500:
                        # Server error - retry
                        delivery_record.status = WebhookStatus.PENDING
                    else:
                        # Client error - don't retry
                        delivery_record.status = WebhookStatus.FAILED
                        await self._update_webhook_failed(webhook_id)
                        return False
                    
                    await self._record_delivery(delivery_record)
                
            except httpx.TimeoutException:
                logger.warning(f"Webhook timeout: {webhook_id} (attempt {attempt})")
                delivery_record.status = WebhookStatus.PENDING
                delivery_record.error_message = "Request timeout"
                await self._record_delivery(delivery_record)
            
            except Exception as e:
                logger.error(f"Webhook delivery error: {webhook_id}: {e}")
                delivery_record.status = WebhookStatus.PENDING
                delivery_record.error_message = str(e)
                await self._record_delivery(delivery_record)
            
            # Schedule retry
            if attempt < max_retries:
                retry_delay = WEBHOOK_RETRY_DELAYS[min(attempt, len(WEBHOOK_RETRY_DELAYS) - 1)]
                next_retry_time = datetime.utcnow() + timedelta(seconds=retry_delay)
                
                if self.db:
                    await self.db.schedule_webhook_retry(webhook_id, event.event_id, next_retry_time)
                
                logger.info(f"Webhook retry scheduled: {webhook_id} in {retry_delay}s")
                await sleep(retry_delay)
            else:
                # Max retries exceeded
                await self._update_webhook_permanently_failed(webhook_id)
                logger.error(f"Webhook failed permanently: {webhook_id}")
                return False
        
        return False
    
    async def get_webhook(self, webhook_id: str) -> Optional[Dict[str, Any]]:
        """Get webhook details"""
        return await self._get_webhook(webhook_id)
    
    async def update_webhook(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[WebhookEventType]] = None,
        active: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """Update webhook configuration"""
        
        webhook = await self._get_webhook(webhook_id)
        if not webhook:
            return None
        
        updates = {}
        if url is not None:
            updates["url"] = url
        if events is not None:
            updates["events"] = [e.value for e in events]
        if active is not None:
            updates["active"] = active
        
        if updates:
            updates["updated_at"] = datetime.utcnow().isoformat()
            
            if self.db:
                await self.db.update_webhook(webhook_id, updates)
            
            webhook.update(updates)
            
            if self.cache:
                await self.cache.set(f"webhook:{webhook_id}", json.dumps(webhook), ttl=3600)
        
        return webhook
    
    async def delete_webhook(self, webhook_id: str) -> bool:
        """Delete webhook"""
        
        if self.db:
            await self.db.delete_webhook(webhook_id)
        
        if self.cache:
            await self.cache.delete(f"webhook:{webhook_id}")
        
        logger.info(f"Webhook deleted: {webhook_id}")
        return True
    
    async def get_webhook_deliveries(
        self,
        webhook_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[WebhookDelivery]:
        """Get delivery history for webhook"""
        
        if not self.db:
            return []
        
        return await self.db.get_webhook_deliveries(webhook_id, limit, offset)
    
    async def replay_event(self, webhook_id: str, event_id: str) -> bool:
        """Replay a failed webhook event"""
        
        if not self.db:
            return False
        
        event_data = await self.db.get_event(event_id)
        if not event_data:
            return False
        
        event = WebhookEvent(
            event_id=event_id,
            event_type=WebhookEventType(event_data["event_type"]),
            timestamp=datetime.fromisoformat(event_data["timestamp"]),
            data=event_data["data"]
        )
        
        return await self.deliver_event(webhook_id, event)
    
    # Private helper methods
    
    async def _get_webhook(self, webhook_id: str) -> Optional[Dict[str, Any]]:
        """Get webhook from cache or database"""
        
        # Try cache first
        if self.cache:
            cached = await self.cache.get(f"webhook:{webhook_id}")
            if cached:
                return json.loads(cached) if isinstance(cached, str) else cached
        
        # Fall back to database
        if self.db:
            webhook = await self.db.get_webhook(webhook_id)
            if webhook and self.cache:
                await self.cache.set(f"webhook:{webhook_id}", json.dumps(webhook), ttl=3600)
            return webhook
        
        return None
    
    async def _record_delivery(self, delivery: WebhookDelivery) -> None:
        """Record webhook delivery attempt"""
        if self.db:
            await self.db.create_webhook_delivery(asdict(delivery))
    
    async def _update_webhook_success(self, webhook_id: str) -> None:
        """Update webhook on successful delivery"""
        if self.db:
            await self.db.update_webhook(webhook_id, {
                "last_event_at": datetime.utcnow().isoformat(),
                "failed_attempts": 0
            })
    
    async def _update_webhook_failed(self, webhook_id: str) -> None:
        """Update webhook on delivery failure"""
        if self.db:
            webhook = await self._get_webhook(webhook_id)
            if webhook is None:
                return
            failed_attempts = (webhook.get("failed_attempts", 0) or 0) + 1
            
            await self.db.update_webhook(webhook_id, {
                "failed_attempts": failed_attempts
            })
            
            # Disable after 10 consecutive failures
            if failed_attempts >= 10:
                await self.update_webhook(webhook_id, active=False)
                logger.warning(f"Webhook disabled due to repeated failures: {webhook_id}")
    
    async def _update_webhook_permanently_failed(self, webhook_id: str) -> None:
        """Update webhook after permanent failure"""
        await self._update_webhook_failed(webhook_id)


# Singleton instance
_webhook_manager_instance = None


def get_webhook_manager(db_client=None, cache_client=None) -> WebhookManager:
    """Get webhook manager instance"""
    global _webhook_manager_instance
    
    if _webhook_manager_instance is None:
        _webhook_manager_instance = WebhookManager(db_client, cache_client)
    
    return _webhook_manager_instance
