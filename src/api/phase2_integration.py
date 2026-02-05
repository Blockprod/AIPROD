"""API Integration for PHASE 2 Features: Webhooks, Caching, and Advanced Documentation

This module provides integration points for PHASE 2 features without modifying main.py.
To enable these features, import and apply them in main.py startup hooks.
"""

from typing import Optional, Dict, Any, List
from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
from src.cache import RedisCache, cache_get, cache_set, cache_health
from src.webhooks import WebhookManager, WebhookEventType, WebhookEvent
from src.api.openapi_docs import get_endpoint_documentation, TAGS_METADATA, API_DOCUMENTATION
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Create router for PHASE 2 features
phase2_router = APIRouter(prefix="/api/v1", tags=["Phase 2 Features"])

# Initialize managers
webhook_manager: Optional[WebhookManager] = None
redis_cache: Optional[RedisCache] = None


def init_phase2_features(app: FastAPI):
    """Initialize PHASE 2 features in FastAPI app
    
    Call this in your app startup hook:
    
    @app.on_event("startup")
    async def startup():
        init_phase2_features(app)
    """
    
    global webhook_manager, redis_cache
    
    # Initialize Redis cache
    redis_cache = RedisCache()
    if redis_cache.is_available():
        logger.info("✅ Redis cache initialized")
    else:
        logger.warning("⚠️ Redis cache unavailable - using fallback")
    
    # Initialize webhook manager
    webhook_manager = WebhookManager(db_client=None, cache_client=redis_cache)
    logger.info("✅ Webhook manager initialized")
    
    # Update OpenAPI schema with PHASE 2 tags
    if app.openapi_schema is None:
        app.openapi_schema = app.openapi()
    
    app.openapi_schema["tags"] = TAGS_METADATA
    
    logger.info("✅ PHASE 2 features initialized")


# ==================== WEBHOOK ENDPOINTS ====================

@phase2_router.post("/webhooks", tags=["Webhooks"])
async def register_webhook(
    url: str,
    events: List[str],
    secret: str,
    active: bool = True
) -> Dict[str, Any]:
    """Register a webhook endpoint for event notifications
    
    **Webhook Events:**
    - job.created: New job submitted
    - job.started: Job processing started
    - job.progress: Job progress update
    - job.completed: Job completed successfully
    - job.failed: Job failed with error
    - batch.created: Batch job created
    - batch.completed: Batch job completed
    
    **Security:**
    All webhooks are signed with HMAC-SHA256. Verify signature:
    ```
    signature = HMAC-SHA256(payload, secret)
    verify: signature == request.headers['X-Webhook-Signature']
    ```
    """
    
    if not webhook_manager:
        raise HTTPException(status_code=503, detail="Webhook service not initialized")
    
    try:
        event_types = [WebhookEventType(e) for e in events]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid event type: {e}")
    
    webhook_data = await webhook_manager.register_webhook(
        url=url,
        events=event_types,
        secret=secret,
        active=active
    )
    
    return {
        "webhook_id": webhook_data["id"],
        "url": webhook_data["url"],
        "events": webhook_data["events"],
        "active": webhook_data["active"],
        "created_at": webhook_data["created_at"]
    }


@phase2_router.get("/webhooks/{webhook_id}", tags=["Webhooks"])
async def get_webhook(webhook_id: str) -> Dict[str, Any]:
    """Get webhook details and configuration"""
    
    if not webhook_manager:
        raise HTTPException(status_code=503, detail="Webhook service not initialized")
    
    webhook = await webhook_manager.get_webhook(webhook_id)
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    return webhook


@phase2_router.put("/webhooks/{webhook_id}", tags=["Webhooks"])
async def update_webhook(
    webhook_id: str,
    url: Optional[str] = None,
    events: Optional[List[str]] = None,
    active: Optional[bool] = None
) -> Dict[str, Any]:
    """Update webhook configuration"""
    
    if not webhook_manager:
        raise HTTPException(status_code=503, detail="Webhook service not initialized")
    
    event_types = None
    if events:
        try:
            event_types = [WebhookEventType(e) for e in events]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid event type: {e}")
    
    webhook = await webhook_manager.update_webhook(
        webhook_id=webhook_id,
        url=url,
        events=event_types,
        active=active
    )
    
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    return webhook


@phase2_router.delete("/webhooks/{webhook_id}", tags=["Webhooks"])
async def delete_webhook(webhook_id: str) -> Dict[str, str]:
    """Delete webhook endpoint"""
    
    if not webhook_manager:
        raise HTTPException(status_code=503, detail="Webhook service not initialized")
    
    success = await webhook_manager.delete_webhook(webhook_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    return {"status": "deleted", "webhook_id": webhook_id}


@phase2_router.get("/webhooks/{webhook_id}/deliveries", tags=["Webhooks"])
async def get_webhook_deliveries(
    webhook_id: str,
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """Get webhook delivery history"""
    
    if not webhook_manager:
        raise HTTPException(status_code=503, detail="Webhook service not initialized")
    
    deliveries = await webhook_manager.get_webhook_deliveries(webhook_id, limit, offset)
    
    return {
        "webhook_id": webhook_id,
        "deliveries": [
            {
                "id": d.id,
                "event_id": d.event_id,
                "attempt": d.attempt,
                "status": d.status.value,
                "response_status": d.response_status,
                "created_at": d.created_at.isoformat(),
                "sent_at": d.sent_at.isoformat() if d.sent_at else None,
                "error": d.error_message
            }
            for d in deliveries
        ],
        "count": len(deliveries)
    }


@phase2_router.post("/webhooks/{webhook_id}/replay/{event_id}", tags=["Webhooks"])
async def replay_webhook_event(webhook_id: str, event_id: str) -> Dict[str, str]:
    """Replay a failed webhook event"""
    
    if not webhook_manager:
        raise HTTPException(status_code=503, detail="Webhook service not initialized")
    
    success = await webhook_manager.replay_event(webhook_id, event_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Webhook or event not found")
    
    return {
        "status": "replaying",
        "webhook_id": webhook_id,
        "event_id": event_id
    }


# ==================== CACHE ENDPOINTS ====================

@phase2_router.get("/cache/health", tags=["Cache"])
async def check_cache_health() -> Dict[str, Any]:
    """Get Redis cache health status"""
    
    return await cache_health()


@phase2_router.get("/cache/{key}", tags=["Cache"])
async def get_cached_value(key: str) -> Dict[str, Any]:
    """Retrieve value from cache"""
    
    value = await cache_get(key)
    
    if value is None:
        raise HTTPException(status_code=404, detail="Cache key not found")
    
    return {"key": key, "value": value}


@phase2_router.post("/cache/{key}", tags=["Cache"])
async def set_cached_value(
    key: str,
    value: Any,
    ttl: int = 300
) -> Dict[str, Any]:
    """Set value in cache with TTL"""
    
    success = await cache_set(key, value, ttl)
    
    if not success:
        raise HTTPException(status_code=503, detail="Cache service unavailable")
    
    return {
        "status": "set",
        "key": key,
        "ttl": ttl
    }


# ==================== API DOCUMENTATION ENDPOINTS ====================

@phase2_router.get("/docs/api/overview", tags=["Documentation"])
async def get_api_overview() -> Dict[str, Any]:
    """Get comprehensive API overview and features"""
    return {"overview": API_DOCUMENTATION["overview"]}


@phase2_router.get("/docs/api/pipeline", tags=["Documentation"])
async def get_pipeline_guide() -> Dict[str, str]:
    """Get audio generation pipeline guide"""
    return {"guide": API_DOCUMENTATION["pipeline_guide"]}


@phase2_router.get("/docs/api/batch", tags=["Documentation"])
async def get_batch_operations_guide() -> Dict[str, str]:
    """Get batch processing operations guide"""
    return {"guide": API_DOCUMENTATION["batch_operations"]}


@phase2_router.get("/docs/api/errors", tags=["Documentation"])
async def get_error_codes_reference() -> Dict[str, str]:
    """Get error codes and troubleshooting guide"""
    return {"errors": API_DOCUMENTATION["error_codes"]}


@phase2_router.get("/docs/endpoints/{endpoint_name}", tags=["Documentation"])
async def get_endpoint_docs(endpoint_name: str) -> Dict[str, Any]:
    """Get detailed documentation for specific endpoint"""
    
    docs = get_endpoint_documentation(endpoint_name)
    
    if not docs:
        raise HTTPException(status_code=404, detail="Endpoint documentation not found")
    
    return docs


# ==================== SYSTEM HEALTH ====================

@phase2_router.get("/health/detailed", tags=["Health"])
async def detailed_health_check() -> Dict[str, Any]:
    """Get detailed system health including cache and webhooks"""
    
    cache_status = await cache_health()
    webhook_status = {
        "status": "healthy" if webhook_manager else "unavailable",
        "webhooks_registered": 0  # Would query database
    }
    
    return {
        "api": "healthy",
        "cache": cache_status,
        "webhooks": webhook_status,
        "timestamp": datetime.utcnow().isoformat()
    }


# ==================== HELPER FUNCTION ====================

async def emit_webhook_event(
    event_type: WebhookEventType,
    data: Dict[str, Any],
    webhook_ids: Optional[List[str]] = None
) -> int:
    """Emit webhook event to registered endpoints
    
    Usage in main.py after job completion:
    ```
    await emit_webhook_event(
        WebhookEventType.JOB_COMPLETED,
        {"job_id": job.id, "result": job.result},
        webhook_ids=job.subscribed_webhooks
    )
    ```
    """
    
    if not webhook_manager:
        logger.warning("Webhook manager not initialized")
        return 0
    
    event = WebhookEvent(
        event_id=f"evt_{datetime.utcnow().timestamp()}",
        event_type=event_type,
        timestamp=datetime.utcnow(),
        data=data
    )
    
    delivered_count = 0
    
    if webhook_ids:
        for webhook_id in webhook_ids:
            try:
                success = await webhook_manager.deliver_event(webhook_id, event)
                if success:
                    delivered_count += 1
            except Exception as e:
                logger.error(f"Failed to deliver webhook {webhook_id}: {e}")
    
    return delivered_count


def register_phase2_router(app: FastAPI):
    """Register PHASE 2 router with FastAPI app"""
    app.include_router(phase2_router)
    logger.info("PHASE 2 router registered")


__all__ = [
    "init_phase2_features",
    "register_phase2_router",
    "emit_webhook_event",
    "phase2_router"
]
