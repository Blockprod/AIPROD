"""
Système d'audit logging pour les événements de sécurité.
Envoie les logs vers stdout (pour Cloud Logging) et optionnellement vers Datadog.
"""

import os
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types d'événements auditables."""

    API_CALL = "API_CALL"
    AUTH_SUCCESS = "AUTH_SUCCESS"
    AUTH_FAILURE = "AUTH_FAILURE"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    DATA_ACCESS = "DATA_ACCESS"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    SECRET_ACCESS = "SECRET_ACCESS"
    ADMIN_ACTION = "ADMIN_ACTION"
    ERROR = "ERROR"
    SECURITY_ALERT = "SECURITY_ALERT"
    # Phase 1.2 - Export events
    EXPORT = "EXPORT"
    # Phase 1.3 - API Key events
    API_KEY_CREATED = "API_KEY_CREATED"
    API_KEY_ROTATED = "API_KEY_ROTATED"
    API_KEY_REVOKED = "API_KEY_REVOKED"
    API_KEY_MASS_REVOKED = "API_KEY_MASS_REVOKED"
    API_KEY_LISTED = "API_KEY_LISTED"


class AuditLogger:
    """Logger centralisé pour les événements de sécurité."""

    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.service_name = os.getenv("SERVICE_NAME", "aiprod-v33")
        self.dd_enabled = os.getenv("DD_API_KEY") is not None

        if self.dd_enabled:
            try:
                from datadog import statsd  # type: ignore[attr-defined]
                from datadog.api import api_client

                self.dd_client = api_client
            except ImportError:
                self.dd_enabled = False
                logger.warning("Datadog SDK not installed, audit logging disabled")

    def log_event(
        self,
        event_type: AuditEventType,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Enregistre un événement d'audit.

        Args:
            event_type: Type d'événement (AuditEventType)
            user_id: ID de l'utilisateur (email ou UUID)
            action: Action effectuée (ex: "create_job", "access_metrics")
            resource: Ressource affectée (ex: "/pipeline/run", "job_123")
            status: Statut (success, failure, denied)
            details: Détails supplémentaires
            tags: Tags pour Datadog
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        audit_event = {
            "timestamp": timestamp,
            "event_type": event_type.value,
            "environment": self.environment,
            "service": self.service_name,
            "user_id": user_id or "anonymous",
            "action": action or "unknown",
            "resource": resource or "unknown",
            "status": status,
            "details": details or {},
        }

        # Log vers stdout (pour Cloud Logging)
        logger.info(f"AUDIT: {json.dumps(audit_event)}")

        # Log vers Datadog si configuré
        if self.dd_enabled:
            self._send_to_datadog(audit_event, tags)

    def _send_to_datadog(
        self, event: Dict[str, Any], tags: Optional[Dict[str, str]]
    ) -> None:
        """Envoie l'événement vers Datadog."""
        try:
            from datadog import initialize, api

            options = {
                "api_key": os.getenv("DD_API_KEY"),
                "app_key": os.getenv("DD_APP_KEY"),
            }
            initialize(**options)  # type: ignore[arg-type]

            dd_tags = [f"{k}:{v}" for k, v in (tags or {}).items()]
            dd_tags.extend(
                [
                    f"service:{self.service_name}",
                    f"environment:{self.environment}",
                ]
            )

            api.Event.create(  # type: ignore[attr-defined]
                title=f"{event['event_type']}: {event['action']}",
                text=json.dumps(event),
                tags=dd_tags,
                alert_type="info" if event["status"] == "success" else "error",
            )
        except Exception as e:
            logger.error(f"Failed to send audit event to Datadog: {e}")

    def log_api_call(
        self,
        endpoint: str,
        method: str,
        user_id: Optional[str] = None,
        status_code: int = 200,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Enregistre un appel API."""
        self.log_event(
            event_type=AuditEventType.API_CALL,
            user_id=user_id,
            action=f"{method} {endpoint}",
            resource=endpoint,
            status="success" if 200 <= status_code < 300 else "error",
            details={
                "method": method,
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )

    def log_auth_success(self, user_id: str, method: str = "firebase") -> None:
        """Enregistre une authentification réussie."""
        self.log_event(
            event_type=AuditEventType.AUTH_SUCCESS,
            user_id=user_id,
            action=f"auth_success_via_{method}",
            details={"method": method},
        )

    def log_auth_failure(self, user_id: Optional[str], reason: str) -> None:
        """Enregistre un échec d'authentification."""
        self.log_event(
            event_type=AuditEventType.AUTH_FAILURE,
            user_id=user_id,
            action="auth_failure",
            status="failure",
            details={"reason": reason},
        )

    def log_permission_denied(
        self,
        user_id: str,
        action: str,
        resource: str,
        required_role: Optional[str] = None,
    ) -> None:
        """Enregistre un accès refusé."""
        self.log_event(
            event_type=AuditEventType.PERMISSION_DENIED,
            user_id=user_id,
            action=action,
            resource=resource,
            status="denied",
            details={"required_role": required_role},
        )

    def log_secret_access(self, secret_id: str, user_id: Optional[str] = None) -> None:
        """Enregistre un accès à un secret."""
        self.log_event(
            event_type=AuditEventType.SECRET_ACCESS,
            user_id=user_id or "system",
            action="secret_access",
            resource=secret_id,
            details={"secret_id": secret_id},
        )

    def log_security_alert(self, alert_type: str, details: Dict[str, Any]) -> None:
        """Enregistre une alerte de sécurité."""
        self.log_event(
            event_type=AuditEventType.SECURITY_ALERT,
            action=alert_type,
            status="alert",
            details=details,
        )


# Singleton global
_audit_logger = None


def get_audit_logger() -> AuditLogger:
    """Retourne l'instance singleton du AuditLogger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit_log(
    event_type: AuditEventType,
    action: Optional[str] = None,
    resource_from_arg: Optional[int] = None,
):
    """
    Décorateur pour auditer automatiquement les appels de fonction.

    Utilisation :
        @audit_log(AuditEventType.API_CALL, action="create_job")
        async def create_job(user: dict, job_data: dict):
            return {"id": "job_123"}
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user_id = kwargs.get("user", {}).get("uid") if "user" in kwargs else None

            try:
                result = (
                    await func(*args, **kwargs)
                    if hasattr(func, "__await__")
                    else func(*args, **kwargs)
                )

                audit_logger = get_audit_logger()
                audit_logger.log_event(
                    event_type=event_type,
                    user_id=user_id,
                    action=action or func.__name__,
                    status="success",
                )

                return result
            except Exception as e:
                audit_logger = get_audit_logger()
                audit_logger.log_event(
                    event_type=event_type,
                    user_id=user_id,
                    action=action or func.__name__,
                    status="error",
                    details={"error": str(e)},
                )
                raise

        return wrapper

    return decorator
