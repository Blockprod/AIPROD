"""
Structured Logging avec Google Cloud Logging pour Phase 2 (Observabilité)
Logs au format JSON vers Cloud Logging avec correlation IDs et traces distribuées.
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from contextvars import ContextVar
from google.cloud import logging as cloud_logging
import os

# Context variables pour correlation IDs et trace IDs
correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
trace_id: ContextVar[str] = ContextVar("trace_id", default="")
user_id: ContextVar[str] = ContextVar("user_id", default="")


class StructuredLogger:
    """Logger structuré avec support Cloud Logging et correlation IDs."""

    def __init__(self, name: str, use_cloud_logging: bool = True):
        self.name = name
        self.use_cloud_logging = use_cloud_logging and self._has_gcp_credentials()

        # Python logging setup
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        if self.use_cloud_logging:
            # Setup Cloud Logging client
            self.cloud_client = cloud_logging.Client()
            self.cloud_logger = self.cloud_client.logger(name)
        else:
            self.cloud_client = None
            self.cloud_logger = None

    @staticmethod
    def _has_gcp_credentials() -> bool:
        """Vérifier si les credentials GCP sont disponibles."""
        try:
            google_cloud_project = os.getenv("GOOGLE_CLOUD_PROJECT")
            return google_cloud_project is not None
        except Exception:
            return False

    def _build_log_entry(
        self, message: str, level: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """Construire une entrée de log structurée."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "logger": self.name,
            "level": level,
            "message": message,
            "correlation_id": correlation_id.get(),
            "trace_id": trace_id.get(),
            "user_id": user_id.get(),
        }
        # Ajouter les fields supplémentaires
        entry.update(kwargs)
        return entry

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """Logger interne pour tous les niveaux."""
        entry = self._build_log_entry(message, level, **kwargs)

        # Log local (Python logging)
        log_level = getattr(logging, level)
        self.logger.log(log_level, json.dumps(entry))

        # Log vers Cloud Logging si disponible
        if self.use_cloud_logging and self.cloud_logger:
            try:
                self.cloud_logger.log_struct(entry, severity=level)
            except Exception as e:
                # Fallback: log l'erreur localement
                self.logger.error(f"Failed to log to Cloud Logging: {e}")

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log au niveau DEBUG."""
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log au niveau INFO."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log au niveau WARNING."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log au niveau ERROR."""
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log au niveau CRITICAL."""
        self._log("CRITICAL", message, **kwargs)


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Définir ou générer un correlation ID."""
    cid = cid or str(uuid.uuid4())
    correlation_id.set(cid)
    return cid


def set_trace_id(tid: Optional[str] = None) -> str:
    """Définir ou générer un trace ID."""
    tid = tid or str(uuid.uuid4())
    trace_id.set(tid)
    return tid


def set_user_id(uid: str) -> str:
    """Définir l'user ID pour le contexte."""
    user_id.set(uid)
    return uid


def get_correlation_id() -> str:
    """Récupérer le correlation ID du contexte."""
    return correlation_id.get()


def get_trace_id() -> str:
    """Récupérer le trace ID du contexte."""
    return trace_id.get()


def get_user_id() -> str:
    """Récupérer l'user ID du contexte."""
    return user_id.get()


# Instance globale
logger = StructuredLogger(__name__)
