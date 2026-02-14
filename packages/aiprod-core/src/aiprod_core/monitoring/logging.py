"""
Structured JSON Logging for AIPROD
====================================

Production-grade structured logging compatible with:
- Grafana Loki (JSON ingestion)
- AWS CloudWatch Logs
- Google Cloud Logging
- ELK stack (Elasticsearch/Logstash/Kibana)

Features:
- JSON-formatted log entries with consistent schema
- Correlation ID propagation (request_id / job_id / trace_id)
- Log levels with severity mapping
- Contextual bindings (user, tenant, stage)
- Performance timing helpers
- Sensitive data redaction

Falls back to stdlib logging if structlog is not installed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional


# ---------------------------------------------------------------------------
# structlog import with graceful fallback
# ---------------------------------------------------------------------------

try:
    import structlog

    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False


# ---------------------------------------------------------------------------
# Sensitive field redaction
# ---------------------------------------------------------------------------

_REDACT_FIELDS = frozenset({
    "password", "secret", "api_key", "token", "authorization",
    "credit_card", "ssn", "credentials",
})


def _redact(data: Dict[str, Any]) -> Dict[str, Any]:
    """Redact sensitive fields in a log event dict."""
    out = {}
    for k, v in data.items():
        if k.lower() in _REDACT_FIELDS:
            out[k] = "***REDACTED***"
        elif isinstance(v, dict):
            out[k] = _redact(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# JSON formatter for stdlib logging
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Merge extra fields
        for k in ("request_id", "job_id", "user_id", "tenant_id", "stage", "trace_id"):
            v = getattr(record, k, None)
            if v is not None:
                entry[k] = v

        # Exception info
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(_redact(entry), default=str)


# ---------------------------------------------------------------------------
# Logger wrapper
# ---------------------------------------------------------------------------


class StructuredLogger:
    """
    Structured logging wrapper.

    Uses structlog if available, stdlib logging otherwise.
    All log entries are JSON-formatted with consistent schema.
    """

    def __init__(
        self,
        name: str = "aiprod",
        level: str = "INFO",
        bindings: Optional[Dict[str, Any]] = None,
    ):
        self._name = name
        self._level = getattr(logging, level.upper(), logging.INFO)
        self._bindings: Dict[str, Any] = bindings or {}

        if _HAS_STRUCTLOG:
            self._logger = self._init_structlog(name)
        else:
            self._logger = self._init_stdlib(name)

    def _init_structlog(self, name: str):
        """Initialize structlog with JSON rendering."""
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(self._level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        logger = structlog.get_logger(name)
        if self._bindings:
            logger = logger.bind(**self._bindings)
        return logger

    def _init_stdlib(self, name: str) -> logging.Logger:
        """Initialize stdlib logger with JSON formatter."""
        logger = logging.getLogger(name)
        logger.setLevel(self._level)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JSONFormatter())
            logger.addHandler(handler)

        return logger

    # ---- Log methods --------------------------------------------------------

    def info(self, message: str, **kwargs) -> None:
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self._log("error", message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        self._log("debug", message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self._log("critical", message, **kwargs)

    def _log(self, level: str, message: str, **kwargs) -> None:
        merged = {**self._bindings, **kwargs}
        merged = _redact(merged)

        if _HAS_STRUCTLOG:
            getattr(self._logger, level)(message, **merged)
        else:
            extra = {k: v for k, v in merged.items()}
            record_kwargs = {"extra": extra}
            getattr(self._logger, level)(
                f"{message} | {json.dumps(extra, default=str)}"
            )

    # ---- Context binding ----------------------------------------------------

    def bind(self, **kwargs) -> "StructuredLogger":
        """Create a new logger with additional bound fields."""
        new_bindings = {**self._bindings, **kwargs}
        new_logger = StructuredLogger.__new__(StructuredLogger)
        new_logger._name = self._name
        new_logger._level = self._level
        new_logger._bindings = new_bindings
        new_logger._logger = self._logger
        if _HAS_STRUCTLOG:
            new_logger._logger = self._logger.bind(**kwargs)
        return new_logger

    # ---- Timing helpers -----------------------------------------------------

    @contextmanager
    def timed(self, operation: str, **kwargs) -> Generator[None, None, None]:
        """
        Context manager that logs operation duration.

        Usage:
            with logger.timed("denoise", step=42):
                result = denoise(latent)
        """
        start = time.perf_counter()
        self.info(f"{operation}.start", **kwargs)
        try:
            yield
        except Exception as e:
            elapsed = time.perf_counter() - start
            self.error(
                f"{operation}.error",
                duration_sec=round(elapsed, 4),
                error_type=type(e).__name__,
                error_message=str(e),
                **kwargs,
            )
            raise
        else:
            elapsed = time.perf_counter() - start
            self.info(f"{operation}.complete", duration_sec=round(elapsed, 4), **kwargs)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_logger: Optional[StructuredLogger] = None


def get_logger(
    name: str = "aiprod",
    level: str = "INFO",
    **bindings,
) -> StructuredLogger:
    """Get or create the global StructuredLogger singleton."""
    global _logger
    if _logger is None:
        env_level = os.environ.get("AIPROD_LOG_LEVEL", level)
        _logger = StructuredLogger(name=name, level=env_level, bindings=bindings)
    return _logger
