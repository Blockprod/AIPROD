"""
OpenTelemetry Distributed Tracing for AIPROD
=============================================

Provides end-to-end trace propagation across API Gateway → Job Orchestrator
→ GPU Worker → Storage.  Each pipeline stage is a child span with attributes.

Exporters:
- OTLP (to Jaeger / Grafana Tempo)
- Console (debug / local development)

Falls back to no-op if opentelemetry-sdk is not installed.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional


# ---------------------------------------------------------------------------
# OpenTelemetry import with graceful fallback
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


# ---------------------------------------------------------------------------
# No-op spans for graceful degradation
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """No-op span when OpenTelemetry is not available."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, *args, **kwargs) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        pass

    def end(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoOpTracer:
    """No-op tracer when OpenTelemetry is not available."""

    def start_as_current_span(self, name: str, **kwargs):
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs):
        return _NoOpSpan()


# ---------------------------------------------------------------------------
# Tracer manager
# ---------------------------------------------------------------------------


class TracingManager:
    """
    Manages OpenTelemetry tracing configuration.

    Initializes the global TracerProvider with OTLP or console exporter.
    Provides convenience methods for creating spans.
    """

    def __init__(
        self,
        service_name: str = "aiprod",
        service_version: str = "3.0.0",
        environment: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
        console_export: bool = False,
    ):
        self._service_name = service_name
        self._environment = environment or os.environ.get("AIPROD_ENV", "development")
        self._initialized = False

        if _HAS_OTEL:
            self._init_otel(service_name, service_version, otlp_endpoint, console_export)
            self._tracer = trace.get_tracer(service_name, service_version)
            self._initialized = True
        else:
            self._tracer = _NoOpTracer()

    def _init_otel(
        self,
        service_name: str,
        service_version: str,
        otlp_endpoint: Optional[str],
        console_export: bool,
    ) -> None:
        """Initialize OpenTelemetry TracerProvider."""
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: service_name,
            ResourceAttributes.SERVICE_VERSION: service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self._environment,
        })

        provider = TracerProvider(resource=resource)

        # OTLP exporter (Jaeger / Tempo)
        otlp_endpoint = otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            except ImportError:
                pass  # OTLP exporter not installed

        # Console exporter (for local dev)
        if console_export:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

        trace.set_tracer_provider(provider)

    @property
    def tracer(self):
        """Get the underlying tracer."""
        return self._tracer

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Any, None, None]:
        """
        Create a traced span.

        Usage:
            with tracing.span("denoise_step", {"step": 42}) as span:
                result = denoise(latent)
                span.set_attribute("quality_score", 0.95)
        """
        if self._initialized:
            with self._tracer.start_as_current_span(name) as span:
                if attributes:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                try:
                    yield span
                except Exception as e:
                    span.record_exception(e)
                    from opentelemetry.trace import StatusCode
                    span.set_status(StatusCode.ERROR, str(e))
                    raise
        else:
            yield _NoOpSpan()

    @contextmanager
    def pipeline_trace(
        self,
        job_id: str,
        prompt: str,
        resolution: str,
        duration_sec: float,
    ) -> Generator[Any, None, None]:
        """
        Create a root span for an entire pipeline execution.

        Usage:
            with tracing.pipeline_trace("job-123", "A sunset...", "1080p", 5.0) as span:
                # ... all pipeline stages run as child spans
        """
        attrs = {
            "aiprod.job_id": job_id,
            "aiprod.prompt_length": len(prompt),
            "aiprod.resolution": resolution,
            "aiprod.duration_sec": duration_sec,
        }
        with self.span("pipeline.execute", attrs) as span:
            yield span

    def stage_span(self, stage_name: str, **kwargs) -> Any:
        """Create a span for a pipeline stage (text_encode, denoise, etc.)."""
        attrs = {f"aiprod.stage.{k}": v for k, v in kwargs.items()}
        return self.span(f"pipeline.{stage_name}", attrs)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_tracing: Optional[TracingManager] = None


def get_tracing(
    service_name: str = "aiprod",
    otlp_endpoint: Optional[str] = None,
) -> TracingManager:
    """Get or create the global TracingManager singleton."""
    global _tracing
    if _tracing is None:
        _tracing = TracingManager(
            service_name=service_name,
            otlp_endpoint=otlp_endpoint,
        )
    return _tracing
