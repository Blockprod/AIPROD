"""
Prometheus Metrics for AIPROD
==============================

Exposes application-level and model-quality metrics via Prometheus client.

Metric categories:
- Infrastructure: request latency, throughput, error rate, queue depth
- GPU: utilization, VRAM usage, temperature (collected from DCGM)
- Model quality: FID, CLIP-Score, generation quality score, drift
- Business: cost per video, revenue per video, margin
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Generator, Optional


# ---------------------------------------------------------------------------
# Prometheus client abstraction (graceful degradation if not installed)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        Summary,
        start_http_server,
        generate_latest,
    )

    _HAS_PROMETHEUS = True
except ImportError:
    _HAS_PROMETHEUS = False


class _NoOpMetric:
    """No-op metric stub when prometheus_client is not installed."""

    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        pass

    def dec(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------


class MetricsRegistry:
    """
    Central metrics registry for the AIPROD platform.

    Creates and manages Prometheus metrics. Falls back to no-ops if
    prometheus_client is absent.
    """

    def __init__(self, registry: Optional[object] = None):
        self._has_prometheus = _HAS_PROMETHEUS
        if self._has_prometheus:
            self._registry = registry or CollectorRegistry()
        else:
            self._registry = None

        self._build_info = self._make_info(
            "aiprod_build", "AIPROD build information"
        )
        self._set_build_info()

        # ---- Request metrics ------------------------------------------------
        self.request_total = self._make_counter(
            "aiprod_requests_total",
            "Total API requests",
            ["method", "endpoint", "status"],
        )
        self.request_latency = self._make_histogram(
            "aiprod_request_duration_seconds",
            "Request duration in seconds",
            ["method", "endpoint"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        )
        self.request_in_progress = self._make_gauge(
            "aiprod_requests_in_progress",
            "Requests currently being processed",
            ["endpoint"],
        )

        # ---- Pipeline metrics -----------------------------------------------
        self.pipeline_stage_duration = self._make_histogram(
            "aiprod_pipeline_stage_seconds",
            "Duration of each pipeline stage",
            ["stage"],
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
        )
        self.pipeline_total_duration = self._make_histogram(
            "aiprod_pipeline_total_seconds",
            "Total pipeline execution time",
            ["resolution", "duration_class"],
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
        )
        self.pipeline_errors = self._make_counter(
            "aiprod_pipeline_errors_total",
            "Pipeline errors by stage and type",
            ["stage", "error_type"],
        )

        # ---- GPU metrics ----------------------------------------------------
        self.gpu_utilization = self._make_gauge(
            "aiprod_gpu_utilization_percent",
            "GPU compute utilization percentage",
            ["device"],
        )
        self.gpu_vram_used = self._make_gauge(
            "aiprod_gpu_vram_used_bytes",
            "GPU VRAM used in bytes",
            ["device"],
        )
        self.gpu_vram_total = self._make_gauge(
            "aiprod_gpu_vram_total_bytes",
            "GPU VRAM total in bytes",
            ["device"],
        )
        self.gpu_temperature = self._make_gauge(
            "aiprod_gpu_temperature_celsius",
            "GPU temperature in Celsius",
            ["device"],
        )

        # ---- Model quality metrics ------------------------------------------
        self.model_fid_score = self._make_gauge(
            "aiprod_model_fid_score",
            "Fréchet Inception Distance (lower is better)",
            ["model_name", "model_version"],
        )
        self.model_clip_score = self._make_gauge(
            "aiprod_model_clip_score",
            "CLIP similarity score (higher is better)",
            ["model_name", "model_version"],
        )
        self.model_quality_score = self._make_gauge(
            "aiprod_model_quality_score",
            "Composite generation quality score [0-1]",
            ["model_name", "model_version"],
        )
        self.model_inference_count = self._make_counter(
            "aiprod_model_inference_total",
            "Total model inferences",
            ["model_name", "model_version"],
        )

        # ---- Queue metrics --------------------------------------------------
        self.queue_depth = self._make_gauge(
            "aiprod_queue_depth",
            "Number of jobs in queue",
            ["tier"],
        )
        self.queue_wait_time = self._make_histogram(
            "aiprod_queue_wait_seconds",
            "Time jobs spend waiting in queue",
            ["tier"],
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
        )

        # ---- Business metrics -----------------------------------------------
        self.cost_per_video = self._make_histogram(
            "aiprod_cost_per_video_usd",
            "Internal compute cost per generated video (USD)",
            ["resolution"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
        )
        self.revenue_per_video = self._make_histogram(
            "aiprod_revenue_per_video_usd",
            "Revenue per generated video (USD)",
            ["tier"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0),
        )

    # ---- Factory helpers ----------------------------------------------------

    def _make_counter(self, name, doc, labels):
        if self._has_prometheus:
            return Counter(name, doc, labels, registry=self._registry)
        return _NoOpMetric()

    def _make_gauge(self, name, doc, labels):
        if self._has_prometheus:
            return Gauge(name, doc, labels, registry=self._registry)
        return _NoOpMetric()

    def _make_histogram(self, name, doc, labels, buckets=None):
        if self._has_prometheus:
            kwargs = {"registry": self._registry}
            if buckets:
                kwargs["buckets"] = buckets
            return Histogram(name, doc, labels, **kwargs)
        return _NoOpMetric()

    def _make_info(self, name, doc):
        if self._has_prometheus:
            return Info(name, doc, registry=self._registry)
        return _NoOpMetric()

    def _set_build_info(self):
        if self._has_prometheus:
            self._build_info.info({
                "version": "3.0.0",
                "phase": "3",
                "component": "aiprod-platform",
            })

    # ---- Convenience context managers ---------------------------------------

    @contextmanager
    def track_request(self, method: str, endpoint: str) -> Generator[None, None, None]:
        """Context manager to track request latency and count."""
        self.request_in_progress.labels(endpoint=endpoint).inc()
        start = time.perf_counter()
        status = "200"
        try:
            yield
        except Exception:
            status = "500"
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.request_latency.labels(method=method, endpoint=endpoint).observe(elapsed)
            self.request_total.labels(method=method, endpoint=endpoint, status=status).inc()
            self.request_in_progress.labels(endpoint=endpoint).dec()

    @contextmanager
    def track_pipeline_stage(self, stage: str) -> Generator[None, None, None]:
        """Context manager to track pipeline stage duration."""
        start = time.perf_counter()
        try:
            yield
        except Exception as e:
            self.pipeline_errors.labels(stage=stage, error_type=type(e).__name__).inc()
            raise
        finally:
            elapsed = time.perf_counter() - start
            self.pipeline_stage_duration.labels(stage=stage).observe(elapsed)

    # ---- Server -------------------------------------------------------------

    def start_server(self, port: int = 9090) -> None:
        """Start Prometheus HTTP metrics server."""
        if self._has_prometheus:
            start_http_server(port, registry=self._registry)

    def generate_metrics(self) -> bytes:
        """Generate metrics output for scraping."""
        if self._has_prometheus:
            return generate_latest(self._registry)
        return b""


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_metrics: Optional[MetricsRegistry] = None


def get_metrics() -> MetricsRegistry:
    """Get or create the global MetricsRegistry singleton."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsRegistry()
    return _metrics
