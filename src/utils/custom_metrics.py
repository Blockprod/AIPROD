"""
AIPROD - Custom Metrics for Cloud Monitoring

Module pour envoyer des métriques personnalisées à Google Cloud Monitoring.
Permet le suivi de la performance, qualité et coûts du pipeline.
"""

import os
import time
from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from src.utils.monitoring import logger

# Try to import Google Cloud Monitoring
try:
    from google.cloud import monitoring_v3  # type: ignore
    from google.protobuf import timestamp_pb2  # type: ignore

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    logger.warning(
        "google-cloud-monitoring not installed. Metrics will be logged only."
    )


class MetricType(str, Enum):
    """Types de métriques supportées."""

    GAUGE = "gauge"  # Valeur ponctuelle (ex: quality_score)
    COUNTER = "counter"  # Compteur incrémental (ex: jobs_completed)
    DISTRIBUTION = "distribution"  # Distribution (ex: latencies)


@dataclass
class MetricPoint:
    """Représente un point de métrique."""

    name: str
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class CustomMetricsCollector:
    """
    Collecteur de métriques personnalisées pour Cloud Monitoring.

    Usage:
        collector = CustomMetricsCollector()
        collector.report_metric("pipeline_duration", 45.2, {"preset": "quick_social"})
        collector.report_metric("quality_score", 0.87, {"job_id": "abc123"})
    """

    # Métriques prédéfinies avec leurs types
    METRIC_DEFINITIONS = {
        # Performance
        "pipeline_duration": {
            "type": MetricType.GAUGE,
            "unit": "s",
            "description": "Durée totale du pipeline en secondes",
        },
        "agent_duration": {
            "type": MetricType.GAUGE,
            "unit": "s",
            "description": "Durée d'un agent spécifique",
        },
        "render_duration": {
            "type": MetricType.GAUGE,
            "unit": "s",
            "description": "Durée du rendu vidéo",
        },
        # Qualité
        "quality_score": {
            "type": MetricType.GAUGE,
            "unit": "1",
            "description": "Score de qualité (0-1)",
        },
        "semantic_qa_score": {
            "type": MetricType.GAUGE,
            "unit": "1",
            "description": "Score QA sémantique",
        },
        "technical_qa_score": {
            "type": MetricType.GAUGE,
            "unit": "1",
            "description": "Score QA technique",
        },
        # Coûts
        "cost_per_job": {
            "type": MetricType.GAUGE,
            "unit": "USD",
            "description": "Coût par job en dollars",
        },
        "cost_per_minute": {
            "type": MetricType.GAUGE,
            "unit": "USD",
            "description": "Coût par minute de vidéo",
        },
        "cost_savings": {
            "type": MetricType.GAUGE,
            "unit": "USD",
            "description": "Économies vs Runway direct",
        },
        # Compteurs
        "jobs_created": {
            "type": MetricType.COUNTER,
            "unit": "1",
            "description": "Nombre de jobs créés",
        },
        "jobs_completed": {
            "type": MetricType.COUNTER,
            "unit": "1",
            "description": "Nombre de jobs terminés",
        },
        "jobs_failed": {
            "type": MetricType.COUNTER,
            "unit": "1",
            "description": "Nombre de jobs échoués",
        },
        "cache_hits": {
            "type": MetricType.COUNTER,
            "unit": "1",
            "description": "Hits du consistency cache",
        },
        "cache_misses": {
            "type": MetricType.COUNTER,
            "unit": "1",
            "description": "Misses du consistency cache",
        },
        # Backend
        "backend_requests": {
            "type": MetricType.COUNTER,
            "unit": "1",
            "description": "Requêtes par backend",
        },
        "backend_errors": {
            "type": MetricType.COUNTER,
            "unit": "1",
            "description": "Erreurs par backend",
        },
        "backend_fallbacks": {
            "type": MetricType.COUNTER,
            "unit": "1",
            "description": "Fallbacks vers backend secondaire",
        },
    }

    def __init__(self, project_id: Optional[str] = None):
        """
        Initialise le collecteur de métriques.

        Args:
            project_id: ID du projet GCP. Auto-détecté si non fourni.
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID", "aiprod-484120")
        self.project_name = f"projects/{self.project_id}"
        self.metric_prefix = "custom.googleapis.com/aiprod"

        self._client = None
        self._buffer: list[MetricPoint] = []
        self._buffer_size = 10  # Flush every 10 metrics

        # Counters locaux pour métriques incrémentales
        self._counters: Dict[str, int] = {}

        if MONITORING_AVAILABLE:
            try:
                self._client = monitoring_v3.MetricServiceClient()
                logger.info(
                    f"CustomMetrics: Connected to Cloud Monitoring ({self.project_id})"
                )
            except Exception as e:
                logger.warning(
                    f"CustomMetrics: Failed to connect to Cloud Monitoring: {e}"
                )
                self._client = None

    def report_metric(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
        flush: bool = False,
    ) -> None:
        """
        Envoie une métrique à Cloud Monitoring.

        Args:
            name: Nom de la métrique (ex: "pipeline_duration")
            value: Valeur numérique
            labels: Labels additionnels (ex: {"preset": "quick_social"})
            flush: Force l'envoi immédiat

        Example:
            report_metric("pipeline_duration", 45.2, {"preset": "quick_social"})
            report_metric("quality_score", 0.87, {"job_id": "abc123"})
            report_metric("cost_per_minute", 0.92)
        """
        labels = labels or {}

        # Déterminer le type de métrique
        metric_def = self.METRIC_DEFINITIONS.get(name, {})
        metric_type = metric_def.get("type", MetricType.GAUGE)

        # Pour les counters, incrémenter
        if metric_type == MetricType.COUNTER:
            counter_key = (
                f"{name}:{':'.join(f'{k}={v}' for k, v in sorted(labels.items()))}"
            )
            self._counters[counter_key] = self._counters.get(counter_key, 0) + int(
                value
            )
            value = self._counters[counter_key]

        # Créer le point
        point = MetricPoint(
            name=name, value=value, labels=labels, metric_type=metric_type
        )

        # Logger toujours
        logger.info(f"Metric: {name}={value} {labels}")

        # Ajouter au buffer
        self._buffer.append(point)

        # Flush si nécessaire
        if flush or len(self._buffer) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Envoie le buffer de métriques à Cloud Monitoring."""
        if not self._buffer:
            return

        if not self._client:
            # Mode local: juste vider le buffer
            self._buffer.clear()
            return

        try:
            for point in self._buffer:
                self._send_metric_point(point)
            logger.debug(f"CustomMetrics: Flushed {len(self._buffer)} metrics")
        except Exception as e:
            logger.error(f"CustomMetrics: Failed to flush metrics: {e}")
        finally:
            self._buffer.clear()

    def _send_metric_point(self, point: MetricPoint) -> None:
        """Envoie un point de métrique individuel."""
        if not self._client:
            return

        try:
            series = monitoring_v3.TimeSeries()
            series.metric.type = f"{self.metric_prefix}/{point.name}"

            # Ajouter les labels
            for key, value in point.labels.items():
                series.metric.labels[key] = str(value)

            # Resource
            series.resource.type = "global"
            series.resource.labels["project_id"] = self.project_id

            # Point de données
            data_point = monitoring_v3.Point()

            # Timestamp
            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 10**9)
            data_point.interval.end_time.seconds = seconds
            data_point.interval.end_time.nanos = nanos

            # Valeur
            if isinstance(point.value, float):
                data_point.value.double_value = point.value
            else:
                data_point.value.int64_value = int(point.value)

            series.points.append(data_point)

            # Envoyer
            self._client.create_time_series(
                name=self.project_name, time_series=[series]
            )

        except Exception as e:
            logger.error(f"CustomMetrics: Error sending {point.name}: {e}")

    def report_pipeline_metrics(
        self,
        job_id: str,
        preset: str,
        duration_sec: float,
        quality_score: float,
        cost: float,
        backend: str,
    ) -> None:
        """
        Raccourci pour reporter toutes les métriques d'un pipeline.

        Args:
            job_id: ID du job
            preset: Preset utilisé
            duration_sec: Durée totale
            quality_score: Score qualité final
            cost: Coût total
            backend: Backend utilisé
        """
        common_labels = {"job_id": job_id, "preset": preset, "backend": backend}

        self.report_metric("pipeline_duration", duration_sec, common_labels)
        self.report_metric("quality_score", quality_score, common_labels)
        self.report_metric("cost_per_job", cost, common_labels)
        self.report_metric("jobs_completed", 1, {"preset": preset})

        # Flush immédiat pour les métriques importantes
        self._flush_buffer()

    def report_error(
        self,
        error_type: str,
        job_id: Optional[str] = None,
        backend: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """
        Reporte une erreur.

        Args:
            error_type: Type d'erreur (ex: "render_failed", "qa_rejected")
            job_id: ID du job concerné
            backend: Backend concerné
            details: Détails supplémentaires
        """
        labels = {"error_type": error_type}
        if job_id:
            labels["job_id"] = job_id
        if backend:
            labels["backend"] = backend

        self.report_metric("jobs_failed", 1, labels, flush=True)

        if backend:
            self.report_metric("backend_errors", 1, {"backend": backend})

        logger.error(f"Pipeline Error: {error_type} - {details}")

    def report_cache_hit(self, brand_id: str, cache_type: str = "consistency") -> None:
        """Reporte un cache hit."""
        self.report_metric(
            "cache_hits", 1, {"brand_id": brand_id, "cache_type": cache_type}
        )

    def report_cache_miss(self, brand_id: str, cache_type: str = "consistency") -> None:
        """Reporte un cache miss."""
        self.report_metric(
            "cache_misses", 1, {"brand_id": brand_id, "cache_type": cache_type}
        )

    def report_backend_fallback(
        self, from_backend: str, to_backend: str, reason: str
    ) -> None:
        """Reporte un fallback entre backends."""
        self.report_metric(
            "backend_fallbacks",
            1,
            {"from_backend": from_backend, "to_backend": to_backend, "reason": reason},
            flush=True,
        )
        logger.warning(f"Backend Fallback: {from_backend} -> {to_backend} ({reason})")


# Instance globale
_metrics_collector: Optional[CustomMetricsCollector] = None


def get_metrics_collector() -> CustomMetricsCollector:
    """Retourne l'instance globale du collecteur de métriques."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = CustomMetricsCollector()
    return _metrics_collector


def report_metric(
    name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None
) -> None:
    """
    Fonction helper pour reporter une métrique.

    Usage:
        report_metric("pipeline_duration", 45.2, {"preset": "quick_social"})
        report_metric("quality_score", 0.87, {"job_id": "abc123"})
        report_metric("cost_per_minute", 0.92)
    """
    get_metrics_collector().report_metric(name, value, labels)


def report_pipeline_complete(
    job_id: str,
    preset: str,
    duration_sec: float,
    quality_score: float,
    cost: float,
    backend: str,
) -> None:
    """Reporte la complétion d'un pipeline."""
    get_metrics_collector().report_pipeline_metrics(
        job_id, preset, duration_sec, quality_score, cost, backend
    )


def report_error(
    error_type: str,
    job_id: Optional[str] = None,
    backend: Optional[str] = None,
    details: Optional[str] = None,
) -> None:
    """Reporte une erreur."""
    get_metrics_collector().report_error(error_type, job_id, backend, details)
