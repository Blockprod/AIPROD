# Ensure prom_router is importable when using 'from src.utils.metrics_collector import prom_router'
__all__ = ["MetricsCollector", "prom_router"]
"""
Métriques et monitoring pour AIPROD V33
Collecte et exposition des métriques de performance.
"""
from datetime import datetime
from typing import Dict, Any, List
from src.utils.monitoring import logger
from prometheus_client import Gauge, Counter, generate_latest, REGISTRY, start_http_server
from fastapi import APIRouter, Response


# Définition des métriques Prometheus au niveau du module (singleton)
# Ces métriques sont globales et partagées par toutes les instances
PROM_PIPELINE_EXECUTIONS = Counter(
    'pipeline_executions_total',
    'Nombre total d’exécutions du pipeline'
)
PROM_PIPELINE_ERRORS = Counter(
    'pipeline_errors_total',
    'Nombre total d’erreurs du pipeline'
)
PROM_LATENCY_MS = Counter(
    'pipeline_latency_seconds_total',
    'Latence totale cumulée du pipeline (secondes)',
    ['type']  # permet de distinguer par type si besoin plus tard
)
PROM_COST_TOTAL = Counter(
    'pipeline_cost_total_dollars',
    'Coût total cumulé du pipeline en dollars'
)
PROM_QUALITY_TOTAL = Counter(
    'pipeline_quality_score_total',
    'Score qualité total cumulé (non normalisé)'
)
PROM_AVG_LATENCY_MS = Gauge(
    'pipeline_avg_latency_ms',
    'Latence moyenne du pipeline (ms)'
)
PROM_AVG_COST = Gauge(
    'pipeline_avg_cost_dollars',
    'Coût moyen par exécution du pipeline ($)'
)
PROM_AVG_QUALITY = Gauge(
    'pipeline_avg_quality_score',
    'Score de qualité moyen par exécution (0-1)'
)

# Initialisation des labels pour éviter les erreurs
PROM_LATENCY_MS.labels(type="execution")


class MetricsCollector:
    """
    Collecteur de métriques pour le pipeline AIPROD V33.
    
    Suit les KPIs suivants :
    - Nombre d'exécutions et d'erreurs
    - Latence (ms)
    - Coût ($)
    - Score de qualité (0-1)
    Met à jour à la fois une structure interne et les métriques Prometheus.
    """

    def __init__(self):
        self.metrics: Dict[str, float] = {
            "pipeline_executions": 0,
            "pipeline_errors": 0,
            "total_latency_ms": 0.0,
            "total_cost": 0.0,
            "total_quality_score": 0.0,
            "avg_latency_ms": 0.0,
            "avg_cost": 0.0,
            "avg_quality": 0.0
        }
        self.execution_history: List[Dict[str, Any]] = []

    def record_execution(
        self,
        result: Dict[str, Any],
        latency_ms: float,
        cost: float,
        quality: float
    ) -> None:
        """
        Enregistre une exécution réussie du pipeline.

        Args:
            result: Résultat du pipeline (dictionnaire quelconque)
            latency_ms: Temps d'exécution en millisecondes
            cost: Coût estimé en dollars
            quality: Score de qualité (généralement entre 0 et 1)
        """
        logger.info(
            f"MetricsCollector: execution recorded | "
            f"latency={latency_ms:.1f}ms | cost=${cost:.4f} | quality={quality:.3f}"
        )

        # Mise à jour des compteurs internes
        self.metrics["pipeline_executions"] += 1
        self.metrics["total_latency_ms"] += latency_ms
        self.metrics["total_cost"] += cost
        self.metrics["total_quality_score"] += quality

        # Recalcul des moyennes
        n = self.metrics["pipeline_executions"]
        if n > 0:
            self.metrics["avg_latency_ms"] = self.metrics["total_latency_ms"] / n
            self.metrics["avg_cost"] = self.metrics["total_cost"] / n
            self.metrics["avg_quality"] = self.metrics["total_quality_score"] / n

        # Mise à jour Prometheus
        PROM_PIPELINE_EXECUTIONS.inc()
        PROM_LATENCY_MS.labels(type="execution").inc(latency_ms / 1000.0)
        PROM_COST_TOTAL.inc(cost)
        PROM_QUALITY_TOTAL.inc(quality)

        PROM_AVG_LATENCY_MS.set(self.metrics["avg_latency_ms"])
        PROM_AVG_COST.set(self.metrics["avg_cost"])
        PROM_AVG_QUALITY.set(self.metrics["avg_quality"])

        # Historique
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "latency_ms": latency_ms,
            "cost": cost,
            "quality": quality,
            "result_summary": {k: str(v)[:100] for k, v in result.items()}
        })

    def record_error(self, error_message: str) -> None:
        """
        Enregistre une erreur survenue pendant une exécution.

        Args:
            error_message: Description ou traceback de l'erreur
        """
        logger.error(f"Pipeline error: {error_message}")
        self.metrics["pipeline_errors"] += 1
        PROM_PIPELINE_ERRORS.inc()

    def get_internal_metrics(self) -> Dict[str, Any]:
        """
        Retourne l'état actuel des métriques internes (non Prometheus).
        """
        return self.metrics.copy()

    def get_execution_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retourne les N dernières exécutions enregistrées.
        """
        return self.execution_history[-limit:]

    def check_alerts(self) -> Dict[str, bool]:
        """
        Vérifie si des seuils d'alerte sont dépassés.

        Returns:
            Dict avec les alertes actives (clé = nom alerte, valeur = True si déclenchée)
        """
        return {
            "high_latency": self.metrics["avg_latency_ms"] > 5000,
            "high_cost": self.metrics["avg_cost"] > 1.0,
            "low_quality": self.metrics["avg_quality"] < 0.60,
            "high_error_rate": (
                self.metrics["pipeline_errors"] > 10 and
                self.metrics["pipeline_executions"] > 0 and
                (self.metrics["pipeline_errors"] / self.metrics["pipeline_executions"]) > 0.10
            )
        }


# =============================================
# ROUTER PROMETHEUS (à inclure dans votre app FastAPI principale)
# =============================================
prom_router = APIRouter(prefix="/metrics", tags=["monitoring"])


@prom_router.get("")
async def prometheus_metrics():
    """
    Endpoint standard Prometheus /metrics
    Retourne les métriques au format text exposition 0.0.4
    """
    return Response(generate_latest(REGISTRY), media_type="text/plain")


# Fonction utilitaire pour démarrer le serveur de métriques
def start_metrics_server(port: int = 8000):
    """
    Démarre un serveur HTTP séparé pour les métriques Prometheus.
    À utiliser si vous ne voulez pas intégrer le router dans FastAPI.
    """
    start_http_server(port)
    logger.info(f"Metrics server started on port {port}")