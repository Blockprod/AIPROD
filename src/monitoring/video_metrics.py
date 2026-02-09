"""
Métriques Prometheus pour génération vidéo - AIPROD
Tracking des coûts, durées et tiers vidéo
"""

from prometheus_client import Counter, Gauge, Histogram
import logging

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════
# COUNTERS - Événements cumulatifs
# ═════════════════════════════════════════════════════════════════════════

video_generation_total = Counter(
    'video_generation_total',
    'Total number of video generations',
    ['tier', 'backend', 'status']
)

video_generation_cost_dollars = Counter(
    'video_generation_cost_dollars',
    'Total cost of video generations in dollars',
    ['tier', 'backend']
)

video_generation_duration_seconds = Counter(
    'video_generation_duration_seconds',
    'Total duration of video generations',
    ['tier']
)

# ═════════════════════════════════════════════════════════════════════════
# GAUGES - Snapshots de l'état actuel
# ═════════════════════════════════════════════════════════════════════════

video_generation_in_progress = Gauge(
    'video_generation_in_progress',
    'Currently processing video generations',
    ['tier']
)

average_cost_per_generation = Gauge(
    'average_cost_per_generation_dollars',
    'Average cost per generation',
    ['tier']
)

average_generation_time = Gauge(
    'average_generation_time_seconds',
    'Average generation time',
    ['tier']
)

# ═════════════════════════════════════════════════════════════════════════
# HISTOGRAMS - Distributions (percentiles, bucketing)
# ═════════════════════════════════════════════════════════════════════════

generation_duration_histogram = Histogram(
    'video_generation_duration_histogram_seconds',
    'Distribution of generation durations',
    ['tier'],
    buckets=(10, 20, 30, 45, 60, 90, 120, 300)
)

generation_cost_histogram = Histogram(
    'video_generation_cost_histogram_dollars',
    'Distribution of generation costs',
    ['tier'],
    buckets=(0.01, 0.02, 0.05, 0.08, 0.10, 0.20, 0.50, 1.00)
)


def record_video_generation_started(tier: str):
    """Track qui une génération vidéo a commencé."""
    video_generation_in_progress.labels(tier=tier).inc()


def record_video_generation_completed(
    tier: str,
    backend: str,
    cost_usd: float,
    duration_sec: int,
    status: str = "success"
):
    """
    Track qu'une génération vidéo est terminée.
    
    Args:
        tier: premium, balanced, ou economy
        backend: runway, veo-3.0, veo-2.0, etc.
        cost_usd: Coût réel en dollars
        duration_sec: Durée en secondes
        status: success, failed, cancelled
    """
    # Décrement in-progress
    video_generation_in_progress.labels(tier=tier).dec()
    
    # Enregistrer l'événement
    video_generation_total.labels(
        tier=tier,
        backend=backend,
        status=status
    ).inc()
    
    # Enregistrer les coûts
    video_generation_cost_dollars.labels(
        tier=tier,
        backend=backend
    ).inc(cost_usd)
    
    # Enregistrer les durées
    video_generation_duration_seconds.labels(tier=tier).inc(duration_sec)
    
    # Histogrammes (pour percentiles + distribution)
    generation_duration_histogram.labels(tier=tier).observe(duration_sec)
    generation_cost_histogram.labels(tier=tier).observe(cost_usd)
    
    logger.info(
        f"Video generation recorded: tier={tier}, "
        f"backend={backend}, cost=${cost_usd:.4f}, "
        f"duration={duration_sec}s, status={status}"
    )


def record_video_generation_failed(tier: str, backend: str = "unknown"):
    """Track qu'une génération vidéo a échoué."""
    video_generation_in_progress.labels(tier=tier).dec(amount=0)
    video_generation_total.labels(
        tier=tier,
        backend=backend,
        status="failed"
    ).inc()
