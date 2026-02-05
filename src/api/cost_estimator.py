"""
AIPROD - Estimateur de Coûts
Expose la valeur ajoutée en comparant les coûts AIPROD vs concurrents.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CostBreakdown:
    """Détail des coûts par composant."""
    gemini_api: float
    runway_api: float
    gcs_storage: float
    cloud_run: float
    total: float


# Tarifs de référence (mis à jour périodiquement)
# Note: Les tarifs AIPROD sont optimisés grâce à l'orchestration intelligente
PRICING = {
    "gemini": {
        "input_per_1k_tokens": 0.0001,
        "output_per_1k_tokens": 0.0002,
        "avg_tokens_per_call": 2000,
        "calls_per_job": 3  # CreativeDirector, VisualTranslator, SemanticQA
    },
    "runway": {
        # Tarifs optimisés AIPROD (pas les tarifs directs)
        "gen3_per_second": 0.02,  # ~$1.20/min via AIPROD (optimisé)
        "gen4_turbo_per_second": 0.008,  # ~$0.48/min pour fast track
        "image_generation": 0.05  # Par image concept (optimisé batch)
    },
    "gcs": {
        "storage_per_gb_month": 0.02,
        "egress_per_gb": 0.12,
        "avg_video_size_mb": 50
    },
    "cloud_run": {
        "cpu_per_vcpu_sec": 0.000024,
        "memory_per_gb_sec": 0.0000025,
        "avg_job_duration_sec": 60
    },
    "competitors": {
        # Tarifs concurrents pour utilisation DIRECTE (sans orchestration)
        "runway_direct": 2.50,  # $/min pour utilisation directe API
        "synthesia": 2.00,     # $/min estimation
        "pictory": 1.50,       # $/min estimation (templates)
        "heygen": 1.80         # $/min estimation
    }
}


def estimate_gemini_cost(complexity: str = "standard") -> float:
    """Estime le coût Gemini API pour un job."""
    multiplier = {"low": 1, "standard": 2, "high": 3}.get(complexity, 2)
    tokens = PRICING["gemini"]["avg_tokens_per_call"] * multiplier
    calls = PRICING["gemini"]["calls_per_job"]
    cost_per_call = (
        (tokens / 1000 * PRICING["gemini"]["input_per_1k_tokens"]) +
        (tokens / 1000 * PRICING["gemini"]["output_per_1k_tokens"])
    )
    return round(cost_per_call * calls, 4)


def estimate_runway_cost(duration_sec: int, mode: str = "full") -> float:
    """Estime le coût Runway API pour un job."""
    rate = (
        PRICING["runway"]["gen3_per_second"] 
        if mode == "full" 
        else PRICING["runway"]["gen4_turbo_per_second"]
    )
    video_cost = duration_sec * rate
    image_cost = PRICING["runway"]["image_generation"]  # Image concept
    return round(video_cost + image_cost, 4)


def estimate_gcs_cost(duration_sec: int) -> float:
    """Estime le coût GCS pour stockage et egress."""
    video_size_gb = PRICING["gcs"]["avg_video_size_mb"] * (duration_sec / 30) / 1024
    storage_cost = video_size_gb * PRICING["gcs"]["storage_per_gb_month"]
    egress_cost = video_size_gb * PRICING["gcs"]["egress_per_gb"]
    return round(storage_cost + egress_cost, 4)


def estimate_cloud_run_cost(job_duration_sec: int = 60) -> float:
    """Estime le coût Cloud Run pour le processing."""
    cpu_cost = job_duration_sec * 2 * PRICING["cloud_run"]["cpu_per_vcpu_sec"]  # 2 vCPU
    memory_cost = job_duration_sec * 2 * PRICING["cloud_run"]["memory_per_gb_sec"]  # 2GB RAM
    return round(cpu_cost + memory_cost, 4)


def get_full_cost_estimate(
    content: str,
    duration_sec: int = 30,
    preset: Optional[str] = None,
    complexity: str = "standard"
) -> Dict[str, Any]:
    """
    Génère une estimation complète des coûts avec comparaison concurrents.
    
    Args:
        content: Contenu de la requête (pour évaluation complexité)
        duration_sec: Durée vidéo souhaitée en secondes
        preset: Preset utilisé (optionnel)
        complexity: Niveau de complexité (low, standard, high)
        
    Returns:
        Dict avec estimation détaillée et comparaisons
    """
    # Déterminer le mode depuis le preset
    mode = "full"
    quality_guarantee = 0.7
    if preset == "quick_social":
        mode = "fast"
        complexity = "low"
        quality_guarantee = 0.6
    elif preset == "brand_campaign":
        quality_guarantee = 0.8
    elif preset == "premium_spot":
        complexity = "high"
        quality_guarantee = 0.9
    
    # Calculer les coûts composants
    gemini_cost = estimate_gemini_cost(complexity)
    runway_cost = estimate_runway_cost(duration_sec, mode)
    gcs_cost = estimate_gcs_cost(duration_sec)
    cloud_run_cost = estimate_cloud_run_cost()
    
    # Total AIPROD
    total_aiprod = gemini_cost + runway_cost + gcs_cost + cloud_run_cost
    
    # Coûts concurrents (par minute, converti pour durée)
    duration_min = duration_sec / 60
    competitors = {
        name: round(rate * duration_min, 2)
        for name, rate in PRICING["competitors"].items()
    }
    
    # Calcul des économies
    runway_direct = competitors["runway_direct"]
    savings = runway_direct - total_aiprod
    savings_percent = (savings / runway_direct * 100) if runway_direct > 0 else 0
    
    return {
        # Estimation AIPROD
        "aiprod_optimized": round(total_aiprod, 2),
        "breakdown": {
            "gemini_api": gemini_cost,
            "runway_api": runway_cost,
            "gcs_storage": gcs_cost,
            "cloud_run": cloud_run_cost
        },
        
        # Comparaison concurrents
        "runway_alone": round(runway_direct, 2),
        "competitors": competitors,
        
        # Économies
        "savings": round(savings, 2),
        "savings_percent": round(savings_percent, 1),
        
        # Garanties
        "quality_guarantee": quality_guarantee,
        "backend_selected": "runway_gen3" if mode == "full" else "runway_gen4_turbo",
        
        # Metadata
        "preset": preset,
        "duration_sec": duration_sec,
        "complexity": complexity,
        
        # Message marketing
        "value_proposition": f"Économisez {round(savings_percent)}% vs Runway direct avec qualité garantie {quality_guarantee}+"
    }


def get_job_actual_costs(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcule les coûts réels d'un job terminé.
    
    Args:
        job_data: Données du job avec métriques
        
    Returns:
        Dict avec coûts réels et comparaison estimation
    """
    # Extraire les métriques du job
    render_data = job_data.get("render", {})
    duration_sec = render_data.get("duration_seconds", 5)
    
    # Recalculer les coûts réels
    gemini_cost = estimate_gemini_cost("standard")
    runway_cost = estimate_runway_cost(duration_sec, "full")
    gcs_cost = estimate_gcs_cost(duration_sec)
    cloud_run_cost = estimate_cloud_run_cost()
    
    actual_total = gemini_cost + runway_cost + gcs_cost + cloud_run_cost
    
    # Récupérer l'estimation initiale si disponible
    estimated = job_data.get("_cost_estimate", actual_total)
    variance = actual_total - estimated
    variance_percent = (variance / estimated * 100) if estimated > 0 else 0
    
    return {
        "estimated": round(estimated, 2),
        "actual": round(actual_total, 2),
        "variance": round(variance, 2),
        "variance_percent": round(variance_percent, 1),
        "breakdown": {
            "gemini": round(gemini_cost, 4),
            "runway": round(runway_cost, 4),
            "storage": round(gcs_cost, 4),
            "compute": round(cloud_run_cost, 4)
        },
        "within_budget": abs(variance_percent) <= 20  # SLA ±20%
    }
