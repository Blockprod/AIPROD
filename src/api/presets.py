"""
AIPROD - Système de Presets
Simplifie l'expérience utilisateur en cachant la complexité des 11 agents.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PresetTier(str, Enum):
    """Tiers de presets disponibles."""
    QUICK_SOCIAL = "quick_social"
    BRAND_CAMPAIGN = "brand_campaign"
    PREMIUM_SPOT = "premium_spot"


@dataclass
class PresetConfig:
    """Configuration d'un preset."""
    name: str
    description: str
    pipeline_mode: str  # "fast" ou "full"
    quality_threshold: float
    max_duration_sec: int
    max_cost_per_minute: float
    allow_icc: bool  # Interactive Creative Control
    consistency_cache: bool
    multi_review: bool
    priority: str
    estimated_cost: float  # Coût estimé par vidéo de 30s


# Définition des presets
PRESETS: Dict[str, PresetConfig] = {
    PresetTier.QUICK_SOCIAL.value: PresetConfig(
        name="Quick Social",
        description="Génération rapide pour réseaux sociaux (30s max)",
        pipeline_mode="fast",
        quality_threshold=0.6,
        max_duration_sec=30,
        max_cost_per_minute=0.50,
        allow_icc=False,
        consistency_cache=False,
        multi_review=False,
        priority="low",
        estimated_cost=0.30
    ),
    PresetTier.BRAND_CAMPAIGN.value: PresetConfig(
        name="Brand Campaign",
        description="Pipeline complet avec ICC pour campagnes de marque",
        pipeline_mode="full",
        quality_threshold=0.8,
        max_duration_sec=120,
        max_cost_per_minute=0.95,
        allow_icc=True,
        consistency_cache=True,
        multi_review=False,
        priority="medium",
        estimated_cost=0.90
    ),
    PresetTier.PREMIUM_SPOT.value: PresetConfig(
        name="Premium Spot",
        description="Qualité maximale avec multi-review pour spots publicitaires",
        pipeline_mode="full",
        quality_threshold=0.9,
        max_duration_sec=180,
        max_cost_per_minute=1.50,
        allow_icc=True,
        consistency_cache=True,
        multi_review=True,
        priority="high",
        estimated_cost=1.50
    )
}


def get_preset(preset_name: str) -> Optional[PresetConfig]:
    """
    Récupère la configuration d'un preset par son nom.
    
    Args:
        preset_name: Nom du preset (quick_social, brand_campaign, premium_spot)
        
    Returns:
        PresetConfig si trouvé, None sinon
    """
    return PRESETS.get(preset_name)


def get_all_presets() -> Dict[str, Dict[str, Any]]:
    """
    Récupère tous les presets disponibles avec leurs configurations.
    
    Returns:
        Dict des presets avec leurs détails
    """
    return {
        name: {
            "name": config.name,
            "description": config.description,
            "pipeline_mode": config.pipeline_mode,
            "quality_threshold": config.quality_threshold,
            "max_duration_sec": config.max_duration_sec,
            "max_cost_per_minute": config.max_cost_per_minute,
            "allow_icc": config.allow_icc,
            "consistency_cache": config.consistency_cache,
            "multi_review": config.multi_review,
            "priority": config.priority,
            "estimated_cost": config.estimated_cost
        }
        for name, config in PRESETS.items()
    }


def apply_preset_to_request(request_data: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
    """
    Applique un preset à une requête pipeline.
    
    Args:
        request_data: Données de la requête originale
        preset_name: Nom du preset à appliquer
        
    Returns:
        Requête enrichie avec les configurations du preset
    """
    preset = get_preset(preset_name)
    if not preset:
        return request_data
    
    # Enrichir la requête avec les paramètres du preset
    enriched = request_data.copy()
    enriched["_preset"] = preset_name
    enriched["_preset_config"] = {
        "pipeline_mode": preset.pipeline_mode,
        "quality_threshold": preset.quality_threshold,
        "max_duration_sec": preset.max_duration_sec,
        "max_cost_per_minute": preset.max_cost_per_minute,
        "allow_icc": preset.allow_icc,
        "consistency_cache": preset.consistency_cache,
        "multi_review": preset.multi_review
    }
    
    # Ajuster la priorité si non spécifiée
    if enriched.get("priority") == "low":
        enriched["priority"] = preset.priority
    
    return enriched


from typing import Union

def estimate_cost_for_preset(preset_name: str, duration_sec: int = 30) -> Dict[str, Union[float, str]]:
    """
    Estime le coût pour un preset donné.
    
    Args:
        preset_name: Nom du preset
        duration_sec: Durée estimée en secondes
        
    Returns:
        Dict avec coût estimé et comparaison
    """
    preset = get_preset(preset_name)
    if not preset:
        return {"error": "Preset not found"}
    
    duration_min = duration_sec / 60
    
    # Coût AIPROD optimisé
    aiprod_cost = duration_min * preset.max_cost_per_minute
    
    # Coût Runway seul (estimation benchmark)
    runway_base_cost = 2.50  # ~$2.50/min pour Runway Gen-3 direct
    runway_cost = duration_min * runway_base_cost
    
    return {
        "preset": preset_name,
        "duration_sec": duration_sec,
        "aiprod_optimized": round(aiprod_cost, 2),
        "runway_alone": round(runway_cost, 2),
        "savings": round(runway_cost - aiprod_cost, 2),
        "savings_percent": round((1 - aiprod_cost / runway_cost) * 100, 1) if runway_cost > 0 else 0,
        "quality_guarantee": preset.quality_threshold,
        "backend_selected": "runway_gen3" if preset.pipeline_mode == "full" else "runway_gen4_turbo"
    }
