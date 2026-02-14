"""
AIPROD External Schema Definitions
===================================

TypedDict definitions for external AIPROD production manifest format.
These schemas represent the contract between AIPROD and external systems.
"""

from typing import TypedDict, Dict, Any, List, Optional


class AIPRODScene(TypedDict, total=False):
    """Single scene in AIPROD production manifest."""
    scene_id: str
    duration_sec: float
    description: str
    camera_movement: str
    lighting_style: str
    mood: str
    characters: List[str]
    props: List[str]
    location: str
    time_of_day: str
    weather: str
    visual_style: Dict[str, Any]


class ConsistencyMarkers(TypedDict, total=False):
    """Consistency markers for visual coherence across scenes."""
    visual_style: Dict[str, Any]
    character_continuity: Dict[str, Any]
    narrative_elements: Dict[str, Any]
    color_palette: List[str]
    cinematography_style: str
    lighting_signature: str


class AIPRODManifest(TypedDict, total=False):
    """Complete AIPROD production manifest."""
    production_id: str
    title: str
    total_duration_sec: float
    scenes: List[AIPRODScene]
    consistency_markers: ConsistencyMarkers
    metadata: Dict[str, Any]
    created_at: float
    version: str


class CostCertification(TypedDict, total=False):
    """Cost breakdown and certification."""
    base_cost: float
    quantization_factor: float
    gpu_cost_factor: float
    batch_efficiency: float
    orchestration_overhead: float
    total_estimated: float
    cost_per_minute: float
    selected_backend: str
    confidence: float


class ShotSpecification(TypedDict, total=False):
    """Detailed shot specification for rendering."""
    shot_id: str
    scene_id: str
    prompt: str
    negative_prompt: str
    duration_sec: float
    seed: int
    consistency_reference: Optional[str]
    technical_params: Dict[str, Any]
