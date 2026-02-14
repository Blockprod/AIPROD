"""
Adaptive guidance module for dynamic diffusion control.

Provides components for analyzing prompts, scaling guidance by timestep,
and adjusting guidance based on generation quality metrics.

Components:
  - PromptAnalyzer: Analyzes prompts to determine guidance needs
  - TimestepScaler: Maps denoising timesteps to guidance multipliers
  - QualityPredictor: Monitors quality and adjusts guidance dynamically
  - AdaptiveGuidanceNode: Unified GraphNode for adaptive guided denoising

Classes and Types:
  - GuidanceProfile: Prompt-analysis output
  - QualityMetrics: Per-step quality measurements
  - QualityTrajectory: Quality tracking across denoising
  - AdaptiveGuidanceProfile: Node configuration

Examples:

    # Basic usage with UnifiedInferenceGraph
    from aiprod_pipelines.inference import UnifiedInferenceGraph, preset
    
    # Use adaptive preset
    graph = preset("t2v_one_stage_adaptive")
    
    # Run inference
    result = graph.execute({
        "prompt": "A cat dancing",
        "negative_prompt": "",
    })
    
    # Access adaptive outputs
    guidance_schedule = result["guidance_schedule"]
    steps_used = result["steps_used"]
    print(f"Used {steps_used} steps, schedule: {guidance_schedule[:5]}...")
    
    # Advanced: Create node with custom configuration
    from aiprod_pipelines.inference.guidance import (
        AdaptiveGuidanceNode,
        AdaptiveGuidanceProfile,
    )
    
    profile = AdaptiveGuidanceProfile(
        enable_prompt_analysis=True,
        enable_timestep_scaling=True,
        enable_quality_adjustment=True,
        enable_early_exit=True,
        min_steps=15,
        prompt_analyzer_path="models/prompt_analyzer.pt",
        quality_predictor_path="models/quality_predictor.pt",
    )
    
    node = AdaptiveGuidanceNode(model, scheduler, profile)
    output = node.execute(context)
"""

from aiprod_pipelines.inference.guidance.adaptive_node import (
    AdaptiveGuidanceNode,
    AdaptiveGuidanceProfile,
)
from aiprod_pipelines.inference.guidance.prompt_analyzer import (
    GuidanceProfile,
    PromptAnalyzer,
    PromptAnalyzerPredictor,
)
from aiprod_pipelines.inference.guidance.quality_predictor import (
    QualityAssessmentEngine,
    QualityMetrics,
    QualityPredictor,
    QualityTrajectory,
)
from aiprod_pipelines.inference.guidance.timestep_scaler import (
    AdaptiveTimestepScaler,
    TimestepScaler,
)

__all__ = [
    # Adaptive node
    "AdaptiveGuidanceNode",
    "AdaptiveGuidanceProfile",
    # Prompt analysis
    "GuidanceProfile",
    "PromptAnalyzer",
    "PromptAnalyzerPredictor",
    # Quality prediction
    "QualityMetrics",
    "QualityTrajectory",
    "QualityPredictor",
    "QualityAssessmentEngine",
    # Timestep scaling
    "TimestepScaler",
    "AdaptiveTimestepScaler",
]
