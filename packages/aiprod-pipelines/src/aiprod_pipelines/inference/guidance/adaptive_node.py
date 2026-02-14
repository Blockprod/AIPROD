"""
Adaptive guidance node for dynamic diffusion control.

Integrates PromptAnalyzer, TimestepScaler, and QualityPredictor into
a unified GraphNode that performs adaptive-guidance denoising.

Classes:
  - AdaptiveGuidanceNode: GraphNode with adaptive guidance
  - AdaptiveGuidanceProfile: Configuration for adaptive guidance
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from aiprod_pipelines.inference.graph import GraphNode, GraphContext
from aiprod_pipelines.inference.guidance.prompt_analyzer import PromptAnalyzerPredictor
from aiprod_pipelines.inference.guidance.quality_predictor import QualityAssessmentEngine
from aiprod_pipelines.inference.guidance.timestep_scaler import AdaptiveTimestepScaler


@dataclass
class AdaptiveGuidanceProfile:
    """Configuration for adaptive guidance."""
    
    enable_prompt_analysis: bool = True
    """Analyze prompt to determine base guidance."""
    
    enable_timestep_scaling: bool = True
    """Scale guidance based on noise level."""
    
    enable_quality_adjustment: bool = True
    """Adjust guidance based on generation quality."""
    
    enable_early_exit: bool = True
    """Exit early if quality converges."""
    
    min_steps: int = 15
    """Minimum denoising steps before early exit allowed."""
    
    prompt_analyzer_path: Optional[str] = None
    """Path to pretrained prompt analyzer model."""
    
    quality_predictor_path: Optional[str] = None
    """Path to pretrained quality predictor model."""


class AdaptiveGuidanceNode(GraphNode):
    """
    GraphNode with adaptive guidance control.
    
    Wraps the denoising process to:
    1. Analyze prompt for baseline guidance
    2. Scale guidance by timestep (high noise → strong guidance)
    3. Adjust guidance based on quality metrics
    4. Exit early if quality converges
    
    Inputs:
        latents: Initial noise or previous latents
        embeddings: Text embeddings from encoder
        prompt: Text prompt (for analysis)
        model: Denoising model
        scheduler: Noise scheduler
    
    Outputs:
        latents_denoised: Final denoised latents
        guidance_schedule: Per-step guidance values
        steps_used: Actual number of steps used (may be < num_steps)
        quality_trajectory: Quality metrics over denoising
        early_exit: Whether early exit was triggered
    """
    
    def __init__(
        self,
        model,
        scheduler,
        profile: Optional[AdaptiveGuidanceProfile] = None,
        node_id: str = "adaptive_denoise",
        **kwargs
    ):
        """
        Initialize AdaptiveGuidanceNode.
        
        Args:
            model: Denoising model
            scheduler: Noise scheduler
            profile: AdaptiveGuidanceProfile configuration
            node_id: Node identifier
            **kwargs: Additional configuration
        """
        super().__init__(node_id=node_id, **kwargs)
        
        self.model = model
        self.scheduler = scheduler
        self.profile = profile or AdaptiveGuidanceProfile()
        
        # Initialize components
        self.prompt_analyzer = None
        self.quality_engine = None
        self.timestep_scaler = AdaptiveTimestepScaler()
        
        # Load pretrained models if paths provided
        if self.profile.prompt_analyzer_path:
            self.prompt_analyzer = PromptAnalyzerPredictor()
            self.prompt_analyzer.load(self.profile.prompt_analyzer_path)
        
        if self.profile.quality_predictor_path:
            self.quality_engine = QualityAssessmentEngine()
            self.quality_engine.load(self.profile.quality_predictor_path)
        
        # Denoising configuration
        self.num_steps = self.config.get("num_inference_steps", 30)
        self.base_guidance_scale = self.config.get("guidance_scale", 7.5)
    
    @property
    def input_keys(self) -> List[str]:
        """Required inputs."""
        return ["latents", "embeddings", "prompt"]
    
    @property
    def output_keys(self) -> List[str]:
        """Produced outputs."""
        return [
            "latents_denoised",
            "guidance_schedule",
            "steps_used",
            "quality_trajectory",
            "early_exit",
        ]
    
    def execute(self, context: GraphContext) -> Dict[str, Any]:
        """
        Execute adaptive guidance denoising.
        
        Args:
            context: GraphContext with latents, embeddings, prompt
        
        Returns:
            Dict with denoised latents and metadata
        """
        self._validate_inputs(context, self.input_keys)
        
        latents = context["latents"]
        embeddings = context["embeddings"]
        prompt = context.get("prompt", "")
        num_steps = context.get("num_inference_steps", self.num_steps)
        
        # Analyze prompt for baseline guidance
        prompt_guidance = self._get_prompt_guidance(prompt, embeddings)
        
        # Initialize guidance schedule
        guidance_schedule = []
        quality_trajectory = []
        
        # Denoise loop with adaptive guidance
        steps_used = 0
        
        for step_idx, timestep in enumerate(self.scheduler.timesteps[:num_steps]):
            # Compute adaptive guidance scale
            guidance_scale = self._compute_adaptive_guidance(
                step_idx,
                timestep,
                prompt_guidance,
                latents,
                embeddings,
            )
            guidance_schedule.append(guidance_scale)
            
            # Denoise step
            noise_pred = self._denoise_step(latents, embeddings, timestep, guidance_scale)
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, timestep, latents)["prev_sample"]
            
            steps_used += 1
            
            # Check for early exit
            if self.profile.enable_early_exit and step_idx >= self.profile.min_steps:
                if self.quality_engine is not None:
                    adjustment, should_exit = self.quality_engine.step(latents, embeddings, timestep)
                    quality_trajectory.extend(self.quality_engine.trajectory.metrics)
                    
                    if should_exit:
                        break
        
        return {
            "latents_denoised": latents,
            "guidance_schedule": guidance_schedule,
            "steps_used": steps_used,
            "quality_trajectory": quality_trajectory,
            "early_exit": steps_used < num_steps,
        }
    
    def _get_prompt_guidance(self, prompt: str, embeddings: torch.Tensor) -> float:
        """
        Determine baseline guidance scale from prompt.
        
        Args:
            prompt: Text prompt
            embeddings: Text embeddings
        
        Returns:
            Baseline guidance scale [4, 10]
        """
        if not self.profile.enable_prompt_analysis or self.prompt_analyzer is None:
            return self.base_guidance_scale
        
        profile = self.prompt_analyzer.analyze(prompt, embeddings)
        return profile.base_guidance
    
    def _compute_adaptive_guidance(
        self,
        step_idx: int,
        timestep: int,
        base_guidance: float,
        latents: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> float:
        """
        Compute guidance scale for current step.
        
        Combines:
        1. Base guidance (from prompt analysis)
        2. Timestep scaling (high noise → strong, low noise → weak)
        3. Quality adjustment (improve quality if diverging)
        
        Args:
            step_idx: Index in denoising loop
            timestep: Current denoising timestep
            base_guidance: Baseline guidance scale
            latents: Current latent state
            embeddings: Text embeddings
        
        Returns:
            Effective guidance scale for this step
        """
        guidance = base_guidance
        
        # Apply timestep scaling
        if self.profile.enable_timestep_scaling:
            timestep_weight = self.timestep_scaler.get_weight(timestep)
            guidance *= timestep_weight
        
        # Apply quality-based adjustment
        if self.profile.enable_quality_adjustment and self.quality_engine is not None:
            adjustment, _ = self.quality_engine.predict_adjustment(
                self.quality_engine.assess_latents(latents, embeddings),
                timestep,
            )
            guidance *= (1.0 + adjustment)  # adjustment is [-0.5, 0.5]
        
        # Clamp guidance to reasonable range
        guidance = max(2.0, min(15.0, guidance))
        
        return guidance
    
    def _denoise_step(
        self,
        latents: torch.Tensor,
        embeddings: torch.Tensor,
        timestep: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        Single denoising step with classifier-free guidance.
        
        Args:
            latents: Current latents
            embeddings: Text embeddings [batch*2, seq_len, hidden] (positive + negative)
            timestep: Current timestep
            guidance_scale: Guidance weight
        
        Returns:
            Predicted noise
        """
        # Would call actual model here
        # For now, placeholder implementation
        noise_pred = torch.randn_like(latents)
        return noise_pred
