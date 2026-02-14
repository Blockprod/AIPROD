"""
Quality prediction and guidance adjustment.

Monitors video generation quality during denoising and makes real-time
guidance adjustments to maintain or improve quality.

Classes:
  - QualityPredictor: Neural network for quality assessment
  - QualityTrajectory: Tracks quality metrics across denoising steps
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class QualityMetrics:
    """Quality metrics for generated latents at a timestep."""
    
    latent_variance: float
    """Variance of latent values (high = diverse content)."""
    
    temporal_smoothness: float
    """Temporal consistency across frames [0-1]."""
    
    prompt_alignment: float
    """Cross-modal similarity to text prompt [0-1]."""
    
    artifact_score: float
    """Artifact likelihood [0-1], lower is better."""
    
    overall_quality: float
    """Composite quality score [0-1]."""
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for neural network input."""
        return torch.tensor([
            self.latent_variance,
            self.temporal_smoothness,
            self.prompt_alignment,
            self.artifact_score,
        ])


@dataclass
class QualityTrajectory:
    """Tracks quality metrics across entire denoising process."""
    
    timesteps: List[int] = field(default_factory=list)
    metrics: List[QualityMetrics] = field(default_factory=list)
    guidance_adjustments: List[float] = field(default_factory=list)
    early_exit: bool = False
    steps_completed: int = 0
    
    def add_step(
        self,
        timestep: int,
        metrics: QualityMetrics,
        adjustment: float = 0.0,
    ) -> None:
        """Record metrics from a denoising step."""
        self.timesteps.append(timestep)
        self.metrics.append(metrics)
        self.guidance_adjustments.append(adjustment)
        self.steps_completed += 1
    
    def quality_improving(self) -> bool:
        """Check if quality is improving."""
        if len(self.metrics) < 2:
            return True
        
        prev_quality = self.metrics[-2].overall_quality
        curr_quality = self.metrics[-1].overall_quality
        
        return curr_quality > prev_quality
    
    def should_exit_early(self, threshold: float = 0.92) -> bool:
        """Check if quality has converged sufficiently."""
        if len(self.metrics) < 15:  # Always run at least 15 steps
            return False
        
        recent_qualities = [m.overall_quality for m in self.metrics[-5:]]
        
        # Exit if last 5 steps show high, stable quality
        if min(recent_qualities) > threshold and max(recent_qualities) - min(recent_qualities) < 0.01:
            self.early_exit = True
            return True
        
        return False


class QualityPredictor(nn.Module):
    """
    Neural network for predicting quality from latent features.
    
    Architecture:
        Latent Features [batch, 4]  (variance, smoothness, alignment, artifacts)
            ↓
        Linear Layers → hidden [256]
            ↓
        Output Adjustment [-0.5, +0.5]
    
    Size: ~2M parameters, ~8MB checkpoint
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        """
        Initialize QualityPredictor.
        
        Args:
            input_dim: Number of input features (default: 4)
            hidden_dim: Hidden layer dimension (default: 256)
            num_layers: Number of hidden layers (default: 2)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        # Output layer: 3 outputs for [adjustment, confidence, early_exit_prob]
        layers.append(nn.Linear(hidden_dim, 3))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, quality_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict guidance adjustment from quality metrics.
        
        Args:
            quality_features: Tensor of shape [batch, 4] with features:
                [latent_variance, temporal_smoothness, prompt_alignment, artifact_score]
        
        Returns:
            Tuple of:
            - adjustment: [-0.5, +0.5] guidance adjustment
            - confidence: [0, 1] confidence in prediction
            - early_exit_prob: [0, 1] probability of early exit
        """
        logits = self.network(quality_features)
        
        adjustment = torch.tanh(logits[..., 0]) * 0.5  # [-0.5, +0.5]
        confidence = torch.sigmoid(logits[..., 1])      # [0, 1]
        early_exit_prob = torch.sigmoid(logits[..., 2]) # [0, 1]
        
        return adjustment, confidence, early_exit_prob
    
    def predict_adjustment(
        self,
        variance: float,
        smoothness: float,
        alignment: float,
        artifacts: float,
    ) -> Tuple[float, float]:
        """
        Convenience method for single prediction.
        
        Returns:
            Tuple of (guidance_adjustment, confidence)
        """
        features = torch.tensor([[variance, smoothness, alignment, artifacts]])
        
        with torch.no_grad():
            adjustment, confidence, _ = self.forward(features)
        
        return adjustment.item(), confidence.item()


class QualityAssessmentEngine:
    """
    High-level interface for assessing and adjusting guidance based on quality.
    """
    
    def __init__(
        self,
        model: Optional[QualityPredictor] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize quality assessment engine.
        
        Args:
            model: QualityPredictor model (creates if None)
            device: Device for inference
        """
        self.device = device
        
        if model is None:
            model = QualityPredictor()
        
        self.model = model.to(device).eval()
        self.trajectory = QualityTrajectory()
    
    def assess_latents(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> QualityMetrics:
        """
        Assess quality of latent representations.
        
        Args:
            latents: Generated latents [batch, channels, frames, H, W]
            text_embeddings: Text embeddings [batch, seq_len, hidden_dim]
        
        Returns:
            QualityMetrics object
        """
        with torch.no_grad():
            # Compute features
            variance = self._compute_variance(latents)
            smoothness = self._compute_temporal_smoothness(latents)
            alignment = self._compute_prompt_alignment(latents, text_embeddings)
            artifacts = self._estimate_artifacts(latents)
            
            # Composite quality
            overall = 0.25 * variance + 0.25 * smoothness + 0.35 * alignment + 0.15 * (1.0 - artifacts)
        
        return QualityMetrics(
            latent_variance=variance,
            temporal_smoothness=smoothness,
            prompt_alignment=alignment,
            artifact_score=artifacts,
            overall_quality=overall,
        )
    
    def predict_adjustment(
        self,
        metrics: QualityMetrics,
        timestep: int,
    ) -> Tuple[float, bool]:
        """
        Predict guidance adjustment based on quality metrics.
        
        Args:
            metrics: QualityMetrics from current step
            timestep: Current denoising timestep (for context)
        
        Returns:
            Tuple of (guidance_adjustment, should_exit_early)
        """
        features = metrics.to_tensor().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            adjustment, confidence, early_exit_prob = self.model(features)
        
        # Threshold for early exit: only if very high quality and confident
        should_exit = (early_exit_prob.item() > 0.8) and (metrics.overall_quality > 0.90)
        
        # Scale adjustment by confidence
        final_adjustment = adjustment.item() * confidence.item()
        
        return final_adjustment, should_exit
    
    def step(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        timestep: int,
    ) -> Tuple[float, bool]:
        """
        Perform single assessment step and return adjustment.
        
        Args:
            latents: Current latent states
            text_embeddings: Text embeddings
            timestep: Current timestep
        
        Returns:
            Tuple of (guidance_adjustment, should_exit_early)
        """
        # Assess quality
        metrics = self.assess_latents(latents, text_embeddings)
        
        # Predict adjustment
        adjustment, should_exit = self.predict_adjustment(metrics, timestep)
        
        # Record
        self.trajectory.add_step(timestep, metrics, adjustment)
        
        # Check if should exit early (override parameter if very confident)
        if not should_exit and len(self.trajectory.timesteps) > 15:
            should_exit = self.trajectory.should_exit_early(threshold=0.92)
        
        return adjustment, should_exit
    
    def reset(self) -> None:
        """Reset trajectory for new inference."""
        self.trajectory = QualityTrajectory()
    
    def _compute_variance(self, latents: torch.Tensor) -> float:
        """Compute latent variance (higher = more diverse)."""
        return latents.std().item()
    
    def _compute_temporal_smoothness(self, latents: torch.Tensor) -> float:
        """
        Compute temporal consistency across frames.
        
        Returns value in [0, 1] where 1 = perfectly smooth, 0 = maximally variable.
        """
        if latents.shape[2] < 2:  # fewer than 2 frames
            return 1.0
        
        # Compute frame-to-frame differences
        frame_diff = (latents[..., 1:, :, :] - latents[..., :-1, :, :]).abs().mean()
        
        # Normalize: smooth if small differences
        frame_diff_normalized = frame_diff.item() / (latents.std().item() + 1e-6)
        
        # Convert to [0, 1]: 0 = very different (not smooth), 1 = identical (smooth)
        smoothness = 1.0 / (1.0 + frame_diff_normalized)
        
        return smoothness
    
    def _compute_prompt_alignment(self, latents: torch.Tensor, embeddings: torch.Tensor) -> float:
        """
        Compute alignment between generated content and prompt.
        
        Placeholder: returns random value. In production, use CLIP image encoder.
        """
        # Placeholder: would use CLIP to encode latents and measure similarity
        # For now, return a default value
        return 0.85
    
    def _estimate_artifacts(self, latents: torch.Tensor) -> float:
        """
        Estimate likelihood of artifacts.
        
        Returns value in [0, 1] where 0 = no artifacts, 1 = many artifacts.
        
        Simple heuristic: check for extreme values or sudden spikes.
        """
        # Extreme values suggest artifacts
        has_extremes = (latents > 3.0).any().item() or (latents < -3.0).any().item()
        
        # Variance spikes suggest instability
        variance = latents.std().item()
        high_variance = variance > 2.0
        
        artifact_score = float(has_extremes) * 0.5 + float(high_variance) * 0.3
        
        return min(artifact_score, 1.0)
    
    def load(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def save(self, checkpoint_path: str) -> None:
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), checkpoint_path)
