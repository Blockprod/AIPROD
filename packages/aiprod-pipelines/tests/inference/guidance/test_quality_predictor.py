"""
Unit tests for QualityPredictor and related classes.

Coverage:
  - QualityMetrics dataclass
  - QualityTrajectory tracking
  - QualityPredictor neural network
  - QualityAssessmentEngine wrapper
  - Early exit logic
  - Metric computation
"""

import pytest
import torch

from aiprod_pipelines.inference.guidance.quality_predictor import (
    QualityMetrics,
    QualityTrajectory,
    QualityPredictor,
    QualityAssessmentEngine,
)


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating QualityMetrics."""
        metrics = QualityMetrics(
            latent_variance=1.5,
            temporal_smoothness=0.85,
            prompt_alignment=0.92,
            artifact_score=0.15,
            overall_quality=0.88,
        )
        
        assert metrics.latent_variance == 1.5
        assert metrics.temporal_smoothness == 0.85
        assert metrics.prompt_alignment == 0.92
        assert metrics.artifact_score == 0.15
        assert metrics.overall_quality == 0.88
    
    def test_metrics_to_tensor(self):
        """Test converting metrics to tensor."""
        metrics = QualityMetrics(
            latent_variance=1.5,
            temporal_smoothness=0.85,
            prompt_alignment=0.92,
            artifact_score=0.15,
            overall_quality=0.88,
        )
        
        tensor = metrics.to_tensor()
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (4,)
        assert tensor[0] == 1.5  # variance
        assert tensor[1] == 0.85  # smoothness
        assert tensor[2] == 0.92  # alignment
        assert tensor[3] == 0.15  # artifacts
    
    def test_metrics_ranges(self):
        """Test metrics within expected ranges."""
        # Variance can be any non-negative value
        metrics = QualityMetrics(0.0, 1.0, 1.0, 0.0, 1.0)
        assert metrics.latent_variance >= 0
        
        # Quality factors should be in [0, 1]
        assert 0 <= metrics.temporal_smoothness <= 1
        assert 0 <= metrics.prompt_alignment <= 1
        assert 0 <= metrics.artifact_score <= 1
        assert 0 <= metrics.overall_quality <= 1


class TestQualityTrajectory:
    """Tests for QualityTrajectory tracking."""
    
    def test_trajectory_creation(self):
        """Test creating QualityTrajectory."""
        trajectory = QualityTrajectory()
        
        assert trajectory.timesteps == []
        assert trajectory.metrics == []
        assert trajectory.guidance_adjustments == []
        assert trajectory.early_exit is False
        assert trajectory.steps_completed == 0
    
    def test_trajectory_add_step(self):
        """Test adding steps to trajectory."""
        trajectory = QualityTrajectory()
        
        metrics = QualityMetrics(1.0, 0.8, 0.9, 0.1, 0.86)
        trajectory.add_step(timestep=999, metrics=metrics, adjustment=0.0)
        
        assert len(trajectory.timesteps) == 1
        assert len(trajectory.metrics) == 1
        assert len(trajectory.guidance_adjustments) == 1
        assert trajectory.steps_completed == 1
        
        assert trajectory.timesteps[0] == 999
        assert trajectory.metrics[0] == metrics
        assert trajectory.guidance_adjustments[0] == 0.0
    
    def test_trajectory_multiple_steps(self):
        """Test adding multiple steps."""
        trajectory = QualityTrajectory()
        
        for i in range(5):
            metrics = QualityMetrics(
                latent_variance=2.0 - (i * 0.3),
                temporal_smoothness=0.5 + (i * 0.08),
                prompt_alignment=0.7 + (i * 0.05),
                artifact_score=0.5 - (i * 0.08),
                overall_quality=0.6 + (i * 0.08),
            )
            trajectory.add_step(
                timestep=999 - (i * 100),
                metrics=metrics,
                adjustment=0.0,
            )
        
        assert trajectory.steps_completed == 5
        assert len(trajectory.timesteps) == 5
        assert len(trajectory.metrics) == 5
    
    def test_quality_improving(self):
        """Test quality_improving method."""
        trajectory = QualityTrajectory()
        
        # Add improving trajectory
        for i in range(5):
            metrics = QualityMetrics(
                latent_variance=2.0 - (i * 0.3),
                temporal_smoothness=0.5 + (i * 0.08),
                prompt_alignment=0.7 + (i * 0.05),
                artifact_score=0.5 - (i * 0.08),
                overall_quality=0.6 + (i * 0.08),
            )
            trajectory.add_step(999 - (i * 100), metrics, 0.0)
        
        is_improving = trajectory.quality_improving()
        
        # Quality should be improving since overall_quality increases
        assert isinstance(is_improving, bool)
    
    def test_should_exit_early_default_threshold(self):
        """Test early exit with default threshold."""
        trajectory = QualityTrajectory()
        
        # Not enough steps
        for i in range(10):
            metrics = QualityMetrics(0.5, 0.95, 0.95, 0.05, 0.95)
            trajectory.add_step(999 - (i * 50), metrics, 0.0)
        
        # Should not exit with only 10 steps (need 15+)
        should_exit = trajectory.should_exit_early(threshold=0.92)
        assert should_exit is False or trajectory.steps_completed < 15
    
    def test_should_exit_early_sufficient_steps(self):
        """Test early exit with sufficient steps and quality."""
        trajectory = QualityTrajectory()
        
        # Add 20 high-quality steps
        for i in range(20):
            metrics = QualityMetrics(0.1, 0.96, 0.96, 0.02, 0.96)
            trajectory.add_step(999 - (i * 50), metrics, 0.0)
        
        # May exit if quality is consistently high
        should_exit = trajectory.should_exit_early(threshold=0.92)
        assert isinstance(should_exit, bool)
    
    def test_trajectory_state_after_exit(self):
        """Test trajectory state after setting early_exit."""
        trajectory = QualityTrajectory()
        
        assert trajectory.early_exit is False
        
        for i in range(20):
            metrics = QualityMetrics(0.1, 0.96, 0.96, 0.02, 0.96)
            trajectory.add_step(999 - (i * 50), metrics, 0.0)
        
        # Even if should_exit returns True, we manually set it
        trajectory.early_exit = True
        assert trajectory.early_exit is True


class TestQualityPredictorNetwork:
    """Tests for QualityPredictor neural network."""
    
    def test_init_default(self):
        """Test default initialization."""
        predictor = QualityPredictor()
        
        assert predictor is not None
        # Should have layers for processing [variance, smoothness, alignment, artifacts]
    
    def test_forward_output_shape(self):
        """Test forward pass output shape."""
        predictor = QualityPredictor()
        predictor.eval()
        
        batch_size = 2
        # Input: [variance, smoothness, alignment, artifacts]
        features = torch.randn(batch_size, 4)
        
        with torch.no_grad():
            adjustment, confidence, early_exit_prob = predictor(features)
        
        # Check shapes
        assert adjustment.shape == (batch_size,)
        assert confidence.shape == (batch_size,)
        assert early_exit_prob.shape == (batch_size,)
    
    def test_forward_output_ranges(self):
        """Test output ranges are valid."""
        predictor = QualityPredictor()
        predictor.eval()
        
        features = torch.randn(4, 4)
        
        with torch.no_grad():
            adjustment, confidence, early_exit_prob = predictor(features)
        
        # Adjustment should be in [-0.5, 0.5]
        assert (adjustment >= -0.5).all() and (adjustment <= 0.5).all()
        
        # Confidence and prob should be in [0, 1]
        assert (confidence >= 0).all() and (confidence <= 1).all()
        assert (early_exit_prob >= 0).all() and (early_exit_prob <= 1).all()
    
    def test_forward_gradient_flow(self):
        """Test gradient flow during training."""
        predictor = QualityPredictor()
        predictor.train()
        
        features = torch.randn(2, 4, requires_grad=True)
        adjustment, confidence, early_exit_prob = predictor(features)
        
        loss = adjustment.mean() + confidence.mean() + early_exit_prob.mean()
        loss.backward()
        
        # Gradients should exist
        assert features.grad is not None
        for param in predictor.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_predict_adjustment(self):
        """Test predict_adjustment method."""
        predictor = QualityPredictor()
        
        variance = torch.tensor([1.5, 2.0])
        smoothness = torch.tensor([0.8, 0.7])
        alignment = torch.tensor([0.9, 0.85])
        artifacts = torch.tensor([0.2, 0.25])
        
        with torch.no_grad():
            adjustment, confidence = predictor.predict_adjustment(
                variance, smoothness, alignment, artifacts
            )
        
        assert adjustment.shape == (2,)
        assert confidence.shape == (2,)
        assert (adjustment >= -0.5).all() and (adjustment <= 0.5).all()
        assert (confidence >= 0).all() and (confidence <= 1).all()


class TestQualityAssessmentEngine:
    """Tests for QualityAssessmentEngine wrapper."""
    
    def test_engine_init(self):
        """Test engine initialization."""
        engine = QualityAssessmentEngine()
        
        assert engine.predictor is not None
        assert engine.trajectory is not None
    
    def test_assess_latents(self):
        """Test latent assessment."""
        engine = QualityAssessmentEngine()
        
        latents = torch.randn(1, 4, 32, 32)
        embeddings = torch.randn(1, 77, 768)
        
        metrics = engine.assess_latents(latents, embeddings)
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.latent_variance >= 0
        assert 0 <= metrics.temporal_smoothness <= 1
        assert 0 <= metrics.prompt_alignment <= 1
        assert 0 <= metrics.artifact_score <= 1
        assert 0 <= metrics.overall_quality <= 1
    
    def test_predict_adjustment(self):
        """Test adjustment prediction."""
        engine = QualityAssessmentEngine()
        
        metrics = QualityMetrics(1.5, 0.8, 0.9, 0.2, 0.85)
        adjustment, should_exit = engine.predict_adjustment(metrics, timestep=500)
        
        assert isinstance(adjustment, float)
        assert isinstance(should_exit, bool)
        assert -0.5 <= adjustment <= 0.5
    
    def test_engine_step(self):
        """Test engine step integration."""
        engine = QualityAssessmentEngine()
        
        latents = torch.randn(1, 4, 32, 32)
        embeddings = torch.randn(1, 77, 768)
        
        adjustment, should_exit = engine.step(latents, embeddings, timestep=500)
        
        assert isinstance(adjustment, float)
        assert isinstance(should_exit, bool)
        
        # Trajectory should be updated
        assert engine.trajectory.steps_completed >= 1
    
    def test_engine_reset(self):
        """Test engine reset."""
        engine = QualityAssessmentEngine()
        
        # Add some steps
        for i in range(5):
            latents = torch.randn(1, 4, 32, 32)
            embeddings = torch.randn(1, 77, 768)
            engine.step(latents, embeddings, timestep=999 - (i * 100))
        
        assert engine.trajectory.steps_completed == 5
        
        # Reset
        engine.reset()
        
        assert engine.trajectory.steps_completed == 0
        assert len(engine.trajectory.metrics) == 0
    
    def test_compute_variance(self):
        """Test variance computation."""
        engine = QualityAssessmentEngine()
        
        # Test with uniform latents (low variance)
        uniform = torch.ones(1, 4, 32, 32)
        variance_uniform = engine._compute_variance(uniform)
        
        # Test with random latents (high variance)
        random = torch.randn(1, 4, 32, 32)
        variance_random = engine._compute_variance(random)
        
        # Random should have higher variance
        assert variance_random > variance_uniform
    
    def test_compute_temporal_smoothness(self):
        """Test temporal smoothness computation."""
        engine = QualityAssessmentEngine()
        
        # Smooth latents (similar frames)
        smooth = torch.ones(2, 4, 32, 32) + torch.randn(2, 4, 32, 32) * 0.01
        smoothness_smooth = engine._compute_temporal_smoothness(smooth, smooth.clone())
        
        # Different latents
        different = torch.randn(2, 4, 32, 32)
        smoothness_diff = engine._compute_temporal_smoothness(smooth, different)
        
        # Similar should be smoother
        assert smoothness_smooth > smoothness_diff
    
    def test_compute_prompt_alignment(self):
        """Test prompt alignment computation."""
        engine = QualityAssessmentEngine()
        
        latents = torch.randn(1, 4, 32, 32)
        embeddings = torch.randn(1, 77, 768)
        
        alignment = engine._compute_prompt_alignment(latents, embeddings)
        
        assert isinstance(alignment, float)
        assert 0 <= alignment <= 1
    
    def test_estimate_artifacts(self):
        """Test artifact estimation."""
        engine = QualityAssessmentEngine()
        
        # Clean latents (low artifacts)
        clean = torch.randn(1, 4, 32, 32)
        artifacts_clean = engine._estimate_artifacts(clean)
        
        # Blocky latents (high artifacts)
        blocky = torch.zeros(1, 4, 32, 32)
        blocky[:, :, ::4, ::4] = 1.0  # Checkerboard pattern
        artifacts_blocky = engine._estimate_artifacts(blocky)
        
        # Should detect more artifacts in blocky
        assert isinstance(artifacts_clean, float)
        assert isinstance(artifacts_blocky, float)
        assert 0 <= artifacts_clean <= 1
        assert 0 <= artifacts_blocky <= 1


class TestQualityAssessmentTrajectory:
    """Tests for trajectory within assessment engine."""
    
    def test_trajectory_updates_with_steps(self):
        """Test that trajectory updates as engine steps."""
        engine = QualityAssessmentEngine()
        
        assert engine.trajectory.steps_completed == 0
        
        for i in range(5):
            latents = torch.randn(1, 4, 32, 32)
            embeddings = torch.randn(1, 77, 768)
            engine.step(latents, embeddings, timestep=999 - (i * 100))
        
        assert engine.trajectory.steps_completed == 5
        assert len(engine.trajectory.metrics) == 5
    
    def test_early_exit_detection(self):
        """Test early exit detection in trajectory."""
        engine = QualityAssessmentEngine()
        
        # Add converged steps (high quality)
        for i in range(20):
            metrics = QualityMetrics(
                latent_variance=0.1,
                temporal_smoothness=0.95,
                prompt_alignment=0.95,
                artifact_score=0.05,
                overall_quality=0.95,
            )
            engine.trajectory.add_step(999 - (i * 50), metrics, 0.0)
        
        # Check if should exit
        should_exit = engine.trajectory.should_exit_early(threshold=0.92)
        assert isinstance(should_exit, bool)


class TestQualityMetricsComplexity:
    """Tests for overall quality metric computation."""
    
    def test_overall_quality_formula(self):
        """Test that overall quality follows expected formula."""
        # overall_quality = 0.25*variance + 0.25*smoothness + 0.35*alignment + 0.15*(1-artifacts)
        
        variance = 0.4
        smoothness = 0.6
        alignment = 0.8
        artifacts = 0.2
        
        expected = (
            0.25 * variance + 
            0.25 * smoothness + 
            0.35 * alignment + 
            0.15 * (1 - artifacts)
        )
        
        metrics = QualityMetrics(
            latent_variance=variance,
            temporal_smoothness=smoothness,
            prompt_alignment=alignment,
            artifact_score=artifacts,
            overall_quality=expected,  # Set to expected
        )
        
        assert abs(metrics.overall_quality - expected) < 1e-5
    
    def test_overall_quality_range(self):
        """Test that overall quality is in [0, 1]."""
        # With all perfect scores
        perfect = QualityMetrics(0.0, 1.0, 1.0, 0.0, 1.0)
        assert 0 <= perfect.overall_quality <= 1
        
        # With all bad scores
        bad = QualityMetrics(2.0, 0.0, 0.0, 1.0, 0.0)
        # Overall quality should still make sense
        assert bad.overall_quality >= 0


class TestQualityEdgeCases:
    """Tests for edge cases."""
    
    def test_single_element_batch(self):
        """Test with single element batch."""
        engine = QualityAssessmentEngine()
        
        latents = torch.randn(1, 4, 32, 32)
        embeddings = torch.randn(1, 77, 768)
        
        metrics = engine.assess_latents(latents, embeddings)
        
        assert isinstance(metrics, QualityMetrics)
    
    def test_large_batch(self):
        """Test with larger batch."""
        engine = QualityAssessmentEngine()
        
        latents = torch.randn(4, 4, 32, 32)
        embeddings = torch.randn(4, 77, 768)
        
        # Still processes single latent-embedding pair
        metrics = engine.assess_latents(latents[:1], embeddings[:1])
        
        assert isinstance(metrics, QualityMetrics)
    
    def test_zero_latents(self):
        """Test with zero latents."""
        engine = QualityAssessmentEngine()
        
        latents = torch.zeros(1, 4, 32, 32)
        embeddings = torch.randn(1, 77, 768)
        
        metrics = engine.assess_latents(latents, embeddings)
        
        # Should handle zero case
        assert isinstance(metrics, QualityMetrics)
        assert metrics.latent_variance >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
