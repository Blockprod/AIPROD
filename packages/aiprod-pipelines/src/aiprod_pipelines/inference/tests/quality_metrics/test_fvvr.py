"""Test suite for FVVR (Frame Visual Relevance Ratio) metric.

Tests:
- CLIP embedding generation
- FVVR score computation
- Batch processing
- Tracking and statistics
- Early exit detection
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from aiprod_pipelines.inference.quality_metrics.fvvr import (
    FVVRMetric,
    FVVRCalculator,
    FVVRTracker,
    compute_fvvr_efficient,
)


class TestFVVRMetric:
    """Test FVVRMetric dataclass."""
    
    def test_fvvr_metric_creation(self):
        """Test creating FVVRMetric."""
        metric = FVVRMetric(
            overall_score=0.85,
            per_frame_scores=torch.tensor([0.80, 0.85, 0.90]),
            quality_grade="good",
        )
        
        assert metric.overall_score == 0.85
        assert metric.quality_grade == "good"
        assert metric.per_frame_scores.shape == (3,)
    
    def test_fvvr_metric_grading(self):
        """Test automatic grade assignment."""
        test_cases = [
            (0.90, "excellent"),
            (0.80, "good"),
            (0.70, "fair"),
            (0.50, "poor"),
        ]
        
        for score, expected_grade in test_cases:
            metric = FVVRMetric(
                overall_score=score,
                per_frame_scores=torch.tensor([score]),
                quality_grade=expected_grade,
            )
            assert metric.quality_grade == expected_grade


class TestFVVRCalculator:
    """Test FVVRCalculator computation."""
    
    @pytest.fixture
    def calculator(self):
        """Create calculator with mocked CLIP models."""
        with patch('aiprod_pipelines.inference.quality_metrics.fvvr.CLIPTextEmbedder'):
            with patch('aiprod_pipelines.inference.quality_metrics.fvvr.CLIPImageEmbedder'):
                calc = FVVRCalculator()
                return calc
    
    def test_calculator_initialization(self, calculator):
        """Test calculator setup."""
        assert calculator is not None
    
    def test_fvvr_score_computation_mock(self, calculator):
        """Test FVVR computation with mocked embeddings."""
        # Mock embeddings
        calculator.text_embedder = Mock()
        calculator.image_embedder = Mock()
        
        prompt_embedding = torch.randn(1, 768)
        frame_embeddings = torch.randn(4, 768)
        
        calculator.text_embedder.encode = Mock(return_value=prompt_embedding)
        calculator.image_embedder.encode = Mock(return_value=frame_embeddings)
        
        # Compute FVVR
        fvvr = calculator.compute_fvvr(
            "a cat jumping",
            torch.randn(4, 3, 512, 512),
        )
        
        # FVVR should be in [0, 1]
        assert 0 <= fvvr <= 1
    
    def test_batch_processing(self, calculator):
        """Test batch FVVR computation."""
        calculator.text_embedder = Mock()
        calculator.image_embedder = Mock()
        
        prompt_embeddings = torch.randn(2, 768)
        frame_embeddings = torch.randn(4, 768)
        
        calculator.text_embedder.encode = Mock(return_value=prompt_embeddings)
        calculator.image_embedder.encode = Mock(return_value=frame_embeddings)
        
        # Batch of 2 videos, 4 frames each
        videos = torch.randn(2, 4, 3, 512, 512)
        prompts = ["cat jumping", "dog running"]
        
        fvvr_scores = calculator.compute_fvvr_batch(prompts, videos)
        
        assert len(fvvr_scores) == 2
        assert all(0 <= score <= 1 for score in fvvr_scores)
    
    def test_efficient_computation(self):
        """Test efficient FVVR using pre-computed embeddings."""
        # Pre-computed embeddings
        prompt_emb = torch.randn(1, 768)
        frame_embeds = torch.randn(4, 768)
        
        # Normalize
        prompt_norm = prompt_emb / (prompt_emb.norm(dim=1, keepdim=True) + 1e-8)
        frames_norm = frame_embeds / (frame_embeds.norm(dim=1, keepdim=True) + 1e-8)
        
        # Compute similarity
        similarity = torch.matmul(frames_norm, prompt_norm.t()).squeeze(1)
        
        # Average (FVVR)
        fvvr = similarity.mean().item()
        
        assert 0 <= fvvr <= 1


class TestFVVRTracker:
    """Test FVVRTracker for monitoring trends."""
    
    def test_tracker_initialization(self):
        """Test tracker setup."""
        tracker = FVVRTracker()
        
        assert tracker.scores == []
        assert tracker.threshold == 0.85
    
    def test_add_score(self):
        """Test adding scores."""
        tracker = FVVRTracker()
        
        tracker.add_score(0.80)
        tracker.add_score(0.85)
        tracker.add_score(0.90)
        
        assert len(tracker.scores) == 3
        assert tracker.scores[-1] == 0.90
    
    def test_trend_detection(self):
        """Test trend detection."""
        tracker = FVVRTracker()
        
        # Add improving scores
        for score in [0.60, 0.65, 0.70, 0.75, 0.80]:
            tracker.add_score(score)
        
        trend = tracker.get_trend()
        assert trend == "improving"
        
        # Add stable scores
        tracker2 = FVVRTracker()
        for score in [0.80, 0.81, 0.80, 0.81, 0.79]:
            tracker2.add_score(score)
        
        trend2 = tracker2.get_trend()
        assert trend2 == "stable"
    
    def test_early_exit_detection(self):
        """Test early exit point detection."""
        tracker = FVVRTracker(threshold=0.85)
        
        # Below threshold
        for score in [0.70, 0.75, 0.80]:
            tracker.add_score(score)
        
        assert not tracker.should_exit()
        
        # Above threshold
        tracker2 = FVVRTracker(threshold=0.85)
        for score in [0.88, 0.87, 0.86]:
            tracker2.add_score(score)
        
        assert tracker2.should_exit()
    
    def test_statistics(self):
        """Test statistics computation."""
        tracker = FVVRTracker()
        
        scores = [0.70, 0.75, 0.80, 0.85, 0.90]
        for score in scores:
            tracker.add_score(score)
        
        stats = tracker.get_statistics()
        
        assert stats["mean"] == pytest.approx(0.80)
        assert stats["min"] == 0.70
        assert stats["max"] == 0.90


class TestIntegration:
    """Integration tests for FVVR system."""
    
    def test_full_workflow(self):
        """Test complete FVVR workflow."""
        # Create dummy data
        prompt = "a beautiful landscape"
        frames = torch.randn(4, 3, 512, 512)
        
        # Test metric creation
        metric = FVVRMetric(
            overall_score=0.82,
            per_frame_scores=torch.tensor([0.80, 0.82, 0.84, 0.82]),
            quality_grade="good",
        )
        
        assert metric.overall_score == 0.82
        assert metric.per_frame_scores.shape == (4,)
        
        # Test tracker
        tracker = FVVRTracker()
        tracker.add_score(metric.overall_score)
        
        stats = tracker.get_statistics()
        assert stats["mean"] == metric.overall_score
    
    def test_multiple_prompts(self):
        """Test multiple prompts tracking."""
        tracker = FVVRTracker()
        
        prompts = [
            "a cat",
            "a dog",
            "a bird",
        ]
        
        scores = [0.75, 0.82, 0.88]
        
        for prompt, score in zip(prompts, scores):
            tracker.add_score(score)
        
        stats = tracker.get_statistics()
        
        assert stats["mean"] == pytest.approx(0.8167, rel=0.01)
        assert stats["max"] == 0.88
        assert stats["min"] == 0.75
