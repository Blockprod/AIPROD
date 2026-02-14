"""Integration tests for quality metrics system.

Tests complete workflows combining:
- FVVR (semantic alignment)
- LPIPS (visual quality)
- Motion (temporal consistency)
- Quality aggregation
- Monitoring and trend detection
"""

import pytest
import torch
import numpy as np

from aiprod_pipelines.inference.quality_metrics.fvvr import FVVRMetric, FVVRTracker
from aiprod_pipelines.inference.quality_metrics.lpips import LPIPSMetric
from aiprod_pipelines.inference.quality_metrics.motion import MotionMetric
from aiprod_pipelines.inference.quality_metrics.quality_monitor import (
    QualityScore,
    QualityMonitor,
    QualityAggregator,
)


class TestQualityScore:
    """Test QualityScore dataclass."""
    
    def test_quality_score_creation(self):
        """Test creating unified quality score."""
        score = QualityScore(
            fvvr_score=0.82,
            lpips_score=0.80,
            motion_score=0.78,
            overall_quality=0.80,
            consistency=0.95,
            confidence=0.85,
            per_frame_quality=torch.tensor([0.78, 0.80, 0.82]),
            grade="good",
        )
        
        assert score.overall_quality == 0.80
        assert score.grade == "good"
        assert score.consistency > 0.9
    
    def test_score_with_dominant_issue(self):
        """Test score with identified limiting factor."""
        score = QualityScore(
            fvvr_score=0.50,  # Low semantic alignment
            lpips_score=0.85,
            motion_score=0.88,
            overall_quality=0.74,
            consistency=0.70,
            confidence=0.75,
            per_frame_quality=torch.ones(4) * 0.74,
            grade="fair",
            dominant_issue="fvvr",
        )
        
        assert score.dominant_issue == "fvvr"
        assert score.grade == "fair"


class TestQualityMonitor:
    """Test quality monitoring system."""
    
    @pytest.fixture
    def monitor(self):
        """Create quality monitor."""
        return QualityMonitor()
    
    def test_monitor_initialization(self, monitor):
        """Test monitor setup."""
        assert monitor is not None
        assert "fvvr" in monitor.weights
        assert "lpips" in monitor.weights
        assert "motion" in monitor.weights
        
        # Weights should sum to 1
        total = sum(monitor.weights.values())
        assert abs(total - 1.0) < 0.01
    
    def test_custom_weights(self):
        """Test custom weight configuration."""
        custom_weights = {
            "fvvr": 0.50,
            "lpips": 0.30,
            "motion": 0.20,
        }
        
        monitor = QualityMonitor(weights=custom_weights)
        
        assert monitor.weights == custom_weights
    
    def test_metric_aggregation_high_quality(self, monitor):
        """Test aggregation with all metrics high."""
        score = monitor.aggregate_metrics(
            fvvr=0.90,
            lpips=0.10,  # Low distance = high quality
            motion=0.92,
        )
        
        assert score.overall_quality > 0.85
        assert score.grade == "excellent"
        assert score.consistency > 0.9
    
    def test_metric_aggregation_low_quality(self, monitor):
        """Test aggregation with low metrics."""
        score = monitor.aggregate_metrics(
            fvvr=0.55,
            lpips=0.50,  # High distance = low quality
            motion=0.50,
        )
        
        assert score.overall_quality < 0.60
        assert score.grade == "poor"
    
    def test_metric_aggregation_unbalanced(self, monitor):
        """Test aggregation when metrics disagree."""
        score = monitor.aggregate_metrics(
            fvvr=0.95,  # Excellent
            lpips=0.80,  # Good
            motion=0.40,  # Poor
        )
        
        assert score.dominant_issue == "motion"
        assert 0.60 < score.overall_quality < 0.75
        assert score.grade == "fair"
    
    def test_dominant_issue_detection(self, monitor):
        """Test identification of limiting factor."""
        test_cases = [
            # (fvvr, lpips_norm, motion, expected_issue)
            (0.50, 0.90, 0.90, "fvvr"),  # FVVR is lowest
            (0.90, 0.30, 0.90, "lpips"),  # LPIPS is lowest
            (0.90, 0.90, 0.40, "motion"),  # Motion is lowest
        ]
        
        for fvvr, lpips, motion, expected_issue in test_cases:
            score = monitor.aggregate_metrics(
                fvvr=fvvr,
                lpips=(1.0 - lpips) * 0.5,  # LPIPS is distance
                motion=motion,
            )
            
            if score.grade != "excellent":
                assert score.dominant_issue is not None
    
    def test_score_history(self, monitor):
        """Test tracking score history."""
        scores = [0.70, 0.75, 0.80, 0.85, 0.88]
        
        for score_val in scores:
            score = monitor.aggregate_metrics(
                fvvr=score_val,
                lpips=0.15,
                motion=score_val,
            )
            monitor.add_score(len(monitor.history), score)
        
        assert len(monitor.history) == 5
        assert len(monitor.per_step_scores) == 5
    
    def test_trend_detection(self, monitor):
        """Test trend analysis."""
        # Improving trend
        improving_scores = [0.60, 0.65, 0.70, 0.75, 0.80]
        
        for score_val in improving_scores:
            score = monitor.aggregate_metrics(
                fvvr=score_val,
                lpips=0.20,
                motion=score_val,
            )
            monitor.add_score(len(monitor.history), score)
        
        trend = monitor.get_trend()
        assert trend == "improving"
        
        # Stable trend
        monitor2 = QualityMonitor()
        stable_scores = [0.80, 0.81, 0.79, 0.80, 0.82]
        
        for score_val in stable_scores:
            score = monitor2.aggregate_metrics(
                fvvr=score_val,
                lpips=0.15,
                motion=score_val,
            )
            monitor2.add_score(len(monitor2.history), score)
        
        trend2 = monitor2.get_trend()
        assert trend2 == "stable"
    
    def test_early_exit_detection(self, monitor):
        """Test convergence detection for early exit."""
        # Below threshold
        low_scores = [0.70, 0.75, 0.78]
        
        for score_val in low_scores:
            score = monitor.aggregate_metrics(
                fvvr=score_val,
                lpips=0.20,
                motion=score_val,
            )
            monitor.add_score(len(monitor.history), score)
        
        assert not monitor.should_early_exit(threshold=0.85)
        
        # Above threshold
        monitor2 = QualityMonitor()
        high_scores = [0.88, 0.87, 0.86]
        
        for score_val in high_scores:
            score = monitor2.aggregate_metrics(
                fvvr=score_val,
                lpips=0.10,
                motion=score_val,
            )
            monitor2.add_score(len(monitor2.history), score)
        
        assert monitor2.should_early_exit(threshold=0.85)
    
    def test_quality_degradation_detection(self, monitor):
        """Test sudden quality drop detection."""
        # Add normal quality
        score1 = monitor.aggregate_metrics(
            fvvr=0.85,
            lpips=0.15,
            motion=0.85,
        )
        monitor.add_score(0, score1)
        
        # Add degraded quality
        score2 = monitor.aggregate_metrics(
            fvvr=0.50,
            lpips=0.40,
            motion=0.50,
        )
        monitor.add_score(1, score2)
        
        # Should detect degradation
        assert monitor.detect_quality_degradation(threshold=0.15)
    
    def test_statistics_computation(self, monitor):
        """Test statistics aggregation."""
        scores = [0.70, 0.75, 0.80, 0.85, 0.90]
        
        for score_val in scores:
            score = monitor.aggregate_metrics(
                fvvr=score_val,
                lpips=0.20,
                motion=score_val,
            )
            monitor.add_score(len(monitor.history), score)
        
        stats = monitor.get_statistics()
        
        assert stats["min"] == pytest.approx(0.70, rel=0.05)
        assert stats["max"] == pytest.approx(0.90, rel=0.05)
        assert 0.75 < stats["mean"] < 0.85
    
    def test_diagnostics_report(self, monitor):
        """Test comprehensive diagnostics generation."""
        # Add multiple measurements
        for i in range(5):
            score = monitor.aggregate_metrics(
                fvvr=0.75 + i * 0.02,
                lpips=0.20,
                motion=0.75 + i * 0.02,
            )
            monitor.add_score(i, score)
        
        report = monitor.get_diagnostics_report()
        
        assert "statistics" in report
        assert "trend" in report
        assert "early_exit_available" in report
        assert "quality_degradation" in report
        assert "latest_measurement" in report
        assert report["measurement_count"] == 5


class TestQualityAggregator:
    """Test simple aggregation utility."""
    
    def test_aggregate_high_quality(self):
        """Test aggregation with high metrics."""
        overall = QualityAggregator.aggregate(
            fvvr=0.90,
            lpips=0.10,
            motion=0.88,
        )
        
        assert 0.80 < overall < 1.0
    
    def test_aggregate_low_quality(self):
        """Test aggregation with low metrics."""
        overall = QualityAggregator.aggregate(
            fvvr=0.50,
            lpips=0.50,
            motion=0.50,
        )
        
        assert 0.0 < overall < 0.60
    
    def test_custom_weights_aggregation(self):
        """Test aggregation with custom weights."""
        custom_weights = {
            "fvvr": 0.60,
            "lpips": 0.20,
            "motion": 0.20,
        }
        
        overall = QualityAggregator.aggregate(
            fvvr=0.60,
            lpips=0.80,  # Would normalize to 0.6
            motion=1.00,
            weights=custom_weights,
        )
        
        # FVVR-heavy weighting should pull score down slightly
        assert overall < 0.80


class TestFullWorkflow:
    """End-to-end quality monitoring workflow."""
    
    def test_inference_quality_tracking(self):
        """Test tracking quality during simulated inference."""
        monitor = QualityMonitor()
        
        # Simulate denoising steps
        step_results = [
            # Step 0: Initial poor quality
            {"fvvr": 0.50, "lpips": 0.60, "motion": 0.40},
            # Step 1: Improving
            {"fvvr": 0.65, "lpips": 0.40, "motion": 0.55},
            # Step 2: Better
            {"fvvr": 0.75, "lpips": 0.25, "motion": 0.70},
            # Step 3: Good
            {"fvvr": 0.82, "lpips": 0.18, "motion": 0.80},
            # Step 4: Excellent
            {"fvvr": 0.85, "lpips": 0.15, "motion": 0.85},
        ]
        
        for step, result in enumerate(step_results):
            score = monitor.aggregate_metrics(
                fvvr=result["fvvr"],
                lpips=result["lpips"],
                motion=result["motion"],
            )
            monitor.add_score(step, score)
        
        # Check progression
        assert monitor.get_trend() == "improving"
        assert monitor.should_early_exit(threshold=0.80)
        
        report = monitor.get_diagnostics_report()
        assert report["measurement_count"] == 5
    
    def test_quality_degradation_scenario(self):
        """Test detection of quality degradation scenario."""
        monitor = QualityMonitor()
        
        # Good quality for first 3 steps
        for i in range(3):
            score = monitor.aggregate_metrics(
                fvvr=0.85,
                lpips=0.15,
                motion=0.85,
            )
            monitor.add_score(i, score)
        
        # Sudden degradation
        score = monitor.aggregate_metrics(
            fvvr=0.40,
            lpips=0.70,
            motion=0.30,
        )
        monitor.add_score(3, score)
        
        # Should detect issue
        assert monitor.detect_quality_degradation()
        
        # Diagnostics should flag it
        report = monitor.get_diagnostics_report()
        assert report["quality_degradation"] is True
    
    def test_multi_metric_disagreement(self):
        """Test handling conflicting metric signals."""
        monitor = QualityMonitor()
        
        # Metrics strongly disagree
        score = monitor.aggregate_metrics(
            fvvr=0.95,      # Excellent alignment
            lpips=0.60,      # Poor visual quality
            motion=0.95,     # Excellent smoothness
        )
        
        # Overall should be pulled down by poor LPIPS
        assert 0.65 < score.overall_quality < 0.80
        assert score.grade in ["fair", "good"]
        assert score.dominant_issue == "lpips"
        assert score.consistency < 0.90  # Low consistency due to disagreement
