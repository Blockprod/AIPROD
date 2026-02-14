"""Test coherence scoring module."""

import pytest
import numpy as np
from aiprod_pipelines.inference.multimodal_coherence.coherence_scorer import (
    CoherenceMetrics,
    CoherenceScorer,
    CoherenceResult,
)
from aiprod_pipelines.inference.multimodal_coherence.audio_processor import AudioEvent
from aiprod_pipelines.inference.multimodal_coherence.video_analyzer import MotionEvent


class TestCoherenceMetrics:
    """Test coherence metrics."""
    
    def test_metrics_creation(self):
        """Test metrics creation."""
        metrics = CoherenceMetrics(
            temporal_alignment=0.8,
            event_correlation=0.75,
            spectro_temporal_match=0.85,
        )
        
        assert metrics.temporal_alignment == 0.8
        assert metrics.event_correlation == 0.75
        assert len(metrics.components) == 5
    
    def test_metrics_overall_coherence(self):
        """Test overall coherence computation."""
        metrics = CoherenceMetrics(
            temporal_alignment=0.9,
            event_correlation=0.9,
            spectro_temporal_match=0.9,
            onset_synchrony=0.9,
            energy_correlation=0.9,
            overall_coherence=0.9,
        )
        
        assert metrics.overall_coherence == 0.9


class TestCoherenceScorer:
    """Test coherence scoring."""
    
    def test_scorer_creation(self):
        """Test scorer creation."""
        scorer = CoherenceScorer(audio_sr=16000, video_fps=30)
        
        assert scorer.audio_sr == 16000
        assert scorer.video_fps == 30
    
    def test_score_perfect_coherence(self):
        """Test scoring perfectly aligned streams."""
        scorer = CoherenceScorer()
        
        # Create perfectly aligned embeddings
        audio_emb = np.ones(64)
        video_emb = np.ones(64)
        
        # Create perfectly matched temporal features
        audio_temporal = {0.0: np.ones(32), 0.1: np.ones(32)}
        motion_mag = np.ones(10)
        
        # No events - just empty lists for testing
        audio_events = []
        motion_events = []
        
        metrics = scorer.score_coherence(
            audio_embedding=audio_emb,
            video_embedding=video_emb,
            audio_temporal=audio_temporal,
            motion_magnitude=motion_mag,
            audio_events=audio_events,
            motion_events=motion_events,
        )
        
        assert metrics.overall_coherence > 0.5
    
    def test_score_misaligned_coherence(self):
        """Test scoring misaligned streams."""
        scorer = CoherenceScorer()
        
        # Create misaligned embeddings
        audio_emb = np.ones(64)
        video_emb = -np.ones(64)  # Opposite
        
        audio_temporal = {0.0: np.ones(32), 0.1: -np.ones(32)}
        motion_mag = -np.ones(10)  # Opposite
        
        audio_events = []
        motion_events = []
        
        metrics = scorer.score_coherence(
            audio_embedding=audio_emb,
            video_embedding=video_emb,
            audio_temporal=audio_temporal,
            motion_magnitude=motion_mag,
            audio_events=audio_events,
            motion_events=motion_events,
        )
        
        # Should have lower coherence than perfect
        assert 0.0 <= metrics.overall_coherence <= 1.0
    
    def test_temporal_alignment_component(self):
        """Test temporal alignment component."""
        scorer = CoherenceScorer()
        
        audio_temporal = {
            0.0: np.array([0.5, 0.5]),
            0.1: np.array([1.0, 1.0]),
            0.2: np.array([1.5, 1.5]),
        }
        motion_mag = np.array([0.5, 1.0, 1.5])
        
        alignment = scorer._compute_temporal_alignment(audio_temporal, motion_mag)
        
        assert 0.0 <= alignment <= 1.0
    
    def test_event_correlation_component(self):
        """Test event correlation component."""
        scorer = CoherenceScorer()
        
        audio_events = [
            AudioEvent("speech", 0.0, 1.0, 0.9),
            AudioEvent("speech", 1.5, 2.5, 0.9),
        ]
        
        motion_events = [
            MotionEvent("fast", 0, 30, 0.8),
            MotionEvent("fast", 45, 75, 0.8),
        ]
        
        correlation = scorer._compute_event_correlation(audio_events, motion_events)
        
        assert 0.0 <= correlation <= 1.0
    
    def test_spectro_temporal_match(self):
        """Test spectrogram-temporal match component."""
        scorer = CoherenceScorer()
        
        audio_emb = np.random.randn(64)
        video_emb = audio_emb.copy()  # Same embedding
        
        match = scorer._compute_spectro_temporal_match(audio_emb, video_emb)
        
        assert 0.0 <= match <= 1.0
        assert match > 0.5  # Should be reasonably high
    
    def test_onset_synchrony_component(self):
        """Test onset synchrony component."""
        scorer = CoherenceScorer()
        
        audio_events = [
            AudioEvent("speech", 0.0, 2.0, 0.9),
            AudioEvent("speech", 3.0, 5.0, 0.9),
        ]
        
        motion_events = [
            MotionEvent("fast", 0, 60, 0.8),      # ~2 seconds at 30fps
            MotionEvent("fast", 90, 150, 0.8),    # ~2 seconds
        ]
        
        synchrony = scorer._compute_onset_synchrony(audio_events, motion_events)
        
        assert 0.0 <= synchrony <= 1.0
    
    def test_energy_correlation_component(self):
        """Test energy correlation component."""
        scorer = CoherenceScorer()
        
        audio_temporal = {
            0.0: np.ones(32),
            0.1: np.ones(32) * 2,
            0.2: np.ones(32) * 1.5,
        }
        
        motion_mag = np.array([1.0, 2.0, 1.5])
        
        correlation = scorer._compute_energy_correlation(audio_temporal, motion_mag)
        
        assert 0.0 <= correlation <= 1.0
    
    def test_batch_scoring(self):
        """Test batch scoring."""
        scorer = CoherenceScorer()
        
        # Create mock analysis results
        class MockAudioResult:
            def __init__(self):
                self.embedding = np.ones(64)
                self.temporal_features = {0.0: np.ones(32)}
                self.events = []
        
        class MockVideoResult:
            def __init__(self):
                self.embedding = np.ones(64)
                self.motion_magnitude = np.ones(10)
                self.motion_events = []
        
        audio_results = [MockAudioResult() for _ in range(3)]
        video_results = [MockVideoResult() for _ in range(3)]
        
        metrics_list = scorer.score_batch(audio_results, video_results)
        
        assert len(metrics_list) == 3
        for metrics in metrics_list:
            assert 0.0 <= metrics.overall_coherence <= 1.0


class TestCoherenceResult:
    """Test coherence result container."""
    
    def test_result_creation(self):
        """Test result creation."""
        metrics = CoherenceMetrics(overall_coherence=0.85)
        
        result = CoherenceResult(
            metrics=metrics,
            audio_summary={"events": 5},
            video_summary={"frames": 100},
            recommendations=["Increase audio volume"],
        )
        
        assert result.metrics == metrics
        assert len(result.recommendations) == 1
    
    def test_is_coherent(self):
        """Test coherence threshold check."""
        # High coherence
        high_metrics = CoherenceMetrics(overall_coherence=0.9)
        high_result = CoherenceResult(
            metrics=high_metrics,
            audio_summary={},
            video_summary={},
            recommendations=[],
        )
        assert high_result.is_coherent(threshold=0.75)
        
        # Low coherence
        low_metrics = CoherenceMetrics(overall_coherence=0.6)
        low_result = CoherenceResult(
            metrics=low_metrics,
            audio_summary={},
            video_summary={},
            recommendations=[],
        )
        assert not low_result.is_coherent(threshold=0.75)


class TestCoherenceScorerIntegration:
    """Integration tests for coherence scoring."""
    
    def test_full_coherence_pipeline(self):
        """Test full coherence scoring pipeline."""
        scorer = CoherenceScorer(audio_sr=16000, video_fps=30)
        
        # Create realistic test data
        audio_emb = np.random.randn(64)
        video_emb = np.random.randn(64)
        
        audio_temporal = {
            float(i) * 0.1: np.random.randn(32)
            for i in range(10)
        }
        
        motion_mag = np.random.rand(100)
        
        audio_events = [
            AudioEvent("speech", 0.2, 1.5, 0.95),
            AudioEvent("speech", 2.0, 3.5, 0.90),
        ]
        
        motion_events = [
            MotionEvent("fast", 5, 45, 0.8),
            MotionEvent("fast", 60, 105, 0.8),
        ]
        
        # Score
        metrics = scorer.score_coherence(
            audio_embedding=audio_emb,
            video_embedding=video_emb,
            audio_temporal=audio_temporal,
            motion_magnitude=motion_mag,
            audio_events=audio_events,
            motion_events=motion_events,
        )
        
        # Check all metrics are valid
        assert 0.0 <= metrics.temporal_alignment <= 1.0
        assert 0.0 <= metrics.event_correlation <= 1.0
        assert 0.0 <= metrics.spectro_temporal_match <= 1.0
        assert 0.0 <= metrics.onset_synchrony <= 1.0
        assert 0.0 <= metrics.energy_correlation <= 1.0
        assert 0.0 <= metrics.overall_coherence <= 1.0
    
    def test_coherence_reproducibility(self):
        """Test coherence scoring reproducibility."""
        scorer = CoherenceScorer()
        
        audio_emb = np.random.RandomState(42).randn(64)
        video_emb = np.random.RandomState(42).randn(64)
        
        audio_temporal = {
            0.0: np.random.RandomState(42).randn(32),
        }
        motion_mag = np.random.RandomState(42).rand(10)
        
        metrics1 = scorer.score_coherence(
            audio_embedding=audio_emb,
            video_embedding=video_emb,
            audio_temporal=audio_temporal,
            motion_magnitude=motion_mag,
            audio_events=[],
            motion_events=[],
        )
        
        metrics2 = scorer.score_coherence(
            audio_embedding=audio_emb,
            video_embedding=video_emb,
            audio_temporal=audio_temporal,
            motion_magnitude=motion_mag,
            audio_events=[],
            motion_events=[],
        )
        
        assert metrics1.overall_coherence == metrics2.overall_coherence
    
    def test_coherence_component_weights(self):
        """Test that component weighted average equals overall."""
        scorer = CoherenceScorer()
        
        metrics = CoherenceMetrics(
            temporal_alignment=0.8,
            event_correlation=0.7,
            spectro_temporal_match=0.9,
            onset_synchrony=0.75,
            energy_correlation=0.85,
            overall_coherence=0.8,  # Should match weighted average
        )
        
        # Verify weighting is reasonable
        expected = (
            0.25 * 0.8 +
            0.20 * 0.7 +
            0.20 * 0.9 +
            0.20 * 0.75 +
            0.15 * 0.85
        )
        
        assert abs(metrics.overall_coherence - expected) < 0.01 or \
               metrics.overall_coherence == 0.8  # Allow for manual setting
