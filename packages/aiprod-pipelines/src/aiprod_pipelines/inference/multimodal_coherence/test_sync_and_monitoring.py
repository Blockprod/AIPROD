"""Test sync engine and monitoring."""

import pytest
import numpy as np
from datetime import datetime
from aiprod_pipelines.inference.multimodal_coherence.sync_engine import (
    SyncAlignment,
    SyncMetrics,
    SyncEngine,
    AdaptiveSyncController,
)
from aiprod_pipelines.inference.multimodal_coherence.coherence_monitor import (
    CoherenceSnapshot,
    CoherenceMonitor,
    MultimodalCoherenceNode,
    CoherenceReporter,
)
from aiprod_pipelines.inference.multimodal_coherence.coherence_scorer import CoherenceMetrics


class TestSyncAlignment:
    """Test sync alignment."""
    
    def test_alignment_creation(self):
        """Test alignment creation."""
        align = SyncAlignment(
            time_offset=0.05,
            confidence=0.95,
            adjustment_type="interpolate",
        )
        
        assert align.time_offset == 0.05
        assert align.confidence == 0.95
        assert align.adjustment_type == "interpolate"
    
    def test_alignment_latency_conversion(self):
        """Test latency conversion."""
        align = SyncAlignment(time_offset=0.1)
        
        assert align.latency_ms == 100.0


class TestSyncMetrics:
    """Test sync metrics."""
    
    def test_metrics_creation(self):
        """Test metrics creation."""
        metrics = SyncMetrics(
            lip_sync_score=0.92,
            audio_lead=0.02,
            sync_stability=0.95,
        )
        
        assert metrics.lip_sync_score == 0.92
        assert metrics.audio_lead == 0.02


class TestSyncEngine:
    """Test synchronization engine."""
    
    def test_engine_creation(self):
        """Test engine creation."""
        engine = SyncEngine(audio_sr=16000, video_fps=30)
        
        assert engine.audio_sr == 16000
        assert engine.video_fps == 30
    
    def test_compute_alignment_no_events(self):
        """Test alignment with no events."""
        engine = SyncEngine()
        
        align = engine.compute_alignment([], [])
        
        assert align.time_offset == 0.0
    
    def test_compute_alignment_perfect(self):
        """Test alignment with perfectly matched onsets."""
        engine = SyncEngine()
        
        audio_onsets = [0.0, 1.0, 2.0, 3.0]
        motion_onsets = [0.0, 1.0, 2.0, 3.0]
        
        align = engine.compute_alignment(audio_onsets, motion_onsets)
        
        assert abs(align.time_offset) < 0.01
        assert align.adjustment_type == "none"
    
    def test_compute_alignment_offset(self):
        """Test alignment with consistent offset."""
        engine = SyncEngine()
        
        audio_onsets = [0.05, 1.05, 2.05, 3.05]
        motion_onsets = [0.0, 1.0, 2.0, 3.0]
        
        align = engine.compute_alignment(audio_onsets, motion_onsets)
        
        # Should detect ~50ms audio lead
        assert 0.04 < align.time_offset < 0.06
    
    def test_apply_alignment_interpolate(self):
        """Test frame interpolation for alignment."""
        engine = SyncEngine()
        
        # Create test frames
        T, H, W, C = 10, 64, 64, 3
        frames = np.random.randint(0, 256, (T, H, W, C), dtype=np.uint8)
        
        align = SyncAlignment(
            time_offset=0.02,
            adjustment_type="interpolate",
        )
        
        adjusted = engine.apply_alignment(frames, align)
        
        assert adjusted.shape == frames.shape
    
    def test_compute_sync_metrics(self):
        """Test sync metrics computation."""
        engine = SyncEngine()
        
        align_history = [
            SyncAlignment(time_offset=0.01),
            SyncAlignment(time_offset=0.02),
            SyncAlignment(time_offset=0.015),
        ]
        
        metrics = engine.compute_sync_metrics(
            align_history,
            audio_onsets=[0.0, 1.0, 2.0],
            motion_onsets=[0.0, 1.0, 2.0],
            total_duration=3.0,
        )
        
        assert 0.0 <= metrics.lip_sync_score <= 1.0
        assert 0.0 <= metrics.sync_stability <= 1.0
    
    def test_detect_sync_issues(self):
        """Test issue detection."""
        engine = SyncEngine()
        
        # Large offset
        align = SyncAlignment(time_offset=0.15)
        metrics = SyncMetrics(
            lip_sync_score=0.6,
            sync_stability=0.5,
        )
        
        issues = engine.detect_sync_issues(align, metrics)
        
        assert len(issues) > 0


class TestAdaptiveSyncController:
    """Test adaptive sync controller."""
    
    def test_controller_creation(self):
        """Test controller creation."""
        controller = AdaptiveSyncController()
        
        assert controller.engine is not None
        assert len(controller.alignment_history) == 0
    
    def test_update_sync(self):
        """Test sync update."""
        controller = AdaptiveSyncController()
        
        result = controller.update_sync(
            audio_onsets=[0.0, 1.0, 2.0],
            motion_onsets=[0.0, 1.0, 2.0],
        )
        
        assert "alignment" in result
        assert "metrics" in result
        assert "issues" in result
        assert len(controller.alignment_history) == 1
    
    def test_get_adjustment_parameters(self):
        """Test getting adjustment parameters."""
        controller = AdaptiveSyncController()
        
        controller.update_sync([], [])
        
        params = controller.get_adjustment_parameters()
        
        assert "offset" in params
        assert "confidence" in params


class TestCoherenceSnapshot:
    """Test coherence snapshot."""
    
    def test_snapshot_creation(self):
        """Test snapshot creation."""
        snapshot = CoherenceSnapshot(
            timestamp=datetime.now(),
            overall_coherence=0.85,
            components={"alignment": 0.8},
            issues=[],
        )
        
        assert snapshot.overall_coherence == 0.85
        assert isinstance(snapshot.components, dict)


class TestCoherenceMonitor:
    """Test coherence monitoring."""
    
    def test_monitor_creation(self):
        """Test monitor creation."""
        monitor = CoherenceMonitor(window_size=50)
        
        assert monitor.window_size == 50
        assert len(monitor.history) == 0
    
    def test_record_coherence(self):
        """Test recording coherence."""
        monitor = CoherenceMonitor()
        
        metrics = CoherenceMetrics(
            temporal_alignment=0.8,
            event_correlation=0.75,
            spectro_temporal_match=0.85,
            onset_synchrony=0.8,
            energy_correlation=0.82,
            overall_coherence=0.8,
        )
        
        monitor.record_coherence(metrics, None, issues=[])
        
        assert len(monitor.history) == 1
    
    def test_get_statistics(self):
        """Test statistics computation."""
        monitor = CoherenceMonitor()
        
        for coherence in [0.8, 0.85, 0.90, 0.88]:
            metrics = CoherenceMetrics(overall_coherence=coherence)
            monitor.record_coherence(metrics, None)
        
        stats = monitor.get_statistics()
        
        assert "mean_coherence" in stats
        assert stats["measurements"] == 4
    
    def test_detect_degradation(self):
        """Test degradation detection."""
        monitor = CoherenceMonitor(degradation_threshold=0.1)
        
        # Record degrading coherence
        for i in range(5):
            metrics = CoherenceMetrics(overall_coherence=0.9)
            monitor.record_coherence(metrics, None)
        
        for i in range(5):
            metrics = CoherenceMetrics(overall_coherence=0.7)
            monitor.record_coherence(metrics, None)
        
        degradation = monitor.detect_degradation()
        
        assert degradation is not None
        assert degradation["detected"] is True
    
    def test_get_issue_summary(self):
        """Test issue summary."""
        monitor = CoherenceMonitor()
        
        for _ in range(3):
            metrics = CoherenceMetrics(overall_coherence=0.8)
            monitor.record_coherence(
                metrics,
                None,
                issues=["Low bandwidth"],
            )
        
        summary = monitor.get_issue_summary()
        
        assert "Low bandwidth" in summary
        assert summary["Low bandwidth"] == 3
    
    def test_get_health_report(self):
        """Test health report."""
        monitor = CoherenceMonitor()
        
        for _ in range(10):
            metrics = CoherenceMetrics(overall_coherence=0.85)
            monitor.record_coherence(metrics, None)
        
        report = monitor.get_health_report()
        
        assert "health" in report
        assert "statistics" in report
        assert report["health"] in ["excellent", "good", "fair", "poor", "unknown"]


class TestMultimodalCoherenceNode:
    """Test coherence graph node."""
    
    def test_node_creation(self):
        """Test node creation."""
        node = MultimodalCoherenceNode()
        
        assert node.monitor is not None
    
    def test_node_execute(self):
        """Test node execution."""
        node = MultimodalCoherenceNode()
        
        class MockResult:
            def __init__(self):
                self.events = []
                self.motion_events = []
                self.num_frames = 100
        
        metrics = CoherenceMetrics(overall_coherence=0.85)
        sync_metrics = SyncMetrics(lip_sync_score=0.9)
        
        result = node.execute(
            audio_analysis=MockResult(),
            video_analysis=MockResult(),
            coherence_metrics=metrics,
            sync_metrics=sync_metrics,
        )
        
        assert "coherence" in result
        assert "components" in result
        assert "sync" in result


class TestCoherenceReporter:
    """Test coherence reporter."""
    
    def test_reporter_creation(self):
        """Test reporter creation."""
        monitor = CoherenceMonitor()
        reporter = CoherenceReporter(monitor)
        
        assert reporter.monitor == monitor
    
    def test_generate_summary_report(self):
        """Test report generation."""
        monitor = CoherenceMonitor()
        
        for _ in range(5):
            metrics = CoherenceMetrics(overall_coherence=0.85)
            monitor.record_coherence(metrics, None)
        
        reporter = CoherenceReporter(monitor)
        report = reporter.generate_summary_report()
        
        assert isinstance(report, str)
        assert "COHERENCE REPORT" in report
        assert "Health Status" in report


class TestMonitoringIntegration:
    """Integration tests for monitoring."""
    
    def test_full_monitoring_pipeline(self):
        """Test full monitoring pipeline."""
        monitor = CoherenceMonitor()
        
        # Simulate coherence measurements over time
        coherence_values = [
            0.85, 0.86, 0.85, 0.84, 0.83,  # Stable good
            0.82, 0.80, 0.78, 0.75, 0.72,  # Degradation
            0.70, 0.70, 0.71, 0.72, 0.73,  # Stabilization
        ]
        
        for coherence in coherence_values:
            metrics = CoherenceMetrics(overall_coherence=coherence)
            monitor.record_coherence(
                metrics,
                None,
                issues=["Low quality"] if coherence < 0.75 else [],
            )
        
        # Check statistics
        stats = monitor.get_statistics()
        assert stats["measurements"] == len(coherence_values)
        
        # Check degradation detection
        degradation = monitor.detect_degradation()
        assert degradation is not None
        
        # Check health report
        health = monitor.get_health_report()
        assert health["health"] in ["excellent", "good", "fair", "poor"]
        
        # Generate report
        reporter = CoherenceReporter(monitor)
        report = reporter.generate_summary_report()
        assert len(report) > 0
    
    def test_monitoring_with_sync_info(self):
        """Test monitoring with synchronization info."""
        monitor = CoherenceMonitor()
        
        for i in range(5):
            coherence = 0.85 - i * 0.05
            
            metrics = CoherenceMetrics(overall_coherence=coherence)
            sync_metrics = SyncMetrics(
                lip_sync_score=0.9,
                audio_lead=i * 0.01,
                sync_stability=0.95,
            )
            
            monitor.record_coherence(
                coherence_metrics=metrics,
                sync_metrics=sync_metrics,
                metadata={"iteration": i},
            )
        
        stats = monitor.get_statistics()
        assert "measurements" in stats
