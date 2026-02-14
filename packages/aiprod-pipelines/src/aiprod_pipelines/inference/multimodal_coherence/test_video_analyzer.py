"""Test video analysis module."""

import pytest
import numpy as np
from aiprod_pipelines.inference.multimodal_coherence.video_analyzer import (
    VideoFeature,
    MotionEvent,
    VideoAnalyzer,
    VideoAnalysisResult,
    BatchVideoAnalyzer,
)


class TestVideoFeature:
    """Test video feature container."""
    
    def test_feature_creation(self):
        """Test feature creation."""
        feature = VideoFeature(
            motion_magnitude=np.array([0.1, 0.2, 0.3]),
            content_embedding=np.random.randn(10, 64),
        )
        
        assert feature.motion_magnitude.shape == (3,)
        assert feature.content_embedding.shape == (10, 64)
    
    def test_feature_scene_cuts(self):
        """Test scene cuts in feature."""
        feature = VideoFeature(scene_cuts=[10, 30, 50])
        
        assert len(feature.scene_cuts) == 3
        assert 10 in feature.scene_cuts


class TestMotionEvent:
    """Test motion event."""
    
    def test_event_creation(self):
        """Test motion event creation."""
        event = MotionEvent(
            event_type="fast",
            start_frame=10,
            end_frame=50,
            magnitude=0.8,
            direction=(0.5, 0.2),
        )
        
        assert event.event_type == "fast"
        assert event.duration_frames == 40
        assert event.magnitude == 0.8
    
    def test_event_types(self):
        """Test various motion event types."""
        for event_type in ["static", "slow", "fast", "cut"]:
            event = MotionEvent(event_type, 0, 10, 0.5)
            assert event.event_type == event_type


class TestVideoAnalyzer:
    """Test video analysis."""
    
    def test_analyzer_creation(self):
        """Test analyzer creation."""
        analyzer = VideoAnalyzer(fps=30)
        
        assert analyzer.fps == 30
    
    def test_extract_features(self):
        """Test feature extraction."""
        analyzer = VideoAnalyzer()
        
        # Create synthetic frames
        T, H, W, C = 10, 64, 64, 3
        frames = np.random.randint(0, 256, (T, H, W, C), dtype=np.uint8)
        
        features = analyzer.extract_features(frames)
        
        assert features.content_embedding is not None
        assert features.content_embedding.shape[0] == T
    
    def test_scene_cut_detection(self):
        """Test scene cut detection."""
        analyzer = VideoAnalyzer()
        
        # Create frames with scene cut
        T, H, W, C = 20, 64, 64, 3
        frames = np.random.randint(100, 150, (T, H, W, C), dtype=np.uint8)
        
        # Insert abrupt change at frame 10
        frames[10:] = np.random.randint(0, 50, (T - 10, H, W, C), dtype=np.uint8)
        
        features = analyzer.extract_features(frames)
        
        assert len(features.scene_cuts) > 0
    
    def test_motion_detection(self):
        """Test motion event detection."""
        analyzer = VideoAnalyzer()
        
        # Create frames with motion
        T, H, W, C = 15, 64, 64, 3
        frames = np.zeros((T, H, W, C), dtype=np.uint8)
        
        # First part: static
        frames[0:5] = 100
        
        # Second part: moving
        for t in range(5, 10):
            frames[t] = 100 + t * 10
        
        # Third part: static again
        frames[10:15] = 200
        
        events = analyzer.detect_motion_events(frames)
        
        assert len(events) > 0
    
    def test_visual_embedding(self):
        """Test visual embedding."""
        analyzer = VideoAnalyzer()
        
        T, H, W, C = 8, 64, 64, 3
        frames = np.random.randint(0, 256, (T, H, W, C), dtype=np.uint8)
        
        embedding = analyzer.extract_visual_embedding(frames)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
    
    def test_motion_magnitude(self):
        """Test motion magnitude computation."""
        analyzer = VideoAnalyzer()
        
        T, H, W, C = 10, 64, 64, 3
        frames = np.random.randint(0, 256, (T, H, W, C), dtype=np.uint8)
        
        motion = analyzer.compute_motion_magnitude(frames)
        
        assert len(motion) == T - 1


class TestVideoAnalysisResult:
    """Test video analysis result."""
    
    def test_result_creation(self):
        """Test result creation."""
        features = VideoFeature()
        result = VideoAnalysisResult(
            features=features,
            motion_events=[],
            embedding=np.zeros(64),
            motion_magnitude=np.zeros(9),
        )
        
        assert result.features == features
        assert result.num_frames == 10


class TestBatchVideoAnalyzer:
    """Test batch video analysis."""
    
    def test_batch_process(self):
        """Test batch processing."""
        batch_analyzer = BatchVideoAnalyzer(fps=30)
        
        # Create batch of videos
        frames_list = [
            np.random.randint(0, 256, (10, 64, 64, 3), dtype=np.uint8)
            for _ in range(3)
        ]
        
        results = batch_analyzer.process_batch(frames_list)
        
        assert len(results) == 3
        for result in results:
            assert hasattr(result, 'embedding')
            assert hasattr(result, 'motion_events')


class TestVideoAnalyzerIntegration:
    """Integration tests for video analysis."""
    
    def test_full_analysis_pipeline(self):
        """Test full analysis pipeline."""
        analyzer = VideoAnalyzer(fps=30)
        
        # Create realistic video: 30 frames
        T, H, W, C = 30, 128, 128, 3
        frames = np.random.randint(0, 256, (T, H, W, C), dtype=np.uint8)
        
        # Extract all components
        features = analyzer.extract_features(frames)
        assert features is not None
        
        motion_events = analyzer.detect_motion_events(frames)
        assert isinstance(motion_events, list)
        
        embedding = analyzer.extract_visual_embedding(frames)
        assert embedding.shape[0] > 0
        
        motion = analyzer.compute_motion_magnitude(frames)
        assert len(motion) == T - 1
    
    def test_static_video(self):
        """Test analysis of static video."""
        analyzer = VideoAnalyzer()
        
        # Create completely static video
        T, H, W, C = 20, 64, 64, 3
        frames = np.ones((T, H, W, C), dtype=np.uint8) * 128
        
        events = analyzer.detect_motion_events(frames)
        motion = analyzer.compute_motion_magnitude(frames)
        
        # Should detect mostly static
        assert motion.mean() < 0.1
    
    def test_rapid_motion_video(self):
        """Test analysis of rapid motion."""
        analyzer = VideoAnalyzer()
        
        # Create video with rapid changes
        T, H, W, C = 20, 64, 64, 3
        frames = np.zeros((T, H, W, C), dtype=np.uint8)
        
        for t in range(T):
            frames[t] = ((t % 2) * 255)  # Flip between 0 and 255
        
        motion = analyzer.compute_motion_magnitude(frames)
        
        # Should detect high motion
        assert motion.mean() > 0.5
    
    def test_embedding_consistency(self):
        """Test embedding consistency."""
        analyzer = VideoAnalyzer()
        
        T, H, W, C = 10, 64, 64, 3
        rng = np.random.RandomState(42)
        frames = rng.randint(0, 256, (T, H, W, C), dtype=np.uint8)
        
        # Same frames should produce same embedding
        emb1 = analyzer.extract_visual_embedding(frames)
        emb2 = analyzer.extract_visual_embedding(frames)
        
        assert np.allclose(emb1, emb2)
    
    def test_scene_transition_detection(self):
        """Test detection of scene transitions."""
        analyzer = VideoAnalyzer()
        
        # Create video with scene transition
        T, H, W, C = 20, 64, 64, 3
        frames = np.zeros((T, H, W, C), dtype=np.uint8)
        
        # First scene
        frames[0:10] = 50
        
        # Scene cut
        frames[10:20] = 200
        
        features = analyzer.extract_features(frames)
        
        # Should detect scene cut around frame 10
        assert len(features.scene_cuts) > 0
        assert min(features.scene_cuts) >= 8  # Close to expected
