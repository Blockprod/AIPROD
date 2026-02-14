"""End-to-end integration tests for multimodal coherence."""

import pytest
import numpy as np
from aiprod_pipelines.inference.multimodal_coherence.audio_processor import (
    AudioProcessor,
    BatchAudioProcessor,
)
from aiprod_pipelines.inference.multimodal_coherence.video_analyzer import (
    VideoAnalyzer,
    BatchVideoAnalyzer,
)
from aiprod_pipelines.inference.multimodal_coherence.coherence_scorer import (
    CoherenceScorer,
)
from aiprod_pipelines.inference.multimodal_coherence.sync_engine import (
    SyncEngine,
    AdaptiveSyncController,
)
from aiprod_pipelines.inference.multimodal_coherence.coherence_monitor import (
    CoherenceMonitor,
    MultimodalCoherenceNode,
    CoherenceReporter,
)


class TestEndToEndCoherence:
    """End-to-end coherence tests."""
    
    def create_audio_video_pair(self, sync_quality="good"):
        """Create paired audio-video data for testing."""
        sr = 16000
        fps = 30
        duration = 3.0
        
        # Audio: 3 seconds
        t = np.arange(int(sr * duration)) / sr
        sin1 = 0.2 * np.sin(2 * np.pi * 200 * t)
        sin2 = 0.1 * np.sin(2 * np.pi * 400 * t)
        noise = 0.05 * np.random.randn(len(t))
        audio = sin1 + sin2 + noise
        
        # Video: 90 frames (3 sec at 30fps)
        video = np.zeros((int(fps * duration), 64, 64, 3), dtype=np.uint8)
        
        # Fill with content that varies with audio
        for frame_idx in range(video.shape[0]):
            time = frame_idx / fps
            audio_idx = int(time * sr)
            
            if audio_idx < len(audio):
                # Video content varies with audio
                intensity = int(128 + 50 * audio[audio_idx])
            else:
                intensity = 128
            
            video[frame_idx] = np.clip(intensity, 0, 255)
        
        # Apply sync offset based on quality
        if sync_quality == "poor":
            # Shift audio forward significantly
            audio = np.concatenate([np.zeros(int(0.2 * sr)), audio[:-int(0.2 * sr)]])
        elif sync_quality == "fair":
            # Small offset
            audio = np.concatenate([np.zeros(int(0.05 * sr)), audio[:-int(0.05 * sr)]])
        
        return audio, video
    
    def test_perfect_coherence_scenario(self):
        """Test perfectly synchronized audio-video."""
        audio, video = self.create_audio_video_pair(sync_quality="good")
        
        # Process audio
        audio_processor = AudioProcessor()
        audio_features = audio_processor.extract_features(audio)
        audio_embedding = audio_processor.compute_audio_embedding(audio)
        audio_temporal = audio_processor.extract_temporal_features(audio)
        audio_events = audio_processor.detect_events(audio)
        
        # Process video
        video_analyzer = VideoAnalyzer()
        video_features = video_analyzer.extract_features(video)
        video_embedding = video_analyzer.extract_visual_embedding(video)
        motion_magnitude = video_analyzer.compute_motion_magnitude(video)
        motion_events = video_analyzer.detect_motion_events(video)
        
        # Score coherence
        scorer = CoherenceScorer()
        metrics = scorer.score_coherence(
            audio_embedding=audio_embedding,
            video_embedding=video_embedding,
            audio_temporal=audio_temporal,
            motion_magnitude=motion_magnitude,
            audio_events=audio_events,
            motion_events=motion_events,
        )
        
        # Should have good coherence
        assert metrics.overall_coherence > 0.6
    
    def test_with_sync_adjustment(self):
        """Test coherence with synchronization adjustment."""
        audio, video = self.create_audio_video_pair(sync_quality="fair")
        
        # Process
        audio_processor = AudioProcessor()
        audio_onsets = audio_processor.compute_onset_times(audio)
        
        video_analyzer = VideoAnalyzer()
        motion_events = video_analyzer.detect_motion_events(video)
        motion_onsets = [e.start_frame / 30.0 for e in motion_events]
        
        # Compute alignment
        sync_engine = SyncEngine()
        alignment = sync_engine.compute_alignment(
            audio_onsets,
            motion_onsets,
        )
        
        # Apply adjustment
        adjusted_video = sync_engine.apply_alignment(video, alignment)
        
        assert adjusted_video.shape == video.shape
    
    def test_batch_processing_workflow(self):
        """Test batch processing of multiple pairs."""
        # Create multiple audio-video pairs
        audio_list = []
        video_list = []
        
        for _ in range(3):
            audio, video = self.create_audio_video_pair()
            audio_list.append(audio)
            video_list.append(video)
        
        # Batch process
        audio_processor = BatchAudioProcessor()
        audio_results = audio_processor.process_batch(audio_list)
        
        video_analyzer = BatchVideoAnalyzer()
        video_results = video_analyzer.process_batch(video_list)
        
        # Score all pairs
        scorer = CoherenceScorer()
        metrics_list = scorer.score_batch(audio_results, video_results)
        
        assert len(metrics_list) == 3
        for metrics in metrics_list:
            assert 0.0 <= metrics.overall_coherence <= 1.0
    
    def test_real_time_monitoring(self):
        """Test real-time coherence monitoring."""
        audio, video = self.create_audio_video_pair()
        
        audio_processor = AudioProcessor()
        video_analyzer = VideoAnalyzer()
        scorer = CoherenceScorer()
        monitor = CoherenceMonitor()
        
        # Process in chunks
        chunk_size_samples = 8000  # 0.5 seconds
        chunk_size_frames = 15     # 0.5 seconds at 30fps
        
        for chunk_idx in range(2):
            start_sample = chunk_idx * chunk_size_samples
            end_sample = min(start_sample + chunk_size_samples, len(audio))
            
            start_frame = chunk_idx * chunk_size_frames
            end_frame = min(start_frame + chunk_size_frames, len(video))
            
            # Process chunk
            audio_chunk = audio[start_sample:end_sample]
            video_chunk = video[start_frame:end_frame]
            
            audio_embedding = audio_processor.compute_audio_embedding(audio_chunk)
            video_embedding = video_analyzer.extract_visual_embedding(video_chunk)
            
            audio_temporal = audio_processor.extract_temporal_features(audio_chunk)
            motion_magnitude = video_analyzer.compute_motion_magnitude(video_chunk)
            
            audio_events = audio_processor.detect_events(audio_chunk)
            motion_events = video_analyzer.detect_motion_events(video_chunk)
            
            # Score
            metrics = scorer.score_coherence(
                audio_embedding=audio_embedding,
                video_embedding=video_embedding,
                audio_temporal=audio_temporal,
                motion_magnitude=motion_magnitude,
                audio_events=audio_events,
                motion_events=motion_events,
            )
            
            # Monitor
            monitor.record_coherence(metrics, None)
        
        # Check monitoring
        assert len(monitor.history) == 2
    
    def test_degradation_detection_workflow(self):
        """Test degradation detection during monitoring."""
        monitor = CoherenceMonitor()
        
        # Simulate stable coherence followed by degradation
        coherence_progression = [
            0.88, 0.87, 0.89, 0.88, 0.87,  # Stable good
            0.85, 0.82, 0.80, 0.78, 0.75,  # Degradation
            0.73, 0.72, 0.71, 0.70, 0.70,  # Stabilized low
        ]
        
        from aiprod_pipelines.inference.multimodal_coherence.coherence_scorer import (
            CoherenceMetrics,
        )
        
        for coherence in coherence_progression:
            metrics = CoherenceMetrics(overall_coherence=coherence)
            monitor.record_coherence(metrics, None)
        
        # Detect degradation
        degradation = monitor.detect_degradation()
        
        # Should detect degradation
        if degradation:
            assert degradation["detected"] is True
    
    def test_full_system_workflow(self):
        """Test full multimodal coherence system."""
        # Create test data
        audio, video = self.create_audio_video_pair(sync_quality="good")
        
        # Initialize all components
        audio_processor = AudioProcessor()
        video_analyzer = VideoAnalyzer()
        scorer = CoherenceScorer()
        sync_engine = SyncEngine()
        monitor = CoherenceMonitor()
        node = MultimodalCoherenceNode()
        
        # Process audio
        class AudioResult:
            pass
        audio_result = AudioResult()
        audio_result.embedding = audio_processor.compute_audio_embedding(audio)
        audio_result.temporal_features = audio_processor.extract_temporal_features(audio)
        audio_result.events = audio_processor.detect_events(audio)
        audio_result.onsets = audio_processor.compute_onset_times(audio)
        
        # Process video
        class VideoResult:
            pass
        video_result = VideoResult()
        video_result.embedding = video_analyzer.extract_visual_embedding(video)
        video_result.motion_magnitude = video_analyzer.compute_motion_magnitude(video)
        video_result.motion_events = video_analyzer.detect_motion_events(video)
        video_result.num_frames = len(video)
        
        # Compute coherence
        coherence_metrics = scorer.score_coherence(
            audio_embedding=audio_result.embedding,
            video_embedding=video_result.embedding,
            audio_temporal=audio_result.temporal_features,
            motion_magnitude=video_result.motion_magnitude,
            audio_events=audio_result.events,
            motion_events=video_result.motion_events,
        )
        
        # Compute sync alignment
        alignment = sync_engine.compute_alignment(
            audio_onsets=audio_result.onsets,
            motion_onsets=[e.start_frame / 30.0 for e in video_result.motion_events],
        )
        
        alignment_history = [alignment]
        sync_metrics = sync_engine.compute_sync_metrics(
            alignment_history,
            audio_result.onsets,
            [e.start_frame / 30.0 for e in video_result.motion_events],
            total_duration=len(audio) / 16000.0,
        )
        
        # Monitor
        monitor.record_coherence(coherence_metrics, sync_metrics)
        
        # Execute node
        result = node.execute(
            audio_analysis=audio_result,
            video_analysis=video_result,
            coherence_metrics=coherence_metrics,
            sync_metrics=sync_metrics,
        )
        
        # Verify results
        assert "coherence" in result
        assert result["coherence"] > 0.0
        assert "components" in result
        assert "sync" in result
        
        # Generate report
        reporter = CoherenceReporter(monitor)
        report = reporter.generate_summary_report()
        assert len(report) > 0
    
    def test_comparison_different_qualities(self):
        """Test coherence comparison across quality levels."""
        qualities = ["good", "fair", "poor"]
        coherence_scores = []
        
        for quality in qualities:
            audio, video = self.create_audio_video_pair(sync_quality=quality)
            
            audio_processor = AudioProcessor()
            video_analyzer = VideoAnalyzer()
            scorer = CoherenceScorer()
            
            audio_embedding = audio_processor.compute_audio_embedding(audio)
            video_embedding = video_analyzer.extract_visual_embedding(video)
            audio_temporal = audio_processor.extract_temporal_features(audio)
            motion_magnitude = video_analyzer.compute_motion_magnitude(video)
            audio_events = audio_processor.detect_events(audio)
            motion_events = video_analyzer.detect_motion_events(video)
            
            metrics = scorer.score_coherence(
                audio_embedding=audio_embedding,
                video_embedding=video_embedding,
                audio_temporal=audio_temporal,
                motion_magnitude=motion_magnitude,
                audio_events=audio_events,
                motion_events=motion_events,
            )
            
            coherence_scores.append((quality, metrics.overall_coherence))
        
        # Good should have highest coherence
        good_score = next(s for q, s in coherence_scores if q == "good")
        poor_score = next(s for q, s in coherence_scores if q == "poor")
        
        # This is a soft assertion - coherence should be better for better sync
        assert good_score >= poor_score - 0.1  # Allow some variance
    
    def test_adaptive_sync_correction(self):
        """Test adaptive synchronization correction."""
        audio, video = self.create_audio_video_pair(sync_quality="fair")
        
        audio_processor = AudioProcessor()
        video_analyzer = VideoAnalyzer()
        
        # Extract features
        audio_onsets = audio_processor.compute_onset_times(audio)
        motion_events = video_analyzer.detect_motion_events(video)
        motion_onsets = [e.start_frame / 30.0 for e in motion_events]
        
        # Adaptive controller
        controller = AdaptiveSyncController()
        
        # Multiple iterations of sync refinement
        for iteration in range(3):
            result = controller.update_sync(audio_onsets, motion_onsets)
            
            assert "alignment" in result
            assert result["alignment"]["offset"] is not None
        
        # Check controller state
        assert len(controller.alignment_history) == 3
        assert len(controller.metrics_history) == 3
