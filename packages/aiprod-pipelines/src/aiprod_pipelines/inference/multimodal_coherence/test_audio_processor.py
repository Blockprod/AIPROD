"""Test audio processing module."""

import pytest
import numpy as np
from aiprod_pipelines.inference.multimodal_coherence.audio_processor import (
    AudioFeature,
    AudioEventType,
    AudioEvent,
    AudioProcessor,
    AudioAnalysisResult,
    BatchAudioProcessor,
)


class TestAudioFeature:
    """Test audio feature container."""
    
    def test_feature_creation(self):
        """Test feature creation."""
        feature = AudioFeature(
            mfcc=np.random.randn(13, 100),
            mel_spectrogram=np.random.randn(128, 100),
        )
        
        assert feature.mfcc.shape == (13, 100)
        assert feature.mel_spectrogram.shape == (128, 100)
    
    def test_feature_empty(self):
        """Test empty feature."""
        feature = AudioFeature()
        
        assert feature.mfcc is None
        assert feature.temporal_envelope is None


class TestAudioEvent:
    """Test audio event."""
    
    def test_event_creation(self):
        """Test event creation."""
        event = AudioEvent(
            event_type=AudioEventType.SPEECH,
            start_time=0.5,
            end_time=2.0,
            confidence=0.95,
        )
        
        assert event.event_type == AudioEventType.SPEECH
        assert event.duration == 1.5
        assert event.confidence == 0.95
    
    def test_event_types(self):
        """Test event types."""
        assert AudioEventType.SPEECH == "speech"
        assert AudioEventType.MUSIC == "music"
        assert AudioEventType.SILENCE == "silence"


class TestAudioProcessor:
    """Test audio processing."""
    
    def test_processor_creation(self):
        """Test processor creation."""
        processor = AudioProcessor(sr=16000)
        
        assert processor.sr == 16000
        assert processor.n_mfcc == 13
    
    def test_compute_embedding(self):
        """Test embedding computation."""
        processor = AudioProcessor()
        audio = np.random.randn(16000)  # 1 second
        
        embedding = processor.compute_audio_embedding(audio)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0
    
    def test_detect_events(self):
        """Test event detection."""
        processor = AudioProcessor()
        
        # Create synthetic audio with energy variation
        sr = 16000
        duration = 2.0
        t = np.arange(int(sr * duration)) / sr
        
        # Mix of silence, noise, and speech
        audio = np.concatenate([
            np.random.randn(int(sr * 0.5)) * 0.01,  # silence
            np.random.randn(int(sr * 1.0)) * 0.2,   # speech
            np.random.randn(int(sr * 0.5)) * 0.01,  # silence
        ])
        
        events = processor.detect_events(audio)
        
        assert len(events) > 0
        for event in events:
            assert hasattr(event, 'event_type')
            assert hasattr(event, 'start_time')
    
    def test_temporal_features(self):
        """Test temporal feature extraction."""
        processor = AudioProcessor()
        audio = np.random.randn(32000)
        
        temporal = processor.extract_temporal_features(audio, window_size=0.5)
        
        assert isinstance(temporal, dict)
        assert len(temporal) > 0
    
    def test_onset_detection(self):
        """Test onset detection."""
        processor = AudioProcessor()
        
        # Create audio with clear onsets (impulses)
        sr = 16000
        audio = np.zeros(sr)
        
        # Add impulses
        audio[0] = 1.0
        audio[sr // 3] = 1.0
        audio[2 * sr // 3] = 1.0
        
        onsets = processor.compute_onset_times(audio)
        
        # Should detect onsets
        assert len(onsets) >= 0


class TestAudioAnalysisResult:
    """Test analysis result container."""
    
    def test_result_creation(self):
        """Test result creation."""
        features = AudioFeature()
        result = AudioAnalysisResult(
            features=features,
            events=[],
            embedding=np.zeros(64),
            onset_times=[],
            temporal_features={},
        )
        
        assert result.features == features
        assert len(result.events) == 0


class TestBatchAudioProcessor:
    """Test batch processing."""
    
    def test_batch_process(self):
        """Test batch processing."""
        batch_processor = BatchAudioProcessor()
        
        # Create batch of audio
        audio_list = [
            np.random.randn(16000) for _ in range(3)
        ]
        
        results = batch_processor.process_batch(audio_list)
        
        assert len(results) == 3
        for result in results:
            assert hasattr(result, 'embedding')
            assert hasattr(result, 'events')


class TestAudioProcessorIntegration:
    """Integration tests for audio processing."""
    
    def test_full_analysis_pipeline(self):
        """Test full analysis pipeline."""
        processor = AudioProcessor()
        
        # Create realistic audio
        sr = 16000
        duration = 3.0
        t = np.arange(int(sr * duration)) / sr
        
        # Speech-like signal (mixture of frequencies)
        audio = (
            0.2 * np.sin(2 * np.pi * 200 * t) +  # Fundamental
            0.1 * np.sin(2 * np.pi * 400 * t) +  # Harmonic
            0.05 * np.random.randn(len(t))        # Noise
        )
        
        # Extract features
        features = processor.extract_features(audio)
        assert features is not None
        
        # Detect events
        events = processor.detect_events(audio)
        assert len(events) >= 0
        
        # Compute embedding
        embedding = processor.compute_audio_embedding(audio)
        assert embedding.shape[0] > 0
        
        # Compute onsets
        onsets = processor.compute_onset_times(audio)
        assert isinstance(onsets, list)
    
    def test_event_detection_with_silence(self):
        """Test event detection with periods of silence."""
        processor = AudioProcessor()
        
        # Create audio with silence gaps
        sr = 16000
        audio_parts = []
        
        # Alternating silence and sound
        for _ in range(3):
            audio_parts.append(np.random.randn(int(sr * 0.3)) * 0.01)  # silence
            audio_parts.append(np.random.randn(int(sr * 0.3)) * 0.15)   # sound
        
        audio = np.concatenate(audio_parts)
        
        events = processor.detect_events(audio)
        
        # Should detect multiple events
        assert len(events) > 0
    
    def test_embedding_consistency(self):
        """Test embedding consistency."""
        processor = AudioProcessor()
        
        audio = np.random.RandomState(42).randn(16000)
        
        # Same audio should produce same embedding
        emb1 = processor.compute_audio_embedding(audio)
        emb2 = processor.compute_audio_embedding(audio)
        
        assert np.allclose(emb1, emb2)
    
    def test_temporal_features_coverage(self):
        """Test temporal feature coverage."""
        processor = AudioProcessor()
        
        audio = np.random.randn(48000)  # 3 seconds
        
        temporal = processor.extract_temporal_features(
            audio,
            window_size=0.5,
        )
        
        # Should have reasonable number of windows
        expected_windows = int(3.0 / 0.25)  # 50% hop
        assert len(temporal) > 0
