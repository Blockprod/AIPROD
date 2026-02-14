"""Audio processing for multimodal coherence analysis.

Provides:
- Audio feature extraction (MFCC, spectral, temporal)
- Audio embedding generation
- Audio event detection
- Temporal audio analysis
"""

from typing import Optional, Dict, Tuple, List, Any
import numpy as np
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class AudioFeature:
    """Container for audio features."""
    
    def __init__(
        self,
        mfcc: Optional[np.ndarray] = None,
        mel_spectrogram: Optional[np.ndarray] = None,
        chroma: Optional[np.ndarray] = None,
        temporal_envelope: Optional[np.ndarray] = None,
        onset_strength: Optional[np.ndarray] = None,
        zero_crossing_rate: Optional[np.ndarray] = None,
    ):
        """
        Initialize audio features.
        
        Args:
            mfcc: Mel-frequency cepstral coefficients
            mel_spectrogram: Mel-scale spectrogram
            chroma: Chromagram (musical pitch)
            temporal_envelope: Energy envelope over time
            onset_strength: Onset detection strength
            zero_crossing_rate: Zero crossing rate
        """
        self.mfcc = mfcc
        self.mel_spectrogram = mel_spectrogram
        self.chroma = chroma
        self.temporal_envelope = temporal_envelope
        self.onset_strength = onset_strength
        self.zero_crossing_rate = zero_crossing_rate


class AudioEventType:
    """Audio event classification."""
    
    SPEECH = "speech"
    MUSIC = "music"
    SILENCE = "silence"
    NOISE = "noise"
    TRANSIENT = "transient"  # Sharp attack (percussion, etc.)
    SUSTAINED = "sustained"  # Long-held tone


class AudioEvent:
    """Detected audio event."""
    
    def __init__(
        self,
        event_type: str,
        start_time: float,
        end_time: float,
        confidence: float,
        features: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize audio event.
        
        Args:
            event_type: Type of audio event
            start_time: Start time in seconds
            end_time: End time in seconds
            confidence: Confidence score 0-1
            features: Additional features
        """
        self.event_type = event_type
        self.start_time = start_time
        self.end_time = end_time
        self.confidence = confidence
        self.duration = end_time - start_time
        self.features = features or {}


class AudioProcessor:
    """Process audio for coherence analysis."""
    
    def __init__(self, sr: int = 16000):
        """
        Initialize processor.
        
        Args:
            sr: Sample rate (Hz)
        """
        self.sr = sr
        self.n_mfcc = 13
        self.n_chroma = 12
        self.n_mel = 128
    
    def extract_features(
        self,
        audio: np.ndarray,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> AudioFeature:
        """
        Extract multimodal audio features.
        
        Args:
            audio: Audio waveform [1D array]
            frame_length: FFT frame length
            hop_length: Hop length
            
        Returns:
            AudioFeature container
        """
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not available, using basic features")
            return self._extract_basic_features(audio)
        
        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=frame_length,
            hop_length=hop_length,
        )
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mel,
            n_fft=frame_length,
            hop_length=hop_length,
        )
        
        # Chroma
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sr,
            n_chroma=self.n_chroma,
            n_fft=frame_length,
            hop_length=hop_length,
        )
        
        # Temporal envelope
        S = np.abs(librosa.stft(audio, n_fft=frame_length, hop_length=hop_length))
        envelope = np.mean(S, axis=0)
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sr,
            hop_length=hop_length,
        )
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]
        
        return AudioFeature(
            mfcc=mfcc,
            mel_spectrogram=mel_spec,
            chroma=chroma,
            temporal_envelope=envelope,
            onset_strength=onset_env,
            zero_crossing_rate=zcr,
        )
    
    def _extract_basic_features(self, audio: np.ndarray) -> AudioFeature:
        """Extract basic features without librosa."""
        # Simple MFCC-like via direct spectrogram
        n_fft = 2048
        hops = np.linspace(0, len(audio), n_fft // 512)
        envelope = np.array([np.abs(audio[int(h):int(h)+1024]).mean() 
                            for h in hops[:-1]])
        
        # Temporal envelope
        temporal = np.convolve(np.abs(audio), np.ones(512)/512, mode='same')
        
        return AudioFeature(
            temporal_envelope=envelope,
            onset_strength=np.gradient(envelope),
            zero_crossing_rate=np.array([]),
        )
    
    def detect_events(self, audio: np.ndarray) -> List[AudioEvent]:
        """
        Detect audio events.
        
        Args:
            audio: Audio waveform
            
        Returns:
            List of detected events
        """
        features = self.extract_features(audio)
        events = []
        
        # Simple event detection via energy
        if features.temporal_envelope is not None:
            envelope = features.temporal_envelope
            
            # Normalize
            env_norm = (envelope - envelope.min()) / (envelope.max() - envelope.min() + 1e-8)
            
            # Detect silence
            silent_threshold = 0.05
            speech_threshold = 0.2
            
            in_event = False
            event_start = 0
            event_type = AudioEventType.SILENCE
            
            for i, energy in enumerate(env_norm):
                time = i * 512 / self.sr
                
                if energy < silent_threshold:
                    if in_event and event_type != AudioEventType.SILENCE:
                        events.append(AudioEvent(
                            event_type=event_type,
                            start_time=event_start,
                            end_time=time,
                            confidence=0.8,
                        ))
                    in_event = False
                
                elif energy < speech_threshold:
                    if not in_event:
                        event_start = time
                        event_type = AudioEventType.NOISE
                    in_event = True
                
                else:
                    if not in_event:
                        event_start = time
                        event_type = AudioEventType.SPEECH
                    in_event = True
            
            if in_event:
                events.append(AudioEvent(
                    event_type=event_type,
                    start_time=event_start,
                    end_time=len(audio) / self.sr,
                    confidence=0.8,
                ))
        
        return events
    
    def compute_audio_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute audio embedding for coherence scoring.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Audio embedding vector
        """
        features = self.extract_features(audio)
        
        embeddings = []
        
        if features.mfcc is not None:
            embeddings.append(features.mfcc.mean(axis=1))
        
        if features.mel_spectrogram is not None:
            mel_mean = np.log(features.mel_spectrogram.mean(axis=1) + 1e-9)
            embeddings.append(mel_mean)
        
        if features.chroma is not None:
            embeddings.append(features.chroma.mean(axis=1))
        
        if features.temporal_envelope is not None:
            env_mean = np.array([
                features.temporal_envelope.mean(),
                np.std(features.temporal_envelope),
                np.max(features.temporal_envelope),
            ])
            embeddings.append(env_mean)
        
        if embeddings:
            return np.concatenate(embeddings)
        else:
            return np.zeros(64)
    
    def extract_temporal_features(
        self,
        audio: np.ndarray,
        window_size: float = 0.5,
    ) -> Dict[float, np.ndarray]:
        """
        Extract features over sliding windows.
        
        Args:
            audio: Audio waveform
            window_size: Window size in seconds
            
        Returns:
            Dictionary mapping time to feature vectors
        """
        window_samples = int(window_size * self.sr)
        hop_samples = window_samples // 2
        
        temporal_features = {}
        
        for start in range(0, len(audio) - window_samples, hop_samples):
            window = audio[start:start + window_samples]
            time = start / self.sr
            
            embedding = self.compute_audio_embedding(window)
            temporal_features[time] = embedding
        
        return temporal_features
    
    def compute_onset_times(self, audio: np.ndarray) -> List[float]:
        """
        Detect onset times (attack points).
        
        Args:
            audio: Audio waveform
            
        Returns:
            List of onset times in seconds
        """
        features = self.extract_features(audio)
        
        if features.onset_strength is None:
            return []
        
        onset_strength = features.onset_strength
        
        # Detect peaks
        threshold = np.mean(onset_strength) + 0.5 * np.std(onset_strength)
        
        onsets = []
        for i, strength in enumerate(onset_strength):
            if strength > threshold:
                if len(onsets) == 0 or i - onsets[-1] > 0.05 * self.sr / 512:
                    onsets.append(i * 512 / self.sr)
        
        return onsets


class AudioAnalysisResult:
    """Container for audio analysis results."""
    
    def __init__(
        self,
        features: AudioFeature,
        events: List[AudioEvent],
        embedding: np.ndarray,
        onset_times: List[float],
        temporal_features: Dict[float, np.ndarray],
    ):
        """Initialize analysis result."""
        self.features = features
        self.events = events
        self.embedding = embedding
        self.onset_times = onset_times
        self.temporal_features = temporal_features


class BatchAudioProcessor:
    """Process multiple audio streams."""
    
    def __init__(self, sr: int = 16000):
        """Initialize batch processor."""
        self.processor = AudioProcessor(sr=sr)
    
    def process_batch(
        self,
        audio_list: List[np.ndarray],
    ) -> List[AudioAnalysisResult]:
        """
        Process multiple audio streams.
        
        Args:
            audio_list: List of audio waveforms
            
        Returns:
            List of analysis results
        """
        results = []
        
        for audio in audio_list:
            features = self.processor.extract_features(audio)
            events = self.processor.detect_events(audio)
            embedding = self.processor.compute_audio_embedding(audio)
            onsets = self.processor.compute_onset_times(audio)
            temporal = self.processor.extract_temporal_features(audio)
            
            result = AudioAnalysisResult(
                features=features,
                events=events,
                embedding=embedding,
                onset_times=onsets,
                temporal_features=temporal,
            )
            results.append(result)
        
        return results
