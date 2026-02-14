"""Coherence scoring between audio and video.

Provides:
- Audio-video alignment scoring
- Temporal synchronization analysis
- Event correlation
- Coherence metrics
"""

from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class CoherenceMetrics:
    """Container for coherence metrics."""
    
    def __init__(
        self,
        temporal_alignment: float = 0.0,
        event_correlation: float = 0.0,
        spectro_temporal_match: float = 0.0,
        onset_synchrony: float = 0.0,
        energy_correlation: float = 0.0,
        overall_coherence: float = 0.0,
    ):
        """
        Initialize metrics.
        
        Args:
            temporal_alignment: Alignment score 0-1
            event_correlation: Event correlation 0-1
            spectro_temporal_match: Spectrogram-temporal match 0-1
            onset_synchrony: Onset time synchronization 0-1
            energy_correlation: Energy envelope correlation 0-1
            overall_coherence: Overall coherence 0-1
        """
        self.temporal_alignment = temporal_alignment
        self.event_correlation = event_correlation
        self.spectro_temporal_match = spectro_temporal_match
        self.onset_synchrony = onset_synchrony
        self.energy_correlation = energy_correlation
        self.overall_coherence = overall_coherence
        self.components = [
            temporal_alignment,
            event_correlation,
            spectro_temporal_match,
            onset_synchrony,
            energy_correlation,
        ]


class CoherenceScorer:
    """Score coherence between audio and video."""
    
    def __init__(
        self,
        audio_sr: int = 16000,
        video_fps: int = 30,
        frame_duration: float = 0.1,
    ):
        """
        Initialize scorer.
        
        Args:
            audio_sr: Audio sample rate
            video_fps: Video frames per second
            frame_duration: Duration per analysis frame (seconds)
        """
        self.audio_sr = audio_sr
        self.video_fps = video_fps
        self.frame_duration = frame_duration
        self.audio_hop = int(frame_duration * audio_sr)
        self.video_hop = max(1, int(frame_duration * video_fps))
    
    def score_coherence(
        self,
        audio_embedding: np.ndarray,
        video_embedding: np.ndarray,
        audio_temporal: Dict[float, np.ndarray],
        motion_magnitude: np.ndarray,
        audio_events: List[Any],
        motion_events: List[Any],
    ) -> CoherenceMetrics:
        """
        Compute multimodal coherence score.
        
        Args:
            audio_embedding: Global audio embedding
            video_embedding: Global visual embedding
            audio_temporal: Temporal audio features
            motion_magnitude: Video motion magnitude timeline
            audio_events: List of detected audio events
            motion_events: List of detected motion events
            
        Returns:
            CoherenceMetrics
        """
        # Compute individual components
        temporal_align = self._compute_temporal_alignment(
            audio_temporal,
            motion_magnitude,
        )
        
        event_corr = self._compute_event_correlation(
            audio_events,
            motion_events,
        )
        
        spectro_match = self._compute_spectro_temporal_match(
            audio_embedding,
            video_embedding,
        )
        
        onset_sync = self._compute_onset_synchrony(
            audio_events,
            motion_events,
        )
        
        energy_corr = self._compute_energy_correlation(
            audio_temporal,
            motion_magnitude,
        )
        
        # Weighted average for overall score
        overall = (
            0.25 * temporal_align +
            0.20 * event_corr +
            0.20 * spectro_match +
            0.20 * onset_sync +
            0.15 * energy_corr
        )
        
        return CoherenceMetrics(
            temporal_alignment=float(temporal_align),
            event_correlation=float(event_corr),
            spectro_temporal_match=float(spectro_match),
            onset_synchrony=float(onset_sync),
            energy_correlation=float(energy_corr),
            overall_coherence=float(overall),
        )
    
    def _compute_temporal_alignment(
        self,
        audio_temporal: Dict[float, np.ndarray],
        motion_magnitude: np.ndarray,
    ) -> float:
        """
        Compute temporal alignment score.
        
        Args:
            audio_temporal: Temporal audio features {time: features}
            motion_magnitude: Motion magnitude array
            
        Returns:
            Alignment score 0-1
        """
        if not audio_temporal or len(motion_magnitude) == 0:
            return 0.5
        
        # Convert temporal dict to array
        times = sorted(audio_temporal.keys())
        audio_features = np.array([audio_temporal[t] for t in times])
        
        # Normalize both
        audio_norm = (audio_features - audio_features.mean()) / (audio_features.std() + 1e-8)
        motion_norm = (motion_magnitude - motion_magnitude.mean()) / (motion_magnitude.std() + 1e-8)
        
        # Resample to same length
        if len(audio_norm) != len(motion_norm):
            audio_norm = np.interp(
                np.linspace(0, 1, len(motion_norm)),
                np.linspace(0, 1, len(audio_norm)),
                audio_norm[:, 0] if audio_norm.ndim > 1 else audio_norm,
            )
        
        # Correlation
        correlation = np.corrcoef(audio_norm, motion_norm)[0, 1]
        
        # Convert to 0-1 range
        alignment = (correlation + 1) / 2
        
        return alignment
    
    def _compute_event_correlation(
        self,
        audio_events: List[Any],
        motion_events: List[Any],
    ) -> float:
        """
        Compute correlation between audio and motion events.
        
        Args:
            audio_events: Audio event list
            motion_events: Motion event list
            
        Returns:
            Correlation score 0-1
        """
        if not audio_events or not motion_events:
            return 0.5
        
        # Create event timelines
        max_time = max(
            max([e.end_time for e in audio_events]) if hasattr(audio_events[0], 'end_time') else 0,
            max([e.end_frame / 30 for e in motion_events]) if hasattr(motion_events[0], 'end_frame') else 0,
        )
        
        if max_time == 0:
            return 0.5
        
        num_bins = int(max_time * 10)  # 100ms bins
        audio_timeline = np.zeros(num_bins)
        motion_timeline = np.zeros(num_bins)
        
        # Fill audio timeline
        for event in audio_events:
            if hasattr(event, 'start_time') and hasattr(event, 'end_time'):
                start_bin = int(event.start_time * 10)
                end_bin = int(event.end_time * 10)
                audio_timeline[max(0, start_bin):min(num_bins, end_bin)] = 1
        
        # Fill motion timeline
        for event in motion_events:
            if hasattr(event, 'start_frame') and hasattr(event, 'end_frame'):
                start_bin = int(event.start_frame / 3)  # 30fps / 10 = 3
                end_bin = int(event.end_frame / 3)
                motion_timeline[max(0, start_bin):min(num_bins, end_bin)] = 1
        
        # Compute overlap
        overlap = np.sum(audio_timeline * motion_timeline)
        union = np.sum(np.maximum(audio_timeline, motion_timeline))
        
        if union == 0:
            return 0.5
        
        correlation = overlap / union
        
        return correlation
    
    def _compute_spectro_temporal_match(
        self,
        audio_embedding: np.ndarray,
        video_embedding: np.ndarray,
    ) -> float:
        """
        Compute spectrogram-temporal match score.
        
        Args:
            audio_embedding: Audio embedding
            video_embedding: Video embedding
            
        Returns:
            Match score 0-1
        """
        if len(audio_embedding) == 0 or len(video_embedding) == 0:
            return 0.5
        
        # Normalize embeddings
        audio_norm = (audio_embedding - np.mean(audio_embedding)) / (np.std(audio_embedding) + 1e-8)
        video_norm = (video_embedding - np.mean(video_embedding)) / (np.std(video_embedding) + 1e-8)
        
        # Pad to same length
        min_len = min(len(audio_norm), len(video_norm))
        audio_trim = audio_norm[:min_len]
        video_trim = video_norm[:min_len]
        
        # Cosine similarity
        dot_product = np.dot(audio_trim, video_trim)
        magnitude_product = np.linalg.norm(audio_trim) * np.linalg.norm(video_trim)
        
        if magnitude_product == 0:
            return 0.5
        
        similarity = dot_product / magnitude_product
        
        # Convert to 0-1
        match = (similarity + 1) / 2
        
        return match
    
    def _compute_onset_synchrony(
        self,
        audio_events: List[Any],
        motion_events: List[Any],
    ) -> float:
        """
        Compute onset time synchronization.
        
        Args:
            audio_events: Audio events with onset times
            motion_events: Motion events with onset frames
            
        Returns:
            Synchrony score 0-1
        """
        audio_onsets = []
        motion_onsets = []
        
        # Extract onsets
        for event in audio_events:
            if hasattr(event, 'start_time'):
                audio_onsets.append(event.start_time)
        
        for event in motion_events:
            if hasattr(event, 'start_frame'):
                motion_onsets.append(event.start_frame / 30.0)  # Convert to seconds
        
        if not audio_onsets or not motion_onsets:
            return 0.5
        
        # Find closest pairs and compute synchrony
        total_sync = 0
        count = 0
        
        for a_onset in audio_onsets:
            # Find closest motion onset
            closest_m_onset = min(motion_onsets, key=lambda x: abs(x - a_onset))
            
            # Synchrony based on time difference
            time_diff = abs(a_onset - closest_m_onset)
            
            # Score: 1.0 if within 100ms, decays with distance
            sync = np.exp(-time_diff / 0.2)
            
            total_sync += sync
            count += 1
        
        if count == 0:
            return 0.5
        
        return total_sync / count
    
    def _compute_energy_correlation(
        self,
        audio_temporal: Dict[float, np.ndarray],
        motion_magnitude: np.ndarray,
    ) -> float:
        """
        Compute energy envelope correlation.
        
        Args:
            audio_temporal: Temporal audio features
            motion_magnitude: Motion magnitude timeline
            
        Returns:
            Correlation score 0-1
        """
        if not audio_temporal or len(motion_magnitude) == 0:
            return 0.5
        
        # Extract energy from audio (simple: L2 norm of features)
        times = sorted(audio_temporal.keys())
        audio_energy = np.array([
            np.linalg.norm(audio_temporal[t])
            for t in times
        ])
        
        if len(audio_energy) == 0 or len(motion_magnitude) == 0:
            return 0.5
        
        # Resample to same length
        if len(audio_energy) != len(motion_magnitude):
            audio_energy = np.interp(
                np.linspace(0, 1, len(motion_magnitude)),
                np.linspace(0, 1, len(audio_energy)),
                audio_energy,
            )
        
        # Normalize
        audio_norm = (audio_energy - audio_energy.mean()) / (audio_energy.std() + 1e-8)
        motion_norm = (motion_magnitude - motion_magnitude.mean()) / (motion_magnitude.std() + 1e-8)
        
        # Correlation
        correlation = np.corrcoef(audio_norm, motion_norm)[0, 1]
        
        # Convert to 0-1
        corr_score = (correlation + 1) / 2
        
        return corr_score
    
    def score_batch(
        self,
        audio_results: List[Any],
        video_results: List[Any],
    ) -> List[CoherenceMetrics]:
        """
        Score coherence for multiple pairs.
        
        Args:
            audio_results: List of audio analysis results
            video_results: List of video analysis results
            
        Returns:
            List of coherence metrics
        """
        metrics_list = []
        
        for audio, video in zip(audio_results, video_results):
            metrics = self.score_coherence(
                audio_embedding=audio.embedding,
                video_embedding=video.embedding,
                audio_temporal=audio.temporal_features,
                motion_magnitude=video.motion_magnitude,
                audio_events=audio.events,
                motion_events=video.motion_events,
            )
            metrics_list.append(metrics)
        
        return metrics_list


class CoherenceResult:
    """Complete coherence analysis result."""
    
    def __init__(
        self,
        metrics: CoherenceMetrics,
        audio_summary: Dict[str, Any],
        video_summary: Dict[str, Any],
        recommendations: List[str],
    ):
        """Initialize coherence result."""
        self.metrics = metrics
        self.audio_summary = audio_summary
        self.video_summary = video_summary
        self.recommendations = recommendations
    
    def is_coherent(self, threshold: float = 0.75) -> bool:
        """Check if coherence exceeds threshold."""
        return self.metrics.overall_coherence >= threshold
