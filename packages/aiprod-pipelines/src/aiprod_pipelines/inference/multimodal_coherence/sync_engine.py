"""Real-time audio-video synchronization engine.

Provides:
- Dynamic synchronization correction
- Latency compensation
- Frame/sample alignment
- Adaptive sync adjustment
"""

from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class SyncAlignment:
    """Sync alignment information."""
    
    def __init__(
        self,
        time_offset: float = 0.0,
        confidence: float = 1.0,
        adjustment_type: str = "none",
        latency_ms: float = 0.0,
    ):
        """
        Initialize alignment.
        
        Args:
            time_offset: Time offset in seconds (audio relative to video)
            confidence: Confidence in alignment 0-1
            adjustment_type: Type of adjustment (none, skip_frames, interpolate, resample)
            latency_ms: Detected latency in milliseconds
        """
        self.time_offset = time_offset
        self.confidence = confidence
        self.adjustment_type = adjustment_type
        self.latency_ms = latency_ms


class SyncMetrics:
    """Synchronization quality metrics."""
    
    def __init__(
        self,
        lip_sync_score: float = 0.0,
        audio_lead: float = 0.0,
        sync_stability: float = 0.0,
        frame_jitter: float = 0.0,
        total_drift: float = 0.0,
    ):
        """
        Initialize metrics.
        
        Args:
            lip_sync_score: Lip-sync quality 0-1
            audio_lead: Audio lead in seconds (positive = audio ahead)
            sync_stability: Temporal stability 0-1
            frame_jitter: Frame timing jitter 0-1
            total_drift: Accumulated drift in seconds
        """
        self.lip_sync_score = lip_sync_score
        self.audio_lead = audio_lead
        self.sync_stability = sync_stability
        self.frame_jitter = frame_jitter
        self.total_drift = total_drift


class SyncEngine:
    """Real-time synchronization engine."""
    
    def __init__(
        self,
        audio_sr: int = 16000,
        video_fps: int = 30,
        max_offset: float = 0.5,
    ):
        """
        Initialize sync engine.
        
        Args:
            audio_sr: Audio sample rate
            video_fps: Video frames per second
            max_offset: Maximum allowed offset in seconds
        """
        self.audio_sr = audio_sr
        self.video_fps = video_fps
        self.max_offset = max_offset
        self.frame_duration = 1.0 / video_fps
        
        # Tracking state
        self.cumulative_offset = 0.0
        self.frame_times = []
        self.sample_times = []
    
    def compute_alignment(
        self,
        audio_onsets: List[float],
        motion_onsets: List[float],
        audio_events: Optional[List[Any]] = None,
        motion_events: Optional[List[Any]] = None,
    ) -> SyncAlignment:
        """
        Compute optimal audio-video alignment.
        
        Args:
            audio_onsets: Audio onset times
            motion_onsets: Motion onset times
            audio_events: Audio events (optional)
            motion_events: Motion events (optional)
            
        Returns:
            SyncAlignment with recommended offset
        """
        if not audio_onsets or not motion_onsets:
            return SyncAlignment()
        
        # Find best offset by matching onsets
        best_offset = None
        best_score = -np.inf
        possible_offsets = []
        
        for audio_onset in audio_onsets:
            for motion_onset in motion_onsets:
                offset = audio_onset - motion_onset
                
                # Skip if too large
                if abs(offset) > self.max_offset:
                    continue
                
                # Score based on how well other onsets align
                score = self._score_offset(offset, audio_onsets, motion_onsets)
                
                if score > best_score:
                    best_score = score
                    best_offset = offset
                
                possible_offsets.append((offset, score))
        
        if best_offset is None:
            best_offset = 0.0
        
        # Determine confidence
        if len(possible_offsets) > 1:
            scores = [s for _, s in possible_offsets]
            confidence = (best_score - np.mean(scores)) / (np.std(scores) + 1e-8)
            confidence = np.clip(confidence, 0, 1)
        else:
            confidence = 0.5
        
        # Determine adjustment type
        if abs(best_offset) < 0.02:  # <20ms
            adjustment_type = "none"
        elif abs(best_offset) < 0.1:  # <100ms
            adjustment_type = "interpolate"
        else:
            adjustment_type = "resample"
        
        return SyncAlignment(
            time_offset=float(best_offset),
            confidence=float(confidence),
            adjustment_type=adjustment_type,
            latency_ms=float(best_offset * 1000),
        )
    
    def _score_offset(
        self,
        offset: float,
        audio_onsets: List[float],
        motion_onsets: List[float],
    ) -> float:
        """
        Score quality of offset.
        
        Args:
            offset: Time offset to test
            audio_onsets: Audio onset times
            motion_onsets: Motion onset times
            
        Returns:
            Score (higher is better)
        """
        alignment_errors = []
        
        for motion_onset in motion_onsets:
            adjusted_motion = motion_onset + offset
            
            # Find closest audio onset
            if audio_onsets:
                closest_audio = min(audio_onsets, key=lambda x: abs(x - adjusted_motion))
                error = abs(closest_audio - adjusted_motion)
                alignment_errors.append(error)
        
        if not alignment_errors:
            return 0.0
        
        # Score based on inverse of average error
        mean_error = np.mean(alignment_errors)
        score = 1.0 / (1.0 + mean_error)
        
        return score
    
    def apply_alignment(
        self,
        frames: np.ndarray,
        alignment: SyncAlignment,
    ) -> np.ndarray:
        """
        Apply alignment correction to frames.
        
        Args:
            frames: Video frames [T, H, W, C]
            alignment: Alignment to apply
            
        Returns:
            Adjusted frames
        """
        if abs(alignment.time_offset) < 1e-6:
            return frames
        
        T, H, W, C = frames.shape
        
        # Convert offset to frame shift
        frame_shift = alignment.time_offset * self.video_fps
        
        if alignment.adjustment_type == "skip_frames":
            # Skip/duplicate frames
            frame_shift_int = int(round(frame_shift))
            
            if frame_shift_int > 0:
                # Remove first frames
                adjusted = np.concatenate([
                    frames[frame_shift_int:],
                    frames[-1:].repeat(frame_shift_int, axis=0),
                ])
            else:
                # Add frames at start
                adjusted = np.concatenate([
                    frames[0:1].repeat(-frame_shift_int, axis=0),
                    frames,
                ])
        
        elif alignment.adjustment_type == "interpolate":
            # Interpolate between frames
            adjusted = np.zeros_like(frames)
            
            for t in range(T):
                src_t = t - frame_shift
                
                if src_t < 0:
                    adjusted[t] = frames[0]
                elif src_t >= T - 1:
                    adjusted[t] = frames[-1]
                else:
                    # Linear interpolation
                    t_low = int(np.floor(src_t))
                    t_high = int(np.ceil(src_t))
                    weight = src_t - t_low
                    
                    adjusted[t] = (1 - weight) * frames[t_low] + weight * frames[t_high]
        
        else:
            adjusted = frames
        
        return adjusted
    
    def compute_sync_metrics(
        self,
        alignment_history: List[SyncAlignment],
        audio_onsets: List[float],
        motion_onsets: List[float],
        total_duration: float,
    ) -> SyncMetrics:
        """
        Compute synchronization quality metrics.
        
        Args:
            alignment_history: History of alignments
            audio_onsets: Audio onsets
            motion_onsets: Motion onsets
            total_duration: Total duration in seconds
            
        Returns:
            SyncMetrics
        """
        # Lip-sync score (based on onset matching)
        lip_sync_score = 0.0
        if audio_onsets and motion_onsets:
            onset_errors = []
            for audio_onset in audio_onsets:
                closest_motion = min(motion_onsets, key=lambda x: abs(x - audio_onset))
                error = abs(audio_onset - closest_motion)
                onset_errors.append(error)
            
            mean_error = np.mean(onset_errors)
            # Convert to score: 1.0 if error <20ms, decays with distance
            lip_sync_score = np.exp(-mean_error / 0.02)
        
        # Audio lead (average offset)
        audio_lead = 0.0
        if alignment_history:
            audio_lead = np.mean([a.time_offset for a in alignment_history])
        
        # Sync stability (variance of offsets)
        sync_stability = 1.0
        if len(alignment_history) > 1:
            offsets = [a.time_offset for a in alignment_history]
            offset_std = np.std(offsets)
            # Stability: lower std = higher score
            sync_stability = np.exp(-offset_std / 0.05)
        
        # Frame jitter
        frame_jitter = 0.0
        if len(alignment_history) > 1:
            offsets = np.array([a.time_offset for a in alignment_history])
            frame_jitter = np.mean(np.abs(np.diff(offsets)))
        
        # Total drift
        total_drift = 0.0
        if alignment_history:
            total_drift = alignment_history[-1].time_offset
        
        return SyncMetrics(
            lip_sync_score=float(np.clip(lip_sync_score, 0, 1)),
            audio_lead=float(audio_lead),
            sync_stability=float(np.clip(sync_stability, 0, 1)),
            frame_jitter=float(frame_jitter),
            total_drift=float(total_drift),
        )
    
    def detect_sync_issues(
        self,
        alignment: SyncAlignment,
        metrics: SyncMetrics,
    ) -> List[str]:
        """
        Detect synchronization issues.
        
        Args:
            alignment: Current alignment
            metrics: Sync metrics
            
        Returns:
            List of detected issues
        """
        issues = []
        
        # Check offset
        if abs(alignment.time_offset) > 0.1:
            direction = "ahead" if alignment.time_offset > 0 else "behind"
            issues.append(f"Audio {int(abs(alignment.latency_ms))}ms {direction}")
        
        # Check stability
        if metrics.sync_stability < 0.7:
            issues.append("Unstable synchronization detected")
        
        # Check drift
        if abs(metrics.total_drift) > 0.5:
            issues.append("Excessive temporal drift")
        
        # Check jitter
        if metrics.frame_jitter > 0.02:
            issues.append("High frame timing jitter")
        
        # Check lip-sync
        if metrics.lip_sync_score < 0.75:
            issues.append("Poor lip-sync quality")
        
        return issues


class AdaptiveSyncController:
    """Adaptive synchronization with feedback."""
    
    def __init__(
        self,
        audio_sr: int = 16000,
        video_fps: int = 30,
    ):
        """Initialize controller."""
        self.engine = SyncEngine(audio_sr, video_fps)
        self.alignment_history = []
        self.metrics_history = []
    
    def update_sync(
        self,
        audio_onsets: List[float],
        motion_onsets: List[float],
    ) -> Dict[str, Any]:
        """
        Update synchronization with feedback.
        
        Args:
            audio_onsets: Current audio onsets
            motion_onsets: Current motion onsets
            
        Returns:
            Control parameters for adjustment
        """
        alignment = self.engine.compute_alignment(audio_onsets, motion_onsets)
        self.alignment_history.append(alignment)
        
        # Compute metrics
        metrics = self.engine.compute_sync_metrics(
            self.alignment_history,
            audio_onsets,
            motion_onsets,
            total_duration=max(audio_onsets + motion_onsets) if (audio_onsets or motion_onsets) else 0,
        )
        self.metrics_history.append(metrics)
        
        # Detect issues
        issues = self.engine.detect_sync_issues(alignment, metrics)
        
        return {
            "alignment": {
                "offset": alignment.time_offset,
                "confidence": alignment.confidence,
                "adjustment": alignment.adjustment_type,
                "latency_ms": alignment.latency_ms,
            },
            "metrics": {
                "lip_sync": metrics.lip_sync_score,
                "audio_lead": metrics.audio_lead,
                "stability": metrics.sync_stability,
                "jitter": metrics.frame_jitter,
                "drift": metrics.total_drift,
            },
            "issues": issues,
        }
    
    def get_adjustment_parameters(self) -> Dict[str, float]:
        """Get current adjustment parameters."""
        if not self.alignment_history:
            return {"offset": 0.0, "confidence": 0.0}
        
        latest = self.alignment_history[-1]
        
        return {
            "offset": latest.time_offset,
            "confidence": latest.confidence,
            "latency_ms": latest.latency_ms,
        }
