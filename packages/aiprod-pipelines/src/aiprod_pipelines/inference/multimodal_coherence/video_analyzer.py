"""Video analysis for multimodal coherence.

Provides:
- Video feature extraction (optical flow, content)
- Motion analysis
- Scene detection
- Temporal video analysis
"""

from typing import Optional, Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class VideoFeature:
    """Container for video features."""
    
    def __init__(
        self,
        optical_flow: Optional[np.ndarray] = None,
        motion_magnitude: Optional[np.ndarray] = None,
        content_embedding: Optional[np.ndarray] = None,
        scene_cuts: Optional[List[int]] = None,
        temporal_features: Optional[Dict[int, np.ndarray]] = None,
    ):
        """
        Initialize video features.
        
        Args:
            optical_flow: Optical flow vectors [H, W, 2, T]
            motion_magnitude: Motion magnitude over time [T]
            content_embedding: Content embeddings [T, D]
            scene_cuts: Frame indices of scene cuts
            temporal_features: Per-frame feature vectors
        """
        self.optical_flow = optical_flow
        self.motion_magnitude = motion_magnitude
        self.content_embedding = content_embedding
        self.scene_cuts = scene_cuts or []
        self.temporal_features = temporal_features or {}


class MotionEvent:
    """Detected motion event."""
    
    def __init__(
        self,
        event_type: str,
        start_frame: int,
        end_frame: int,
        magnitude: float,
        direction: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize motion event.
        
        Args:
            event_type: Type of motion (fast, slow, static, cut)
            start_frame: Start frame index
            end_frame: End frame index
            magnitude: Motion magnitude
            direction: Average motion direction (x, y)
        """
        self.event_type = event_type
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.magnitude = magnitude
        self.duration_frames = end_frame - start_frame
        self.direction = direction


class VideoAnalyzer:
    """Analyze video for coherence scoring."""
    
    def __init__(self, fps: int = 30):
        """
        Initialize analyzer.
        
        Args:
            fps: Frames per second
        """
        self.fps = fps
    
    def extract_features(
        self,
        frames: np.ndarray,
        compute_optical_flow: bool = True,
    ) -> VideoFeature:
        """
        Extract video features.
        
        Args:
            frames: Video frames [T, H, W, C]
            compute_optical_flow: Whether to compute optical flow
            
        Returns:
            VideoFeature container
        """
        T, H, W, C = frames.shape
        
        optical_flow = None
        motion_magnitude = None
        
        if compute_optical_flow and T > 1:
            optical_flow, motion_magnitude = self._compute_optical_flow(frames)
        
        # Content embedding (simple approach)
        content_embedding = self._extract_content_embedding(frames)
        
        # Scene detection
        scene_cuts = self._detect_scene_cuts(frames)
        
        # Temporal features
        temporal_features = self._extract_temporal_features(frames)
        
        return VideoFeature(
            optical_flow=optical_flow,
            motion_magnitude=motion_magnitude,
            content_embedding=content_embedding,
            scene_cuts=scene_cuts,
            temporal_features=temporal_features,
        )
    
    def _compute_optical_flow(
        self,
        frames: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optical flow between consecutive frames.
        
        Args:
            frames: Video frames [T, H, W, C]
            
        Returns:
            Tuple of (optical_flow, motion_magnitude)
        """
        T, H, W, C = frames.shape
        
        # Convert to grayscale if needed
        if C > 1:
            gray_frames = np.dot(frames, [0.299, 0.587, 0.114])
        else:
            gray_frames = frames[..., 0]
        
        # Simple motion estimation via difference
        motion_magnitude = np.zeros(T - 1)
        flow_vectors = np.zeros((H, W, 2, T - 1))
        
        for t in range(T - 1):
            frame_diff = frames[t + 1] - frames[t]
            
            if C == 1:
                magnitude = np.abs(frame_diff[..., 0])
            else:
                magnitude = np.sqrt(np.sum(frame_diff ** 2, axis=-1))
            
            motion_magnitude[t] = np.mean(magnitude)
            
            # Estimate flow direction
            dy, dx = np.gradient(np.mean(frames[t], axis=-1))
            flow_vectors[:, :, 0, t] = dx
            flow_vectors[:, :, 1, t] = dy
        
        return flow_vectors, motion_magnitude
    
    def _extract_content_embedding(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract content embeddings for each frame.
        
        Args:
            frames: Video frames [T, H, W, C]
            
        Returns:
            Content embeddings [T, embedding_dim]
        """
        T, H, W, C = frames.shape
        
        # Simple content features
        embeddings = []
        
        for t in range(T):
            frame = frames[t]
            
            # Mean color
            color_mean = frame.reshape(-1, C).mean(axis=0)
            
            # Color variance
            color_std = frame.reshape(-1, C).std(axis=0)
            
            # Edge content
            edges = np.gradient(np.mean(frame, axis=-1))
            edge_sum = np.sqrt(edges[0] ** 2 + edges[1] ** 2).mean()
            
            emb = np.concatenate([color_mean, color_std, [edge_sum]])
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def _detect_scene_cuts(self, frames: np.ndarray) -> List[int]:
        """
        Detect scene cuts (abrupt changes).
        
        Args:
            frames: Video frames [T, H, W, C]
            
        Returns:
            List of frame indices with scene cuts
        """
        T = frames.shape[0]
        
        scene_cuts = []
        
        for t in range(1, T):
            diff = np.abs(frames[t] - frames[t - 1]).mean()
            
            # Threshold for scene cut
            if diff > 50:  # Adjust based on frame range
                scene_cuts.append(t)
        
        return scene_cuts
    
    def _extract_temporal_features(
        self,
        frames: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        """
        Extract per-frame temporal features.
        
        Args:
            frames: Video frames [T, H, W, C]
            
        Returns:
            Dictionary mapping frame index to features
        """
        features_dict = {}
        
        content_emb = self._extract_content_embedding(frames)
        
        for t in range(len(content_emb)):
            features_dict[t] = content_emb[t]
        
        return features_dict
    
    def detect_motion_events(self, frames: np.ndarray) -> List[MotionEvent]:
        """
        Detect motion events.
        
        Args:
            frames: Video frames [T, H, W, C]
            
        Returns:
            List of motion events
        """
        T = frames.shape[0]
        
        features = self.extract_features(frames)
        
        events = []
        
        if features.motion_magnitude is not None:
            motion = features.motion_magnitude
            
            # Normalize
            motion_norm = (motion - motion.min()) / (motion.max() - motion.min() + 1e-8)
            
            # Detect motion events
            in_event = False
            event_start = 0
            event_type = "static"
            
            static_threshold = 0.05
            fast_threshold = 0.5
            
            for t, mag in enumerate(motion_norm):
                if mag < static_threshold:
                    if in_event and event_type != "static":
                        events.append(MotionEvent(
                            event_type=event_type,
                            start_frame=event_start,
                            end_frame=t,
                            magnitude=motion[event_start:t].mean(),
                        ))
                    in_event = False
                
                elif mag < fast_threshold:
                    if not in_event:
                        event_start = t
                        event_type = "slow"
                    in_event = True
                
                else:
                    if not in_event:
                        event_start = t
                        event_type = "fast"
                    in_event = True
            
            if in_event:
                events.append(MotionEvent(
                    event_type=event_type,
                    start_frame=event_start,
                    end_frame=T,
                    magnitude=motion[event_start:].mean(),
                ))
        
        return events
    
    def compute_motion_magnitude(self, frames: np.ndarray) -> np.ndarray:
        """
        Compute motion magnitude timeline.
        
        Args:
            frames: Video frames [T, H, W, C]
            
        Returns:
            Motion magnitude per frame [T-1]
        """
        features = self.extract_features(frames)
        
        if features.motion_magnitude is not None:
            return features.motion_magnitude
        else:
            return np.zeros(frames.shape[0] - 1)
    
    def extract_visual_embedding(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract visual embedding for coherence scoring.
        
        Args:
            frames: Video frames [T, H, W, C]
            
        Returns:
            Visual embedding vector
        """
        features = self.extract_features(frames)
        
        embeddings = []
        
        # Content mean
        if features.content_embedding is not None:
            embeddings.append(features.content_embedding.mean(axis=0))
        
        # Motion summary
        if features.motion_magnitude is not None:
            motion_summary = np.array([
                features.motion_magnitude.mean(),
                np.std(features.motion_magnitude),
                np.max(features.motion_magnitude),
                np.percentile(features.motion_magnitude, 75),
            ])
            embeddings.append(motion_summary)
        
        if embeddings:
            return np.concatenate(embeddings)
        else:
            return np.zeros(64)


class VideoAnalysisResult:
    """Container for video analysis results."""
    
    def __init__(
        self,
        features: VideoFeature,
        motion_events: List[MotionEvent],
        embedding: np.ndarray,
        motion_magnitude: np.ndarray,
    ):
        """Initialize analysis result."""
        self.features = features
        self.motion_events = motion_events
        self.embedding = embedding
        self.motion_magnitude = motion_magnitude
        self.num_frames = len(motion_magnitude) + 1 if motion_magnitude is not None else 0


class BatchVideoAnalyzer:
    """Process multiple video streams."""
    
    def __init__(self, fps: int = 30):
        """Initialize batch analyzer."""
        self.analyzer = VideoAnalyzer(fps=fps)
    
    def process_batch(
        self,
        frames_list: List[np.ndarray],
    ) -> List[VideoAnalysisResult]:
        """
        Process multiple video streams.
        
        Args:
            frames_list: List of video frames [T, H, W, C]
            
        Returns:
            List of analysis results
        """
        results = []
        
        for frames in frames_list:
            features = self.analyzer.extract_features(frames)
            motion_events = self.analyzer.detect_motion_events(frames)
            embedding = self.analyzer.extract_visual_embedding(frames)
            motion_magnitude = self.analyzer.compute_motion_magnitude(frames)
            
            result = VideoAnalysisResult(
                features=features,
                motion_events=motion_events,
                embedding=embedding,
                motion_magnitude=motion_magnitude,
            )
            results.append(result)
        
        return results
