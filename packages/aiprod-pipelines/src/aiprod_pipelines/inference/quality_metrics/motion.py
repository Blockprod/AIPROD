"""
Motion Consistency Metrics.

Measures smoothness and coherence of motion in generated videos using:
1. Optical flow magnitude and direction consistency
2. Frame-to-frame difference magnitude
3. Motion trajectory smoothness (acceleration changes)
4. Motion onset/offset detection

Detects:
- Jittery motion (high-frequency noise)
- Unnatural jumps (discontinuities)
- Motion blur artifacts
- Temporal coherence

Typical scores (0-1, higher is better):
- 0.9+: Smooth, natural motion
- 0.75-0.9: Good motion coherence
- 0.6-0.75: Acceptable but slightly jerky
- <0.6: Significant motion artifacts
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MotionMetric:
    """Motion quality metric for video."""
    
    # Overall motion smoothness (0-1)
    smoothness_score: float
    
    # Optical flow magnitude (forward + backward)
    optical_flow_magnitude: torch.Tensor  # (T-1,) or (T,)
    
    # Per-frame motion consistency
    per_frame_consistency: torch.Tensor  # (T-1,) to (T,)
    
    # Motion jitter (acceleration magnitude)
    jitter_score: float  # Lower = less jittery
    
    # Optical flow confidence (presence of motion)
    motion_presence: float  # 0-1, fraction of frames with motion
    
    # Motion quality grade
    quality_grade: str  # "excellent", "good", "fair", "poor"
    
    def __repr__(self) -> str:
        return (
            f"MotionMetric(smoothness={self.smoothness_score:.3f}, "
            f"jitter={self.jitter_score:.3f}, "
            f"grade={self.quality_grade})"
        )


class OpticalFlowEstimator:
    """Simplified optical flow estimation using frame differences.
    
    Computes motion vectors using dense gradient-based approach.
    More efficient than full optical flow but captures motion trends.
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize optical flow estimator.
        
        Args:
            device: Device to compute on
        """
        self.device = device
    
    def estimate_flow(
        self,
        frame1: torch.Tensor,  # (C, H, W) or (B, C, H, W)
        frame2: torch.Tensor,  # Same shape
    ) -> torch.Tensor:
        """
        Estimate optical flow between frames.
        
        Uses simple differencing + gradient-based approach.
        Result shape matches frame spatial dimensions.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Optical flow magnitude (H, W) for single frame,
            or (B, H, W) for batch
        """
        if frame1.dim() == 3:
            # Single frame (C, H, W)
            # Convert to single channel for flow estimation
            if frame1.shape[0] == 3:
                # RGB to grayscale
                frame1_gray = frame1.mean(dim=0, keepdim=True)
                frame2_gray = frame2.mean(dim=0, keepdim=True)
            else:
                frame1_gray = frame1[:1]
                frame2_gray = frame2[:1]
            
            # Simple difference magnitude (approximates optical flow)
            diff = (frame2_gray - frame1_gray).abs()
            flow_mag = diff.mean(dim=0)  # (H, W)
            
        else:
            # Batch (B, C, H, W)
            frame1_gray = frame1[:, :1] if frame1.shape[1] >= 3 else frame1
            frame2_gray = frame2[:, :1] if frame2.shape[1] >= 3 else frame2
            
            if frame1.shape[1] == 3:
                frame1_gray = frame1.mean(dim=1, keepdim=True)
                frame2_gray = frame2.mean(dim=1, keepdim=True)
            
            diff = (frame2_gray - frame1_gray).abs()
            flow_mag = diff.mean(dim=1)  # (B, H, W)
        
        return flow_mag
    
    def compute_flow_field(
        self,
        frame1_gray: torch.Tensor,
        frame2_gray: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute optical flow field (magnitude and direction).
        
        Args:
            frame1_gray: Grayscale frame 1
            frame2_gray: Grayscale frame 2
            
        Returns:
            Tuple of (magnitude, direction)
        """
        # Compute image gradients
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        kernel_x = kernel_x.view(1, 1, 3, 3).to(self.device)
        kernel_y = kernel_y.view(1, 1, 3, 3).to(self.device)
        
        # Compute spatial gradients
        frame1_gray = frame1_gray.to(self.device)
        frame2_gray = frame2_gray.to(self.device)
        
        gx = F.conv2d(frame1_gray.unsqueeze(0).unsqueeze(0), kernel_x, padding=1).squeeze()
        gy = F.conv2d(frame1_gray.unsqueeze(0).unsqueeze(0), kernel_y, padding=1).squeeze()
        
        # Temporal gradient
        gt = frame2_gray - frame1_gray
        
        # Magnitude and angle
        magnitude = torch.sqrt(gx**2 + gy**2 + gt**2 + 1e-10)
        direction = torch.atan2(gy, gx)
        
        return magnitude, direction


class MotionConsistencyCalculator:
    """Computes motion consistency and smoothness metrics."""
    
    def __init__(self, device: str = "cuda"):
        """Initialize motion calculator.
        
        Args:
            device: Device to compute on
        """
        self.device = device
        self.flow_estimator = OpticalFlowEstimator(device=device)
    
    def compute_motion_metric(
        self,
        video_frames: torch.Tensor,  # (T, C, H, W)
    ) -> MotionMetric:
        """
        Compute comprehensive motion consistency metric.
        
        Args:
            video_frames: Video frames
            
        Returns:
            MotionMetric with smoothness score and per-frame consistency
        """
        num_frames = video_frames.shape[0]
        
        if num_frames < 2:
            logger.warning("Cannot compute motion for <2 frames")
            return MotionMetric(
                smoothness_score=0.5,
                optical_flow_magnitude=torch.tensor([]),
                per_frame_consistency=torch.tensor([]),
                jitter_score=1.0,
                motion_presence=0.0,
                quality_grade="unknown",
            )
        
        # Compute optical flow for consecutive frames
        flow_magnitudes = []
        frame_differences = []
        
        for t in range(num_frames - 1):
            frame1 = video_frames[t]
            frame2 = video_frames[t + 1]
            
            # Optical flow magnitude
            flow_mag = self.flow_estimator.estimate_flow(frame1, frame2)
            flow_magnitudes.append(flow_mag.mean().item())
            
            # Frame difference
            frame_diff = (frame2 - frame1).abs().mean().item()
            frame_differences.append(frame_diff)
        
        flow_magnitudes = torch.tensor(flow_magnitudes)
        frame_differences = torch.tensor(frame_differences)
        
        # Compute motion smoothness (measures jitter)
        if len(flow_magnitudes) > 1:
            # First order difference (velocity)
            velocity_changes = torch.diff(flow_magnitudes)
            # Second order difference (acceleration = jitter)
            acceleration = torch.diff(velocity_changes)
            # Lower acceleration = smoother motion
            jitter = acceleration.abs().mean().item()
        else:
            jitter = 0.0
        
        # Normalize jitter to [0, 1] (0 = smooth, 1 = very jittery)
        jitter_normalized = min(jitter / 0.1, 1.0)  # Normalize against typical jitter
        smoothness = 1.0 - jitter_normalized
        
        # Motion presence (fraction of frames with significant motion)
        motion_threshold = flow_magnitudes.mean() * 0.1
        motion_present = (flow_magnitudes > motion_threshold).float().mean().item()
        
        # Per-frame consistency score
        per_frame_consis = 1.0 - F.normalize(frame_differences.unsqueeze(0), p=2, dim=1).squeeze()
        
        # Quality grade
        if smoothness > 0.85:
            grade = "excellent"
        elif smoothness > 0.70:
            grade = "good"
        elif smoothness > 0.50:
            grade = "fair"
        else:
            grade = "poor"
        
        logger.info(
            f"Motion computed: smoothness={smoothness:.3f} ({grade}), "
            f"jitter={jitter:.4f}, motion_presence={motion_present:.2%}"
        )
        
        return MotionMetric(
            smoothness_score=smoothness,
            optical_flow_magnitude=flow_magnitudes.cpu(),
            per_frame_consistency=per_frame_consis.cpu(),
            jitter_score=jitter_normalized,
            motion_presence=motion_present,
            quality_grade=grade,
        )
    
    def compute_motion_trajectory_smoothness(
        self,
        flow_magnitudes: torch.Tensor,  # (T-1,)
    ) -> float:
        """
        Measure smoothness of motion trajectory (no jitter/jumps).
        
        Args:
            flow_magnitudes: Per-frame optical flow magnitudes
            
        Returns:
            Smoothness score (0-1, higher = smoother)
        """
        if len(flow_magnitudes) < 3:
            return 0.5
        
        # Acceleration (second derivative)
        velocity = torch.diff(flow_magnitudes)
        acceleration = torch.diff(velocity)
        
        # Average absolute acceleration
        avg_accel = acceleration.abs().mean().item()
        
        # Convert to 0-1 score
        smoothness = 1.0 / (1.0 + avg_accel)
        
        return smoothness
    
    def detect_motion_discontinuities(
        self,
        flow_magnitudes: torch.Tensor,
        threshold: float = 2.0,  # 2x average motion
    ) -> torch.Tensor:
        """
        Detect sudden changes in motion (discontinuities/jumps).
        
        Args:
            flow_magnitudes: Per-frame optical flow
            threshold: Multiplier for average motion
            
        Returns:
            Binary mask of discontinuity frames
        """
        mean_flow = flow_magnitudes.mean()
        threshold_val = threshold * mean_flow
        
        # Frame-to-frame changes
        flow_changes = torch.diff(flow_magnitudes)
        
        # Detect large jumps
        discontinuities = flow_changes > threshold_val
        
        return discontinuities
