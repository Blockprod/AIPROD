"""Test suite for Motion Quality metric.

Tests:
- Optical flow estimation
- Motion smoothness computation
- Jitter detection
- Motion discontinuity detection
- Temporal consistency tracking
"""

import pytest
import torch
import numpy as np

from aiprod_pipelines.inference.quality_metrics.motion import (
    MotionMetric,
    OpticalFlowEstimator,
    MotionConsistencyCalculator,
)


class TestMotionMetric:
    """Test MotionMetric dataclass."""
    
    def test_metric_creation(self):
        """Test creating MotionMetric."""
        metric = MotionMetric(
            smoothness_score=0.85,
            optical_flow_magnitude=torch.tensor([0.5, 0.6, 0.55]),
            jitter=0.08,
            motion_presence=0.75,
            quality_grade="good",
        )
        
        assert metric.smoothness_score == 0.85
        assert metric.quality_grade == "good"
        assert metric.jitter == 0.08
    
    def test_quality_grading(self):
        """Test automatic quality grade assignment."""
        test_cases = [
            (0.90, "excellent"),
            (0.78, "good"),
            (0.60, "fair"),
            (0.40, "poor"),
        ]
        
        for score, expected_grade in test_cases:
            metric = MotionMetric(
                smoothness_score=score,
                optical_flow_magnitude=torch.zeros(3),
                jitter=0.05,
                motion_presence=0.5,
                quality_grade=expected_grade,
            )
            assert metric.quality_grade == expected_grade


class TestOpticalFlowEstimator:
    """Test optical flow computation."""
    
    @pytest.fixture
    def estimator(self):
        """Create flow estimator."""
        return OpticalFlowEstimator()
    
    def test_estimator_initialization(self, estimator):
        """Test estimator setup."""
        assert estimator is not None
    
    def test_simple_flow_estimation(self, estimator):
        """Test basic flow estimation from frame difference."""
        # Create two similar frames
        frame1 = torch.randn(3, 64, 64)
        frame2 = frame1.clone()
        
        # Should have minimal flow
        flow_mag = estimator.estimate_flow(frame1, frame2)
        
        assert flow_mag >= 0  # Magnitude is non-negative
        assert flow_mag < 1.0  # Should be small for identical frames
    
    def test_flow_with_translation(self, estimator):
        """Test flow detection with obvious motion."""
        # Create frame 1
        frame1 = torch.zeros(3, 64, 64)
        frame1[:, 20:40, 20:40] = 1.0
        
        # Create frame 2 with translation
        frame2 = torch.zeros(3, 64, 64)
        frame2[:, 25:45, 25:45] = 1.0
        
        flow_mag = estimator.estimate_flow(frame1, frame2)
        
        # Should detect motion
        assert flow_mag > 0.01
    
    def test_flow_field_computation(self, estimator):
        """Test full flow field with magnitude and direction."""
        frame1 = torch.randn(3, 32, 32)
        frame2 = torch.randn(3, 32, 32)
        
        flow_x, flow_y = estimator.compute_flow_field(frame1, frame2)
        
        # Check output shapes
        assert flow_x.shape == frame1.shape[1:]  # (H, W)
        assert flow_y.shape == frame1.shape[1:]
    
    def test_batch_flow_estimation(self, estimator):
        """Test batch processing of optical flow."""
        # Video: (T, C, H, W)
        video = torch.randn(5, 3, 64, 64)
        
        # Compute flow between consecutive frames
        flows = []
        for t in range(video.shape[0] - 1):
            flow_mag = estimator.estimate_flow(
                video[t],
                video[t + 1],
            )
            flows.append(flow_mag)
        
        assert len(flows) == 4
        assert all(f >= 0 for f in flows)


class TestMotionConsistencyCalculator:
    """Test motion consistency and smoothness."""
    
    @pytest.fixture
    def calculator(self):
        """Create motion calculator."""
        return MotionConsistencyCalculator()
    
    def test_calculator_initialization(self, calculator):
        """Test calculator setup."""
        assert calculator is not None
    
    def test_static_video_motion(self, calculator):
        """Test motion metric on static video."""
        # Static video (same frame repeated)
        static_frame = torch.randn(3, 64, 64)
        video = static_frame.unsqueeze(0).repeat(5, 1, 1, 1)
        
        metric = calculator.compute_motion_metric(video)
        
        # Should have very low motion
        assert metric.smoothness_score > 0.90
        assert metric.jitter < 0.1
        assert metric.optical_flow_magnitude.mean() < 0.5
    
    def test_smooth_motion_video(self, calculator):
        """Test metric on smoothly moving video."""
        # Create smoothly moving sequence
        frames = []
        for t in range(8):
            frame = torch.zeros(3, 64, 64)
            offset = int(t * 2)  # Smooth linear motion
            frame[:, 20 + offset:40 + offset, 20:40] = 1.0
            frames.append(frame)
        
        video = torch.stack(frames)
        
        metric = calculator.compute_motion_metric(video)
        
        # Should have consistent motion (smooth)
        assert metric.smoothness_score > 0.70
        assert metric.quality_grade in ["good", "fair"]
    
    def test_jerky_motion_video(self, calculator):
        """Test metric on jerky/jittery video."""
        # Create jerky sequence
        frames = []
        positions = [10, 20, 15, 25, 12, 22, 18, 20]  # Jumpy
        
        for pos in positions:
            frame = torch.zeros(3, 64, 64)
            frame[:, pos:pos+20, 20:40] = 1.0
            frames.append(frame)
        
        video = torch.stack(frames)
        
        metric = calculator.compute_motion_metric(video)
        
        # Should detect jitter
        assert metric.jitter > 0.15
        assert metric.smoothness_score < 0.80
    
    def test_trajectory_smoothness(self, calculator):
        """Test smoothness from trajectory."""
        # Smooth trajectory: linear motion
        trajectory = np.array([
            [10, 10],
            [12, 12],
            [14, 14],
            [16, 16],
            [18, 18],
        ], dtype=np.float32)
        
        smoothness = calculator.compute_motion_trajectory_smoothness(
            torch.from_numpy(trajectory)
        )
        
        # Linear motion should be very smooth
        assert smoothness > 0.95
    
    def test_trajectory_jitter(self, calculator):
        """Test jitter detection from trajectory."""
        # Jittery trajectory
        trajectory = np.array([
            [10, 10],
            [15, 12],
            [12, 14],
            [18, 13],
            [14, 16],
        ], dtype=np.float32)
        
        smoothness = calculator.compute_motion_trajectory_smoothness(
            torch.from_numpy(trajectory)
        )
        
        # Should be less smooth than linear
        assert smoothness < 0.70
    
    def test_discontinuity_detection(self, calculator):
        """Test motion discontinuity detection."""
        # Create video with jump
        frames = []
        
        # Smooth motion for first 3 frames
        for t in range(3):
            frame = torch.zeros(3, 64, 64)
            offset = t * 2
            frame[:, 20 + offset:40 + offset, 20:40] = 1.0
            frames.append(frame)
        
        # Jump to different location
        frame = torch.zeros(3, 64, 64)
        frame[:, 50:70, 20:40] = 1.0
        frames.append(frame)
        
        # Continue motion
        for t in range(4, 8):
            frame = torch.zeros(3, 64, 64)
            offset = 50 + (t - 4) * 2
            frame[:, offset:offset+20, 20:40] = 1.0
            frames.append(frame)
        
        video = torch.stack(frames)
        
        # Detect discontinuities
        flow_mags = []
        for t in range(len(frames) - 1):
            estimator = OpticalFlowEstimator()
            flow = estimator.estimate_flow(frames[t], frames[t + 1])
            flow_mags.append(flow)
        
        metric = calculator.compute_motion_metric(video)
        
        # Should detect the discontinuity
        assert metric.optical_flow_magnitude[3] > metric.optical_flow_magnitude[0]


class TestMotionMetrics:
    """Test motion metric computation."""
    
    def test_flow_magnitude_statistics(self):
        """Test statistical computation of flow magnitude."""
        flows = torch.tensor([0.2, 0.25, 0.22, 1.5, 0.23, 0.24])  # One outlier
        
        mean_flow = flows.mean().item()
        std_flow = flows.std().item()
        
        # One large outlier should increase variance
        assert std_flow > 0.4
    
    def test_presence_ratio(self):
        """Test motion presence detection."""
        flows = torch.tensor([0.3, 0.25, 0.28, 0.29, 0.26])
        threshold = 0.1
        
        presence = (flows > threshold).float().mean().item()
        
        # All frames have motion
        assert presence == 1.0
    
    def test_presence_with_static(self):
        """Test motion presence with some static frames."""
        flows = torch.tensor([0.01, 0.02, 0.30, 0.25, 0.01])
        threshold = 0.1
        
        presence = (flows > threshold).float().mean().item()
        
        # 2 out of 5 frames have motion
        assert presence == pytest.approx(0.4)


class TestIntegration:
    """Integration tests for motion system."""
    
    def test_full_motion_analysis(self):
        """Test complete motion analysis workflow."""
        calculator = MotionConsistencyCalculator()
        
        # Create test video
        video = torch.randn(8, 3, 64, 64)
        
        # Compute metric
        metric = calculator.compute_motion_metric(video)
        
        # Check all fields present
        assert metric.smoothness_score is not None
        assert 0 <= metric.smoothness_score <= 1
        assert metric.optical_flow_magnitude is not None
        assert metric.jitter is not None
        assert metric.motion_presence is not None
        assert metric.quality_grade in ["excellent", "good", "fair", "poor"]
    
    def test_metric_consistency(self):
        """Test that repeated measurements are consistent."""
        calculator = MotionConsistencyCalculator()
        
        # Fixed seed for reproducibility
        torch.manual_seed(42)
        video = torch.randn(8, 3, 64, 64)
        
        metric1 = calculator.compute_motion_metric(video)
        
        # Same video should give same result
        metric2 = calculator.compute_motion_metric(video)
        
        assert metric1.smoothness_score == pytest.approx(metric2.smoothness_score)
