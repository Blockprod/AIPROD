"""Tests for video validation system"""

import pytest
import tempfile
from pathlib import Path
import asyncio
from unittest.mock import Mock, patch

from aiprod_pipelines.inference.validation import (
    SmartDatasetValidator,
    ValidationReport,
    VideoQualityChecker,
    ContentAnalyzer,
    DuplicateDetector,
    DiversityScorer,
)


class TestVideoQualityChecker:
    """Test quality analysis"""
    
    @pytest.mark.asyncio
    async def test_quality_checker_initialization(self):
        checker = VideoQualityChecker()
        assert checker is not None
    
    def test_quality_score_creation(self):
        from aiprod_pipelines.inference.validation.quality_checker import QualityScore
        
        score = QualityScore(
            overall=0.8,
            sharpness=150.5,
            brightness=128,
            contrast=45.2,
            temporal_stability=0.85,
            bitrate_mbps=5.2,
            resolution_score=0.9,
            codec_efficiency=0.88,
        )
        
        assert score.overall == 0.8
        assert score.sharpness == 150.5


class TestContentAnalyzer:
    """Test content analysis"""
    
    @pytest.mark.asyncio
    async def test_content_analyzer_initialization(self):
        analyzer = ContentAnalyzer()
        assert analyzer is not None
    
    def test_content_features_creation(self):
        from aiprod_pipelines.inference.validation.content_analyzer import ContentFeatures
        
        features = ContentFeatures(
            motion_level=0.5,
            scene_count=3,
            color_diversity=0.72,
            object_presence=0.6,
            consistency=0.91,
        )
        
        assert features.motion_level == 0.5
        assert features.scene_count == 3


class TestDuplicateDetector:
    """Test duplicate detection"""
    
    def test_perceptual_hash_creation(self):
        import numpy as np
        from PIL import Image
        
        detector = DuplicateDetector(hash_size=8)
        
        # Create synthetic frame
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # This would normally call the hash function
        # We're just testing initialization here
        assert detector.hash_size == 8
    
    def test_hamming_distance(self):
        detector = DuplicateDetector()
        
        hash1 = "1010101010101010"
        hash2 = "1010101010101010"
        
        distance = detector.hamming_distance(hash1, hash2)
        assert distance == 0
    
    def test_hamming_distance_different(self):
        detector = DuplicateDetector()
        
        hash1 = "1111111111111111"
        hash2 = "0000000000000000"
        
        distance = detector.hamming_distance(hash1, hash2)
        assert distance == 16
    
    def test_hash_similarity(self):
        detector = DuplicateDetector()
        
        hash1 = "1010101010101010"
        hash2 = "1010111010101010"  # 1 bit difference
        
        similarity = detector.hash_similarity(hash1, hash2)
        assert 0.9 < similarity < 1.0


class TestDiversityScorer:
    """Test diversity calculation"""
    
    def test_file_size_diversity(self):
        scorer = DiversityScorer()
        
        # All same size
        sizes = [1000, 1000, 1000, 1000]
        diversity = scorer.compute_file_size_diversity(sizes)
        assert diversity == 0.0
        
        # Wide range of sizes
        sizes = [100, 500, 1000, 5000, 10000]
        diversity = scorer.compute_file_size_diversity(sizes)
        assert 0.0 < diversity <= 1.0
    
    def test_duration_diversity(self):
        scorer = DiversityScorer()
        
        # All same duration
        durations = [10.0, 10.0, 10.0, 10.0]
        diversity = scorer.compute_duration_diversity(durations)
        assert diversity == 0.0
        
        # Varied durations
        durations = [5.0, 10.0, 15.0, 20.0, 30.0]
        diversity = scorer.compute_duration_diversity(durations)
        assert 0.0 < diversity <= 1.0
    
    def test_fps_diversity(self):
        scorer = DiversityScorer()
        
        # All same FPS
        fps_list = [30.0, 30.0, 30.0, 30.0]
        diversity = scorer.compute_fps_diversity(fps_list)
        assert diversity == 0.25  # 1 unique out of 4
        
        # Different FPS values
        fps_list = [24.0, 30.0, 60.0, 24.0]
        diversity = scorer.compute_fps_diversity(fps_list)
        assert diversity == 0.75  # 3 unique out of 4
    
    def test_resolution_diversity(self):
        scorer = DiversityScorer()
        
        # All same resolution
        resolutions = [(1920, 1080), (1920, 1080), (1920, 1080)]
        diversity = scorer.compute_resolution_diversity(resolutions)
        assert diversity == 1/3  # 1 unique out of 3
        
        # Different resolutions
        resolutions = [(1920, 1080), (1280, 720), (640, 480), (1920, 1080)]
        diversity = scorer.compute_resolution_diversity(resolutions)
        assert diversity == 0.75  # 3 unique out of 4
    
    def test_combined_diversity(self):
        scorer = DiversityScorer()
        
        file_sizes = [1000, 2000, 3000, 4000]
        durations = [5.0, 10.0, 15.0, 20.0]
        fps_list = [24.0, 30.0, 60.0, 24.0]
        resolutions = [(1920, 1080), (1280, 720), (640, 480), (1920, 1080)]
        
        diversity = scorer.compute_combined_diversity(
            file_sizes=file_sizes,
            durations=durations,
            fps_list=fps_list,
            resolutions=resolutions,
        )
        
        assert 0.0 <= diversity <= 1.0


class TestSmartDatasetValidator:
    """Test complete validation pipeline"""
    
    def test_validator_initialization(self):
        validator = SmartDatasetValidator()
        assert validator is not None
        assert validator.device in ["cuda", "cpu"]
    
    def test_validation_report_creation(self):
        report = ValidationReport(
            dataset_path="/test/path",
            total_videos=100,
            valid_videos=85,
            diversity_score=0.82,
        )
        
        assert report.total_videos == 100
        assert report.valid_videos == 85
        
        summary = report.summary()
        assert summary["total_videos"] == 100
        assert "85.0%" in summary["pass_rate"]
    
    def test_validation_metrics(self):
        from aiprod_pipelines.inference.validation.dataset_validator import ValidationMetrics
        
        metrics = ValidationMetrics(
            total_videos=100,
            valid_videos=85,
            low_quality_count=10,
            codec_issues=3,
            audio_issues=2,
            resolution_issues=0,
            processing_time_sec=45.2,
        )
        
        assert metrics.pass_rate == 85.0


class TestValidationIssue:
    """Test validation issue reporting"""
    
    def test_validation_issue_creation(self):
        from aiprod_pipelines.inference.validation.dataset_validator import ValidationIssue
        
        issue = ValidationIssue(
            video_path="/videos/test.mp4",
            issue_type="quality",
            severity="warning",
            message="Video appears blurry",
            suggestion="Use sharper videos",
            metadata={"quality_score": 0.4},
        )
        
        assert issue.video_path == "/videos/test.mp4"
        assert issue.severity == "warning"
        assert issue.metadata["quality_score"] == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
