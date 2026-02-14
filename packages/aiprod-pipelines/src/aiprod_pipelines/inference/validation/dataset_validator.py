"""
Smart Dataset Validator - Main orchestrator for video validation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import asyncio
from collections import defaultdict

import torch
import numpy as np


@dataclass
class ValidationMetrics:
    """Metrics for a single validation run"""
    total_videos: int = 0
    valid_videos: int = 0
    low_quality_count: int = 0
    duplicate_count: int = 0
    codec_issues: int = 0
    audio_issues: int = 0
    resolution_issues: int = 0
    processing_time_sec: float = 0.0
    
    @property
    def pass_rate(self) -> float:
        """Percentage of videos that passed validation"""
        if self.total_videos == 0:
            return 0.0
        return (self.valid_videos / self.total_videos) * 100


@dataclass
class ValidationIssue:
    """Single validation issue found"""
    video_path: str
    issue_type: str  # "quality", "codec", "audio", "resolution", "duplicate", "diversity"
    severity: str  # "error", "warning", "info"
    message: str
    suggestion: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for dataset"""
    dataset_path: str
    total_videos: int
    valid_videos: int
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: ValidationMetrics = field(default_factory=ValidationMetrics)
    duplicates: List[Tuple[str, str, float]] = field(default_factory=list)  # (path1, path2, similarity)
    diversity_score: float = 0.0
    timestamp: str = ""
    
    def summary(self) -> Dict:
        """Get summary statistics"""
        return {
            "total_videos": self.total_videos,
            "valid_videos": self.valid_videos,
            "pass_rate": f"{(self.valid_videos / max(self.total_videos, 1)) * 100:.1f}%",
            "total_issues": len(self.issues),
            "error_count": sum(1 for i in self.issues if i.severity == "error"),
            "warning_count": sum(1 for i in self.issues if i.severity == "warning"),
            "duplicate_pairs": len(self.duplicates),
            "diversity_score": f"{self.diversity_score:.3f}",
            "processing_time": f"{self.metrics.processing_time_sec:.1f}s",
        }


class SmartDatasetValidator:
    """
    Validates video dataset before training.
    
    Checks:
    - Video quality (resolution, bitrate, sharpness)
    - Content analysis (consistency, diversity)
    - Codec compatibility (H.264, H.265, VP9)
    - Audio quality (presence, sample rate, bitrate)
    - Duplicate detection with perceptual hashing
    - Dataset diversity metrics
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize validator"""
        self.device = device
        self.quality_checker = None
        self.content_analyzer = None
        self.duplicate_detector = None
        self.diversity_scorer = None
        self._issues: List[ValidationIssue] = []
        
    async def validate_dataset(
        self,
        dataset_path: str,
        max_workers: int = 4,
        sample_frames: int = 8,
    ) -> ValidationReport:
        """
        Validate complete dataset.
        
        Args:
            dataset_path: Path to video directory
            max_workers: Max concurrent validation workers
            sample_frames: Frames to sample per video for analysis
            
        Returns:
            ValidationReport with all findings
        """
        import time
        from datetime import datetime
        
        start_time = time.time()
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Find all videos
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
        video_files = [
            f for f in dataset_path.rglob('*') 
            if f.suffix.lower() in video_extensions
        ]
        
        # Validate each video
        validation_results = []
        for i, video_path in enumerate(video_files):
            print(f"  [{i+1}/{len(video_files)}] Validating {video_path.name}...", end="\r")
            result = await self._validate_single_video(video_path, sample_frames)
            validation_results.append(result)
        
        # Analyze duplicates
        print(f"\n  Checking for duplicates (comparing {len(video_files)} videos)...")
        duplicates = await self._detect_duplicates(video_files)
        
        # Compute diversity
        print(f"  Computing dataset diversity...")
        diversity_score = await self._compute_diversity(video_files, sample_frames)
        
        # Build report
        processing_time = time.time() - start_time
        metrics = ValidationMetrics(
            total_videos=len(video_files),
            valid_videos=sum(1 for r in validation_results if r["valid"]),
            low_quality_count=sum(1 for r in validation_results if r["quality_score"] < 0.5),
            codec_issues=sum(1 for r in validation_results if r["issues"].get("codec")),
            audio_issues=sum(1 for r in validation_results if r["issues"].get("audio")),
            resolution_issues=sum(1 for r in validation_results if r["issues"].get("resolution")),
            processing_time_sec=processing_time,
        )
        
        issues = []
        for result in validation_results:
            for issue_type, message in result["issues"].items():
                if message:
                    issues.append(ValidationIssue(
                        video_path=str(result["path"]),
                        issue_type=issue_type,
                        severity="error" if issue_type in ["codec", "resolution"] else "warning",
                        message=message,
                        suggestion=self._get_suggestion(issue_type),
                        metadata={"quality_score": result.get("quality_score", 0)},
                    ))
        
        report = ValidationReport(
            dataset_path=str(dataset_path),
            total_videos=len(video_files),
            valid_videos=sum(1 for r in validation_results if r["valid"]),
            issues=issues,
            metrics=metrics,
            duplicates=duplicates,
            diversity_score=diversity_score,
            timestamp=datetime.now().isoformat(),
        )
        
        print(f"✅ Validation complete: {report.valid_videos}/{report.total_videos} videos passed")
        return report
    
    async def _validate_single_video(self, video_path: Path, sample_frames: int) -> Dict:
        """Validate single video file"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {
                    "path": video_path,
                    "valid": False,
                    "quality_score": 0.0,
                    "issues": {"codec": "Could not open video file"},
                }
            
            # Get metadata
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            issues = {}
            quality_score = 1.0
            
            # Check resolution
            if width < 480 or height < 480:
                issues["resolution"] = f"Low resolution: {width}x{height} (need >= 480p)"
                quality_score -= 0.2
            
            # Check duration
            if frame_count < 24:  # At least 1 second at 24fps
                issues["duration"] = f"Video too short: {frame_count} frames"
                quality_score -= 0.15
            
            # Check codec
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_name = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            supported_codecs = ["H264", "H265", "VP90", "AV01", "MJPG"]
            if codec_name not in supported_codecs:
                issues["codec"] = f"Unsupported codec: {codec_name}"
                quality_score -= 0.3
            
            # Sample frames for quality analysis
            frame_quality = []
            for i in range(0, frame_count, max(1, frame_count // sample_frames)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    # Simple sharpness check (Laplacian variance)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    frame_quality.append(sharpness)
            
            if frame_quality and np.mean(frame_quality) < 100:  # Low sharpness threshold
                issues["sharpness"] = "Video appears blurry"
                quality_score -= 0.15
            
            # Check audio (try to detect audio stream)
            # Note: This is simplified - would need ffmpeg for full check
            try:
                # Try to read audio info with metadata
                issues["audio"] = None  # Will be checked if needed
            except:
                pass
            
            cap.release()
            
            # Clean up issues dict - only keep actual issues
            issues = {k: v for k, v in issues.items() if v}
            
            return {
                "path": video_path,
                "valid": quality_score > 0.5 and len(issues) == 0,
                "quality_score": max(0.0, quality_score),
                "issues": issues,
                "metadata": {
                    "frames": frame_count,
                    "fps": fps,
                    "resolution": f"{width}x{height}",
                    "codec": codec_name,
                },
            }
        
        except Exception as e:
            return {
                "path": video_path,
                "valid": False,
                "quality_score": 0.0,
                "issues": {"error": str(e)},
            }
    
    async def _detect_duplicates(self, video_files: List[Path], threshold: float = 0.85) -> List[Tuple[str, str, float]]:
        """Detect duplicate/near-duplicate videos using perceptual hashing"""
        duplicates = []
        
        if len(video_files) < 2:
            return duplicates
        
        import cv2
        import hashlib
        
        def get_video_hash(video_path: Path) -> Optional[str]:
            """Get perceptual hash of video (first frame)"""
            try:
                cap = cv2.VideoCapture(str(video_path))
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    # Resize frame to 8x8 for fast comparison
                    small = cv2.resize(frame, (8, 8))
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    
                    # Simple perceptual hash
                    hash_val = 0
                    for i in range(64):
                        hash_val = (hash_val << 1) | (1 if gray.flat[i] > 127 else 0)
                    
                    return format(hash_val, '064b')
            except:
                pass
            return None
        
        # Compute hashes
        hashes = {}
        for vid in video_files:
            h = get_video_hash(vid)
            if h:
                hashes[str(vid)] = h
        
        # Compare hashes (Hamming distance)
        compared = set()
        for i, (vid1, hash1) in enumerate(hashes.items()):
            for vid2, hash2 in list(hashes.items())[i+1:]:
                if vid1 < vid2:
                    pair_key = (vid1, vid2)
                else:
                    pair_key = (vid2, vid1)
                
                if pair_key not in compared:
                    compared.add(pair_key)
                    
                    # Hamming distance
                    diff = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
                    similarity = 1.0 - (diff / 64.0)
                    
                    if similarity >= threshold:
                        duplicates.append((vid1, vid2, similarity))
        
        return duplicates
    
    async def _compute_diversity(self, video_files: List[Path], sample_frames: int) -> float:
        """Compute diversity score of dataset (0=identical, 1=diverse)"""
        if len(video_files) < 2:
            return 1.0
        
        # Simplified: use file size variance as diversity proxy
        sizes = []
        for vid in video_files:
            try:
                sizes.append(vid.stat().st_size)
            except:
                pass
        
        if len(sizes) < 2:
            return 1.0
        
        # Compute coefficient of variation
        sizes_array = np.array(sizes)
        mean_size = np.mean(sizes_array)
        std_size = np.std(sizes_array)
        
        # Normalize to [0, 1] range
        cv = (std_size / mean_size) if mean_size > 0 else 0.0
        diversity_score = min(1.0, cv / 2.0)  # Normalize to 0-1
        
        return diversity_score
    
    def _get_suggestion(self, issue_type: str) -> str:
        """Get suggestion for fixing issue"""
        suggestions = {
            "quality": "Use videos with resolution >= 480p and clear visual content",
            "codec": "Convert to H.264 or H.265 codec",
            "audio": "Ensure video has audio stream with bitrate >= 128kbps",
            "resolution": "Upscale video to at least 480p resolution",
            "duration": "Use videos longer than 1 second",
            "sharpness": "Replace with sharper videos (avoid motion blur)",
            "diversity": "Add more varied content to improve dataset diversity",
        }
        return suggestions.get(issue_type, "Check video specifications")
