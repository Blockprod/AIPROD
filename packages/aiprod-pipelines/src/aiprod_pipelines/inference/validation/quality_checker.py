"""Quality checker module for individual videos"""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np


@dataclass
class QualityScore:
    """Quality metrics for a video"""
    overall: float  # 0-1 overall quality
    sharpness: float  # Laplacian variance
    brightness: float  # Average brightness (0-255)
    contrast: float  # Standard deviation of brightness
    temporal_stability: float  # Frame-to-frame difference
    bitrate_mbps: float  # Estimated bitrate
    resolution_score: float  # 0-1 based on resolution
    codec_efficiency: float  # 0-1 codec compression ratio


class VideoQualityChecker:
    """Analyzes individual video quality metrics"""
    
    def __init__(self):
        pass
    
    async def analyze_video(
        self,
        video_path: str,
        sample_frames: int = 16,
    ) -> QualityScore:
        """
        Analyze video quality metrics.
        
        Args:
            video_path: Path to video file
            sample_frames: Number of frames to analyze
            
        Returns:
            QualityScore with detailed metrics
        """
        import cv2
        from pathlib import Path
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return QualityScore(0, 0, 0, 0, 0, 0, 0, 0)
        
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # File size (for bitrate estimation)
            file_size_bytes = Path(video_path).stat().st_size if Path(video_path).exists() else 0
            duration_sec = frame_count / fps if fps > 0 else 1
            bitrate_mbps = (file_size_bytes * 8) / (1_000_000 * duration_sec) if duration_sec > 0 else 0
            
            # Sample frames
            sharpness_values = []
            brightness_values = []
            frame_diffs = []
            prev_frame = None
            
            for i in range(0, frame_count, max(1, frame_count // sample_frames)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Sharpness (Laplacian variance)
                    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                    sharpness_values.append(sharpness)
                    
                    # Brightness
                    brightness = np.mean(gray)
                    brightness_values.append(brightness)
                    
                    # Temporal stability
                    if prev_frame is not None:
                        diff = np.mean(np.abs(gray.astype(float) - prev_frame.astype(float)))
                        frame_diffs.append(diff)
                    
                    prev_frame = gray.copy()
            
            cap.release()
            
            # Compute scores
            sharpness = np.mean(sharpness_values) if sharpness_values else 0
            brightness = np.mean(brightness_values) if brightness_values else 128
            contrast = np.std(brightness_values) if brightness_values else 0
            temporal_stability = 1.0 - (np.mean(frame_diffs) / 255.0) if frame_diffs else 0.8
            temporal_stability = np.clip(temporal_stability, 0, 1)
            
            # Resolution score (1.0 for 1080p+, 0.7 for 720p, 0.4 for 480p, etc)
            res_pixels = width * height
            if res_pixels >= 1920 * 1080:
                resolution_score = 1.0
            elif res_pixels >= 1280 * 720:
                resolution_score = 0.9
            elif res_pixels >= 854 * 480:
                resolution_score = 0.7
            elif res_pixels >= 640 * 360:
                resolution_score = 0.5
            else:
                resolution_score = 0.3
            
            # Codec efficiency (lower bitrate for same quality = better)
            codec_efficiency = 1.0 / (1 + bitrate_mbps / 10.0)  # Peak at 10mbps
            
            # Overall quality
            overall = (
                0.25 * min(1.0, sharpness / 500.0) +  # Normalize sharpness
                0.15 * (contrast / 50.0) +  # Normalize contrast
                0.25 * resolution_score +
                0.20 * temporal_stability +
                0.15 * codec_efficiency
            )
            overall = np.clip(overall, 0, 1)
            
            return QualityScore(
                overall=overall,
                sharpness=sharpness,
                brightness=brightness,
                contrast=contrast,
                temporal_stability=temporal_stability,
                bitrate_mbps=bitrate_mbps,
                resolution_score=resolution_score,
                codec_efficiency=codec_efficiency,
            )
        
        except Exception as e:
            print(f"Error analyzing video: {e}")
            return QualityScore(0, 0, 0, 0, 0, 0, 0, 0)
