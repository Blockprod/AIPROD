"""Content analyzer for videos"""

from dataclasses import dataclass
from typing import List, Dict
import numpy as np


@dataclass
class ContentFeatures:
    """Features extracted from video content"""
    motion_level: float  # 0-1, amount of motion
    scene_count: int  # Number of detected scenes
    color_diversity: float  # 0-1, diversity of colors
    object_presence: float  # 0-1, presence of objects (faces, text, etc)
    consistency: float  # 0-1, content consistency across frames


class ContentAnalyzer:
    """Analyzes video content for semantic understanding"""
    
    def __init__(self):
        pass
    
    async def analyze_content(self, video_path: str, sample_frames: int = 16) -> ContentFeatures:
        """
        Analyze content of video.
        
        Args:
            video_path: Path to video
            sample_frames: Frames to analyze
            
        Returns:
            ContentFeatures with analysis results
        """
        try:
            import cv2
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return ContentFeatures(0, 0, 0, 0, 0)
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            motion_diffs = []
            color_histograms = []
            scene_changes = []
            
            prev_frame = None
            
            for i in range(0, frame_count, max(1, frame_count // sample_frames)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    # Motion detection
                    if prev_frame is not None:
                        flow = cv2.calcOpticalFlowFarneback(
                            prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                        )
                        motion = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
                        motion_diffs.append(motion)
                    
                    # Color diversity
                    hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    color_histograms.append(hist.flatten())
                    
                    prev_frame = gray.copy()
            
            cap.release()
            
            if not motion_diffs:
                return ContentFeatures(0, 0, 0, 0, 0)
            
            # Compute metrics
            motion_level = np.clip(np.mean(motion_diffs) / 50.0, 0, 1)  # Normalize motion
            
            # Scene detection (large motion changes = scene cut)
            scene_count = sum(1 for d in motion_diffs[1:] if d > np.mean(motion_diffs) * 2)
            
            # Color diversity (entropy of histogram)
            color_histograms = np.array(color_histograms)
            color_diversity = np.mean([
                -np.sum(h * np.log(h + 1e-6)) / np.log(h.shape[0])
                for h in color_histograms if h.sum() > 0
            ])
            color_diversity = np.clip(color_diversity, 0, 1)
            
            # Object detection (simplified - edge detection)
            object_presence = 0.5  # Placeholder
            
            # Frame consistency
            if len(color_histograms) > 1:
                consistency = np.mean([
                    1.0 - np.sum(np.abs(color_histograms[i] - color_histograms[i+1])) / 2.0
                    for i in range(len(color_histograms)-1)
                ])
            else:
                consistency = 1.0
            consistency = np.clip(consistency, 0, 1)
            
            return ContentFeatures(
                motion_level=motion_level,
                scene_count=scene_count,
                color_diversity=color_diversity,
                object_presence=object_presence,
                consistency=consistency,
            )
        
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return ContentFeatures(0, 0, 0, 0, 0)
