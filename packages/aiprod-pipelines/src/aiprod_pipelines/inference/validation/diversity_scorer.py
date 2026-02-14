"""Diversity scorer for datasets"""

from typing import List, Optional
import numpy as np


class DiversityScorer:
    """Computes diversity metrics for video datasets"""
    
    def __init__(self):
        pass
    
    def compute_file_size_diversity(self, file_sizes: List[int]) -> float:
        """
        Compute diversity based on file sizes.
        
        Theory: Wide range of file sizes suggests different content types/lengths
        
        Args:
            file_sizes: List of file sizes in bytes
            
        Returns:
            Diversity score 0-1
        """
        if len(file_sizes) < 2:
            return 1.0
        
        sizes_array = np.array(file_sizes, dtype=np.float64)
        
        # Coefficient of variation
        mean_size = np.mean(sizes_array)
        std_size = np.std(sizes_array)
        
        if mean_size == 0:
            return 0.0
        
        cv = std_size / mean_size
        
        # Normalize: cv=0 → diversity=0, cv=2 → diversity=1
        diversity = min(1.0, cv / 2.0)
        
        return float(diversity)
    
    def compute_duration_diversity(self, durations_sec: List[float]) -> float:
        """
        Compute diversity based on video durations.
        
        Args:
            durations_sec: List of video durations in seconds
            
        Returns:
            Diversity score 0-1
        """
        if len(durations_sec) < 2:
            return 1.0
        
        durations = np.array(durations_sec, dtype=np.float64)
        
        # Coefficient of variation
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        if mean_duration == 0:
            return 0.0
        
        cv = std_duration / mean_duration
        diversity = min(1.0, cv / 2.0)
        
        return float(diversity)
    
    def compute_fps_diversity(self, fps_list: List[float]) -> float:
        """
        Compute diversity based on frame rates.
        
        Args:
            fps_list: List of FPS values
            
        Returns:
            Diversity score 0-1
        """
        if len(fps_list) < 2:
            return 1.0
        
        fps_array = np.array(fps_list, dtype=np.float64)
        
        # Count unique FPS values
        unique_fps = len(np.unique(np.round(fps_array, 1)))
        max_possible = len(fps_list)
        
        # Diversity = how many different FPS values
        diversity = unique_fps / max_possible if max_possible > 0 else 0.0
        
        return float(diversity)
    
    def compute_resolution_diversity(self, resolutions: List[tuple]) -> float:
        """
        Compute diversity based on video resolutions.
        
        Args:
            resolutions: List of (width, height) tuples
            
        Returns:
            Diversity score 0-1
        """
        if len(resolutions) < 2:
            return 1.0
        
        # Count unique resolutions
        unique_resolutions = len(set(resolutions))
        max_possible = len(resolutions)
        
        diversity = unique_resolutions / max_possible if max_possible > 0 else 0.0
        
        return float(diversity)
    
    def compute_combined_diversity(
        self,
        file_sizes: Optional[List[int]] = None,
        durations: Optional[List[float]] = None,
        fps_list: Optional[List[float]] = None,
        resolutions: Optional[List[tuple]] = None,
        weights: Optional[dict] = None,
    ) -> float:
        """
        Compute overall diversity using multiple metrics.
        
        Args:
            file_sizes: List of file sizes
            durations: List of durations
            fps_list: List of FPS values
            resolutions: List of resolutions
            weights: Dict of weights for each metric (default: equal weights)
            
        Returns:
            Combined diversity score 0-1
        """
        default_weights = {
            "file_size": 0.25,
            "duration": 0.25,
            "fps": 0.25,
            "resolution": 0.25,
        }
        
        if weights is None:
            weights = default_weights
        
        # Compute individual scores
        scores = {}
        
        if file_sizes:
            scores["file_size"] = self.compute_file_size_diversity(file_sizes)
        
        if durations:
            scores["duration"] = self.compute_duration_diversity(durations)
        
        if fps_list:
            scores["fps"] = self.compute_fps_diversity(fps_list)
        
        if resolutions:
            scores["resolution"] = self.compute_resolution_diversity(resolutions)
        
        # Weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric, score in scores.items():
            weight = weights.get(metric, 0.25)
            weighted_sum += score * weight
            total_weight += weight
        
        combined_diversity = (weighted_sum / total_weight) if total_weight > 0 else 0.5
        
        return float(combined_diversity)
