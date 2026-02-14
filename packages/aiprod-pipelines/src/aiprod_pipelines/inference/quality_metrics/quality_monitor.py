"""
Quality Monitoring and Aggregation Engine.

Combines multiple quality metrics (FVVR, LPIPS, Motion) into unified quality score.
Supports:
- Per-frame aggregation
- Temporal averaging
- Meta-metric computation (consistency across metrics)
- Trend analysis during inference
- Quality degradation detection

Overall quality = w_fvvr * FVVR + w_lpips * (1 - LPIPS) + w_motion * Motion_Smoothness
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Unified quality metric combining multiple dimensions."""
    
    # Individual component scores (normalized to [0, 1])
    fvvr_score: float          # Prompt alignment (semantic)
    lpips_score: float         # Perceptual quality (higher = lower distance)
    motion_score: float        # Motion smoothness
    
    # Aggregate scores
    overall_quality: float     # Weighted combination of above
    
    # Meta-metrics
    consistency: float         # How much do metrics agree?
    confidence: float          # Confidence in overall score
    
    # Per-frame breakdown
    per_frame_quality: torch.Tensor  # (num_frames,)
    
    # Quality grade
    grade: str  # "excellent", "good", "fair", "poor"
    
    # Diagnostics
    dominant_issue: Optional[str] = None  # What's limiting quality?
    
    def __repr__(self) -> str:
        return (
            f"QualityScore(overall={self.overall_quality:.3f}, "
            f"grade={self.grade}, fvvr={self.fvvr_score:.3f}, "
            f"lpips={self.lpips_score:.3f}, motion={self.motion_score:.3f})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = asdict(self)
        if isinstance(result.get("per_frame_quality"), torch.Tensor):
            result["per_frame_quality"] = result["per_frame_quality"].cpu().tolist()
        return result


class QualityMonitor:
    """Monitors and aggregates quality metrics during inference.
    
    Tracks:
    - Per-step quality progression
    - Trend detection (improving/stable/declining)
    - Early exit opportunities
    - Quality degradation alerts
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        device: str = "cuda",
    ):
        """
        Initialize quality monitor.
        
        Args:
            weights: Weights for metric combination (fvvr, lpips, motion)
            device: Device to compute on
        """
        self.weights = weights or {
            "fvvr": 0.40,      # Prompt alignment is most important
            "lpips": 0.30,     # But visual quality matters
            "motion": 0.30,    # Motion quality matters
        }
        
        # Validate weights sum to 1
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1: {total}, normalizing")
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}
        
        self.device = device
        self.history: List[QualityScore] = []
        self.per_step_scores: List[float] = []
    
    def aggregate_metrics(
        self,
        fvvr: float,
        lpips: float,
        motion: float,
        per_frame_quality: Optional[torch.Tensor] = None,
    ) -> QualityScore:
        """
        Combine multiple metrics into unified quality score.
        
        Args:
            fvvr: FVVR score (0-1, higher = better)
            lpips: LPIPS distance (0+, lower = better), converted to (0-1)
            motion: Motion smoothness (0-1, higher = better)
            per_frame_quality: Optional per-frame quality tensor
            
        Returns:
            QualityScore with unified metric
        """
        # Normalize LPIPS to [0, 1] (assume typical range 0.0-0.5)
        lpips_normalized = max(0, 1.0 - lpips * 2)
        
        # Compute weighted aggregate
        overall = (
            self.weights["fvvr"] * fvvr +
            self.weights["lpips"] * lpips_normalized +
            self.weights["motion"] * motion
        )
        
        # Compute metric consistency (agreement)
        # If metrics disagree, confidence is lower
        scores = torch.tensor([fvvr, lpips_normalized, motion])
        mean_score = scores.mean().item()
        variance = scores.var().item()
        
        # Consistency = inverse of variance (0 variance = perfect agreement)
        consistency = 1.0 / (1.0 + variance)
        
        # Confidence based on consistency
        confidence = consistency * min(1.0, overall + 0.3)  # Higher overall = more confident
        
        # Grade assignment
        if overall >= 0.85:
            grade = "excellent"
        elif overall >= 0.70:
            grade = "good"
        elif overall >= 0.50:
            grade = "fair"
        else:
            grade = "poor"
        
        # Detect dominant limiting factor
        component_scores = {
            "fvvr": fvvr,
            "lpips": lpips_normalized,
            "motion": motion,
        }
        limiting_component = min(component_scores, key=component_scores.get)
        if component_scores[limiting_component] < overall - 0.15:
            dominant_issue = limiting_component
        else:
            dominant_issue = None
        
        # Create per-frame scores if not provided
        if per_frame_quality is None:
            per_frame_quality = torch.full(
                (1,),
                overall,
                dtype=torch.float32,
            )
        
        score = QualityScore(
            fvvr_score=fvvr,
            lpips_score=lpips_normalized,
            motion_score=motion,
            overall_quality=overall,
            consistency=consistency,
            confidence=confidence,
            per_frame_quality=per_frame_quality.cpu(),
            grade=grade,
            dominant_issue=dominant_issue,
        )
        
        logger.info(
            f"Quality aggregated: {overall:.3f} ({grade}), "
            f"consistency={consistency:.3f}, dominant_issue={dominant_issue}"
        )
        
        return score
    
    def add_score(self, step: int, quality_score: QualityScore):
        """
        Add quality score measurement.
        
        Args:
            step: Inference step number
            quality_score: Computed quality score
        """
        self.history.append(quality_score)
        self.per_step_scores.append(quality_score.overall_quality)
    
    def get_trend(self, window: int = 5) -> str:
        """
        Detect trend in quality progression.
        
        Args:
            window: Lookback window size
            
        Returns:
            "improving", "stable", or "declining"
        """
        if len(self.per_step_scores) < window:
            return "unknown"
        
        recent = self.per_step_scores[-window:]
        first_half = sum(recent[:window//2]) / (window // 2)
        second_half = sum(recent[window//2:]) / (len(recent) - window // 2)
        
        diff = second_half - first_half
        
        if diff > 0.02:
            return "improving"
        elif diff < -0.02:
            return "declining"
        else:
            return "stable"
    
    def should_early_exit(self, threshold: float = 0.85) -> bool:
        """
        Check if quality has converged to acceptable level.
        
        Args:
            threshold: Quality threshold for early exit
            
        Returns:
            True if last 3 steps above threshold
        """
        if len(self.per_step_scores) < 3:
            return False
        
        recent_three = self.per_step_scores[-3:]
        return all(score >= threshold for score in recent_three)
    
    def detect_quality_degradation(self, threshold: float = 0.15) -> bool:
        """
        Detect sudden quality drop (potential issue).
        
        Args:
            threshold: Minimum quality change to flag
            
        Returns:
            True if recent drop detected
        """
        if len(self.per_step_scores) < 2:
            return False
        
        prev_score = self.per_step_scores[-2]
        curr_score = self.per_step_scores[-1]
        
        return (prev_score - curr_score) > threshold
    
    def get_statistics(self) -> Dict[str, float]:
        """Get overall statistics from all measurements.
        
        Returns:
            Dictionary with min, max, mean, std of overall quality
        """
        if not self.per_step_scores:
            return {}
        
        scores = torch.tensor(self.per_step_scores)
        
        return {
            "min": scores.min().item(),
            "max": scores.max().item(),
            "mean": scores.mean().item(),
            "std": scores.std().item(),
            "final": self.per_step_scores[-1] if self.per_step_scores else 0.0,
        }
    
    def get_diagnostics_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality diagnostics report.
        
        Returns:
            Dictionary with detailed quality analysis
        """
        stats = self.get_statistics()
        trend = self.get_trend()
        
        if self.history:
            latest = self.history[-1]
            avg_consistency = sum(h.consistency for h in self.history) / len(self.history)
        else:
            latest = None
            avg_consistency = 0.0
        
        return {
            "statistics": stats,
            "trend": trend,
            "early_exit_available": self.should_early_exit(),
            "quality_degradation": self.detect_quality_degradation(),
            "average_consistency": avg_consistency,
            "latest_measurement": latest.to_dict() if latest else None,
            "measurement_count": len(self.history),
        }


class QualityAggregator:
    """Simple utility for one-shot quality aggregation.
    
    Used when you have all metrics available and need  single combined score.
    """
    
    @staticmethod
    def aggregate(
        fvvr: float,
        lpips: float,
        motion: float,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Quick aggregation of metrics.
        
        Args:
            fvvr: Semantic alignment score
            lpips: Perceptual distance (normalized 0-1)
            motion: Motion smoothness score
            weights: Optional custom weights
            
        Returns:
            Overall quality score (0-1)
        """
        if weights is None:
            weights = {"fvvr": 0.4, "lpips": 0.3, "motion": 0.3}
        
        # Normalize LPIPS
        lpips_norm = max(0, 1.0 - lpips * 2)
        
        overall = (
            weights["fvvr"] * fvvr +
            weights["lpips"] * lpips_norm +
            weights["motion"] * motion
        )
        
        return overall
