"""
Video Input Validation System

Smart dataset quality checker for user-provided videos.
- Quality metrics scoring
- Content analysis neural net
- Diversity scorer
- Duplicate detection
- Resolution/codec validation
- Audio quality checking
- Batch validation with reporting
"""

from .dataset_validator import SmartDatasetValidator, ValidationReport, ValidationMetrics
from .quality_checker import VideoQualityChecker, QualityScore
from .content_analyzer import ContentAnalyzer
from .duplicate_detector import DuplicateDetector, DuplicateMatch
from .diversity_scorer import DiversityScorer

__all__ = [
    "SmartDatasetValidator",
    "ValidationReport",
    "ValidationMetrics",
    "VideoQualityChecker",
    "QualityScore",
    "ContentAnalyzer",
    "DuplicateDetector",
    "DuplicateMatch",
    "DiversityScorer",
]
