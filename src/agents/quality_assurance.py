"""
Quality Assurance for AIPROD - Video Validation & Certification

Validates generated videos against quality tier specifications:
- Video resolution, FPS, codec compliance
- Audio loudness, format, channel validation
- Post-production quality checks
- Generates QC reports with compliance certification
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class QCStatus(Enum):
    """Quality control status"""
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"
    PENDING = "pending"


class QCCheckType(Enum):
    """Types of quality checks"""
    VIDEO_RESOLUTION = "video_resolution"
    VIDEO_FPS = "video_fps"
    VIDEO_CODEC = "video_codec"
    VIDEO_BITRATE = "video_bitrate"
    AUDIO_FORMAT = "audio_format"
    AUDIO_LOUDNESS = "audio_loudness"
    AUDIO_CHANNELS = "audio_channels"
    AUDIO_CODEC = "audio_codec"
    COLOR_GRADING = "color_grading"
    EFFECTS = "effects"
    FILE_SIZE = "file_size"
    ARTIFACT_CHECK = "artifact_check"
    FLICKER_DETECTION = "flicker_detection"


@dataclass
class QCCheckResult:
    """Result of a single quality check"""
    check_type: QCCheckType
    specification: str          # Expected value
    actual_value: str           # Actual measured value
    passed: bool
    tolerance: Optional[str] = None  # e.g., "±5%", "-23±1 LUFS"
    message: str = ""
    severity: str = "info"      # "info", "warning", "error"


@dataclass
class QCReport:
    """Complete quality control report for a video"""
    job_id: str
    tier: str                   # "good", "high", "ultra"
    status: QCStatus
    
    # Video details
    video_resolution: str       # "1920x1080"
    video_fps: int
    video_codec: str
    video_bitrate_kbps: int
    video_color_space: str
    
    # Audio details
    audio_format: str
    audio_channels: int
    audio_codec: str
    audio_loudness_lufs: float
    
    # Checks
    checks: List[QCCheckResult]
    
    # Metadata
    generated_at: datetime
    qc_timestamp: datetime = None
    passed_at: Optional[datetime] = None
    
    # Summary
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    
    def __post_init__(self):
        if self.qc_timestamp is None:
            self.qc_timestamp = datetime.utcnow()
        self.total_checks = len(self.checks)
        self.passed_checks = sum(1 for c in self.checks if c.passed and c.severity == "info")
        self.warning_checks = sum(1 for c in self.checks if c.passed and c.severity == "warning")
        self.failed_checks = sum(1 for c in self.checks if not c.passed)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "tier": self.tier,
            "status": self.status.value,
            "video_specs": {
                "resolution": self.video_resolution,
                "fps": self.video_fps,
                "codec": self.video_codec,
                "bitrate_kbps": self.video_bitrate_kbps,
                "color_space": self.video_color_space
            },
            "audio_specs": {
                "format": self.audio_format,
                "channels": self.audio_channels,
                "codec": self.audio_codec,
                "loudness_lufs": self.audio_loudness_lufs
            },
            "qa_summary": {
                "total_checks": self.total_checks,
                "passed": self.passed_checks,
                "warnings": self.warning_checks,
                "failed": self.failed_checks,
                "pass_rate": f"{(self.passed_checks / self.total_checks * 100) if self.total_checks > 0 else 0:.1f}%"
            },
            "checks": [
                {
                    "check": c.check_type.value,
                    "specification": c.specification,
                    "actual": c.actual_value,
                    "status": "passed" if c.passed else "failed",
                    "message": c.message
                }
                for c in self.checks
            ],
            "qc_timestamp": self.qc_timestamp.isoformat(),
            "passed_at": self.passed_at.isoformat() if self.passed_at else None
        }


class QualityAssuranceEngine:
    """
    QA engine: Validates generated videos against tier specs
    
    Usage:
        qa = QualityAssuranceEngine()
        report = qa.validate_video(job_id, tier, video_metadata)
    """
    
    # Quality spec tolerances
    TOLERANCES = {
        QCCheckType.VIDEO_FPS: 0.5,           # ±0.5 fps
        QCCheckType.AUDIO_LOUDNESS: 1.0,      # ±1 LUFS
        QCCheckType.VIDEO_BITRATE: 0.1,       # ±10% of target
    }
    
    @classmethod
    def validate_video(
        cls,
        job_id: str,
        tier: str,
        video_metadata: Dict[str, Any]
    ) -> QCReport:
        """
        Comprehensive video validation against tier specs
        
        Args:
            job_id: Video generation job ID
            tier: Quality tier ("good", "high", "ultra")
            video_metadata: Video technical metadata from ffprobe or similar
                Expected keys: resolution, fps, codec, bitrate_kbps, color_space,
                              audio_format, audio_channels, audio_codec, audio_loudness_lufs
        
        Returns:
            QCReport with validation results
        """
        
        # Import specs dynamically to avoid circular imports
        from .quality_specs import QualitySpecRegistry
        
        # Get tier spec
        tier_spec = QualitySpecRegistry.get_tier_spec(tier)
        
        # Initialize checks list
        checks = []
        
        # VIDEO CHECKS
        # Resolution
        checks.append(cls._check_video_resolution(
            tier_spec.video.resolution,
            video_metadata.get("resolution", "unknown")
        ))
        
        # FPS
        checks.append(cls._check_video_fps(
            tier_spec.video.fps,
            video_metadata.get("fps", 0)
        ))
        
        # Codec
        checks.append(cls._check_video_codec(
            tier_spec.video.codec,
            video_metadata.get("codec", "unknown")
        ))
        
        # Bitrate
        checks.append(cls._check_video_bitrate(
            tier_spec.video.bitrate_kbps,
            video_metadata.get("bitrate_kbps", 0)
        ))
        
        # Color space
        checks.append(cls._check_color_space(
            tier_spec.video.color_space,
            video_metadata.get("color_space", "unknown")
        ))
        
        # AUDIO CHECKS
        # Format
        checks.append(cls._check_audio_format(
            tier_spec.audio.format_name,
            video_metadata.get("audio_format", "unknown")
        ))
        
        # Channels
        checks.append(cls._check_audio_channels(
            tier_spec.audio.channels,
            video_metadata.get("audio_channels", 0)
        ))
        
        # Codec
        checks.append(cls._check_audio_codec(
            tier_spec.audio.codec,
            video_metadata.get("audio_codec", "unknown")
        ))
        
        # Loudness
        checks.append(cls._check_audio_loudness(
            tier_spec.audio.loudness_lufs,
            video_metadata.get("audio_loudness_lufs", 0)
        ))
        
        # QUALITY CHECKS
        # Artifacts
        if video_metadata.get("artifact_score", 0) > 0.2:
            checks.append(QCCheckResult(
                check_type=QCCheckType.ARTIFACT_CHECK,
                specification="Minimal artifacts (score < 0.2)",
                actual_value=f"Artifact score: {video_metadata.get('artifact_score', 0):.3f}",
                passed=False,
                severity="warning",
                message="Some artifacts detected but acceptable for streaming"
            ))
        else:
            checks.append(QCCheckResult(
                check_type=QCCheckType.ARTIFACT_CHECK,
                specification="Minimal artifacts (score < 0.2)",
                actual_value=f"Artifact score: {video_metadata.get('artifact_score', 0):.3f}",
                passed=True,
                message="Clean video, no visible artifacts"
            ))
        
        # Flicker detection (for HIGH/ULTRA)
        if tier in ["high", "ultra"]:
            flicker_detected = video_metadata.get("flicker_detected", False)
            checks.append(QCCheckResult(
                check_type=QCCheckType.FLICKER_DETECTION,
                specification="No flicker detected",
                actual_value="Flicker: " + ("Yes" if flicker_detected else "No"),
                passed=not flicker_detected,
                severity="warning" if flicker_detected else "info",
                message="No temporal flicker detected" if not flicker_detected else "Minor flicker detected - review"
            ))
        
        # Determine overall status
        failed_checks = [c for c in checks if not c.passed and c.severity == "error"]
        warning_checks = [c for c in checks if not c.passed and c.severity == "warning"]
        
        if failed_checks:
            overall_status = QCStatus.FAILED
        elif warning_checks:
            overall_status = QCStatus.PASSED_WITH_WARNINGS
        else:
            overall_status = QCStatus.PASSED
        
        # Create report
        report = QCReport(
            job_id=job_id,
            tier=tier,
            status=overall_status,
            video_resolution=video_metadata.get("resolution", "unknown"),
            video_fps=int(video_metadata.get("fps", 0)),
            video_codec=video_metadata.get("codec", "unknown"),
            video_bitrate_kbps=int(video_metadata.get("bitrate_kbps", 0)),
            video_color_space=video_metadata.get("color_space", "unknown"),
            audio_format=video_metadata.get("audio_format", "unknown"),
            audio_channels=int(video_metadata.get("audio_channels", 0)),
            audio_codec=video_metadata.get("audio_codec", "unknown"),
            audio_loudness_lufs=float(video_metadata.get("audio_loudness_lufs", 0)),
            checks=checks,
            generated_at=datetime.utcnow()
        )
        
        if report.status == QCStatus.PASSED:
            report.passed_at = datetime.utcnow()
        
        logger.info(f"QC Report: {job_id} / {tier} → {report.status.value}")
        return report
    
    @classmethod
    def _check_video_resolution(cls, spec: str, actual: str) -> QCCheckResult:
        """Check video resolution matches spec"""
        passed = spec.lower() in actual.lower() or actual.lower() in spec.lower()
        return QCCheckResult(
            check_type=QCCheckType.VIDEO_RESOLUTION,
            specification=spec,
            actual_value=actual,
            passed=passed,
            severity="error" if not passed else "info",
            message="" if passed else "Resolution mismatch"
        )
    
    @classmethod
    def _check_video_fps(cls, spec: int, actual: float) -> QCCheckResult:
        """Check video FPS matches spec (with tolerance)"""
        tolerance = cls.TOLERANCES[QCCheckType.VIDEO_FPS]
        passed = abs(actual - spec) <= tolerance
        return QCCheckResult(
            check_type=QCCheckType.VIDEO_FPS,
            specification=f"{spec} fps",
            actual_value=f"{actual:.1f} fps",
            tolerance=f"±{tolerance} fps",
            passed=passed,
            severity="error" if not passed else "info"
        )
    
    @classmethod
    def _check_video_codec(cls, spec: str, actual: str) -> QCCheckResult:
        """Check video codec matches spec"""
        passed = spec.lower() in actual.lower() or actual.lower() in spec.lower()
        return QCCheckResult(
            check_type=QCCheckType.VIDEO_CODEC,
            specification=spec,
            actual_value=actual,
            passed=passed,
            severity="error" if not passed else "info"
        )
    
    @classmethod
    def _check_video_bitrate(cls, spec_kbps: int, actual_kbps: int) -> QCCheckResult:
        """Check video bitrate within tolerance"""
        tolerance_kbps = spec_kbps * cls.TOLERANCES[QCCheckType.VIDEO_BITRATE]
        passed = abs(actual_kbps - spec_kbps) <= tolerance_kbps
        return QCCheckResult(
            check_type=QCCheckType.VIDEO_BITRATE,
            specification=f"{spec_kbps} kbps",
            actual_value=f"{actual_kbps} kbps",
            tolerance=f"±{tolerance_kbps:.0f} kbps",
            passed=passed,
            severity="warning" if not passed else "info"
        )
    
    @classmethod
    def _check_color_space(cls, spec: str, actual: str) -> QCCheckResult:
        """Check color space specification"""
        passed = spec.lower() in actual.lower() or actual.lower() in spec.lower()
        return QCCheckResult(
            check_type=QCCheckType.VIDEO_CODEC,
            specification=spec,
            actual_value=actual,
            passed=passed,
            severity="warning" if not passed else "info"
        )
    
    @classmethod
    def _check_audio_format(cls, spec: str, actual: str) -> QCCheckResult:
        """Check audio format matches spec"""
        passed = spec.lower() in actual.lower() or actual.lower() in spec.lower()
        return QCCheckResult(
            check_type=QCCheckType.AUDIO_FORMAT,
            specification=spec,
            actual_value=actual,
            passed=passed,
            severity="error" if not passed else "info"
        )
    
    @classmethod
    def _check_audio_channels(cls, spec: int, actual: int) -> QCCheckResult:
        """Check audio channels match spec"""
        passed = spec == actual
        return QCCheckResult(
            check_type=QCCheckType.AUDIO_CHANNELS,
            specification=f"{spec} channels",
            actual_value=f"{actual} channels",
            passed=passed,
            severity="error" if not passed else "info"
        )
    
    @classmethod
    def _check_audio_codec(cls, spec: str, actual: str) -> QCCheckResult:
        """Check audio codec matches spec"""
        passed = spec.lower() in actual.lower() or actual.lower() in spec.lower()
        return QCCheckResult(
            check_type=QCCheckType.AUDIO_CODEC,
            specification=spec,
            actual_value=actual,
            passed=passed,
            severity="error" if not passed else "info"
        )
    
    @classmethod
    def _check_audio_loudness(cls, spec_lufs: float, actual_lufs: float) -> QCCheckResult:
        """Check audio loudness matches spec (with tolerance)"""
        tolerance = cls.TOLERANCES[QCCheckType.AUDIO_LOUDNESS]
        passed = abs(actual_lufs - spec_lufs) <= tolerance
        return QCCheckResult(
            check_type=QCCheckType.AUDIO_LOUDNESS,
            specification=f"{spec_lufs} LUFS",
            actual_value=f"{actual_lufs:.1f} LUFS",
            tolerance=f"±{tolerance} LUFS",
            passed=passed,
            severity="warning" if not passed else "info"
        )


if __name__ == "__main__":
    # Test QC report
    logger.info("Quality Assurance Engine Test")
    logger.info("=" * 60)
    
    # Simulate perfect video metadata
    metadata = {
        "resolution": "1920x1080",
        "fps": 24.0,
        "codec": "H.264",
        "bitrate_kbps": 3500,
        "color_space": "Rec.709 (SDR)",
        "audio_format": "Stereo",
        "audio_channels": 2,
        "audio_codec": "AAC-LC",
        "audio_loudness_lufs": -23.0,
        "artifact_score": 0.05,
        "flicker_detected": False
    }
    
    qa_engine = QualityAssuranceEngine()
    report = qa_engine.validate_video("job-123", "good", metadata)
    
    logger.info(f"\n✅ QC Report: {report.job_id}")
    logger.info(f"   Tier: {report.tier}")
    logger.info(f"   Status: {report.status.value}")
    logger.info(f"   Passed: {report.passed_checks}/{report.total_checks}")
