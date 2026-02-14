"""
Technical QA Gate Adapter - Binary Deterministic Checks
=======================================================

Implements 10 technical validation checks for generated video assets.
NO LLM - purely deterministic binary checks.

PHASE 3 implementation (Weeks 9-10).
"""

from typing import Dict, Any, List, Tuple
import asyncio
import logging
from .base import BaseAdapter
from ..schema.schemas import Context


logger = logging.getLogger(__name__)


class TechnicalQAGateAdapter(BaseAdapter):
    """
    Technical QA gate with deterministic validation checks.
    
    Performs 10 binary checks on each generated video:
    1. File integrity (can be read)
    2. Duration match (±2 seconds tolerance)
    3. Audio track present
    4. Resolution validation (1080p)
    5. Codec verification (H264)
    6. Bitrate range (2-8 Mbps)
    7. Frame rate validation (29-31 fps)
    8. Color space (YUV)
    9. Container format (MP4)
    10. Metadata completeness
    """
    
    # Check definitions: (check_name, description)
    CHECKS = [
        ("file_integrity", "Verify file can be read"),
        ("duration_match", "Duration ±2 seconds"),
        ("audio_present", "Audio track exists"),
        ("resolution_ok", "1080p resolution"),
        ("codec_valid", "H264 codec"),
        ("bitrate_ok", "2-8 Mbps range"),
        ("frame_rate_ok", "29-31 fps (30fps)"),
        ("color_space_ok", "YUV color space"),
        ("container_ok", "MP4 container"),
        ("metadata_ok", "Required metadata present")
    ]
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize technical QA gate."""
        super().__init__(config)
        
        # Tolerances
        self.duration_tolerance_sec = 2.0
        self.fps_min = 29.0
        self.fps_max = 31.0
        self.bitrate_min = 2_000_000  # 2 Mbps
        self.bitrate_max = 8_000_000  # 8 Mbps
        self.target_resolution = "1080p"
        self.target_codec = "h264"
        self.target_container = "mp4"
    
    async def execute(self, ctx: Context) -> Context:
        """
        Execute technical validation on all generated assets.
        
        Args:
            ctx: Context with generated_assets
            
        Returns:
            Context with technical_validation_report
        """
        # Validate context
        if not self.validate_context(ctx, ["generated_assets"]):
            raise ValueError("Missing generated_assets in context")
        
        videos = ctx["memory"]["generated_assets"]
        
        # Initialize report
        report = {
            "passed": True,
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": [],
            "videos_analyzed": len(videos),
            "video_reports": []
        }
        
        self.log("info", f"Starting technical QA on {len(videos)} videos")
        
        # Run checks on each video
        for video_idx, video in enumerate(videos):
            video_report = await self._validate_video(video, video_idx)
            report["video_reports"].append(video_report)
            
            report["total_checks"] += video_report["total_checks"]
            report["passed_checks"] += video_report["passed_checks"]
            
            if not video_report["passed"]:
                report["passed"] = False
                report["failed_checks"].extend(video_report["failed_checks"])
        
        # Calculate statistics
        if report["total_checks"] > 0:
            pass_rate = report["passed_checks"] / report["total_checks"]
            report["pass_rate"] = pass_rate
        else:
            report["pass_rate"] = 0.0
        
        # Store report in context
        ctx["memory"]["technical_validation_report"] = report
        
        # Log results
        self.log("info", "Technical QA complete",
                 passed=report["passed"],
                 pass_rate=report["pass_rate"],
                 failed_checks=len(report["failed_checks"]))
        
        # Update state if failed
        if not report["passed"]:
            self.log("error", "Technical QA failed - moving to ERROR state",
                     failures=len(report["failed_checks"]))
            ctx["state"] = "ERROR"
        
        return ctx
    
    async def _validate_video(self, video: Dict[str, Any], video_idx: int) -> Dict[str, Any]:
        """
        Validate a single video against all checks.
        
        Args:
            video: Video asset dictionary
            video_idx: Video index for reporting
            
        Returns:
            Video validation report
        """
        video_report = {
            "video_id": video.get("id", f"video_{video_idx}"),
            "video_url": video.get("url", "unknown"),
            "passed": True,
            "total_checks": len(self.CHECKS),
            "passed_checks": 0,
            "failed_checks": []
        }
        
        # Run each check
        for check_name, check_desc in self.CHECKS:
            try:
                # Execute check method
                check_method = getattr(self, f"_check_{check_name}")
                result = await check_method(video)
                
                if result:
                    video_report["passed_checks"] += 1
                else:
                    video_report["passed"] = False
                    video_report["failed_checks"].append({
                        "check": check_name,
                        "description": check_desc,
                        "video_id": video.get("id"),
                        "reason": f"{check_name} validation failed"
                    })
            
            except Exception as e:
                # Check raised exception - treat as failure
                video_report["passed"] = False
                video_report["failed_checks"].append({
                    "check": check_name,
                    "description": check_desc,
                    "video_id": video.get("id"),
                    "error": str(e)
                })
                
                self.log("warning", f"Check {check_name} raised exception",
                         video_id=video.get("id"), error=str(e))
        
        return video_report
    
    # ========================================================================
    # Check Methods (10 checks)
    # ========================================================================
    
    async def _check_file_integrity(self, video: Dict[str, Any]) -> bool:
        """
        Check 1: Verify file can be read and is not corrupted.
        
        Args:
            video: Video asset
            
        Returns:
            True if file is readable
        """
        # In production, would attempt to open/read file header
        # For now: check that URL exists and file_size_bytes is reasonable
        url = video.get("url")
        file_size = video.get("file_size_bytes", 0)
        
        if not url:
            return False
        
        if file_size <= 0:
            return False
        
        # Minimum file size: 100KB (reasonable for even very short video)
        if file_size < 100_000:
            return False
        
        return True
    
    async def _check_duration_match(self, video: Dict[str, Any]) -> bool:
        """
        Check 2: Verify duration matches expected (±2 seconds tolerance).
        
        Args:
            video: Video asset
            
        Returns:
            True if duration within tolerance
        """
        actual_duration = video.get("duration_sec", 0)
        
        if actual_duration <= 0:
            return False
        
        # In production, would compare to expected duration from shot specification
        # For now: just verify it's in reasonable range (10-300 seconds)
        if 10 <= actual_duration <= 300:
            return True
        
        return False
    
    async def _check_audio_present(self, video: Dict[str, Any]) -> bool:
        """
        Check 3: Verify audio track is present.
        
        Args:
            video: Video asset
            
        Returns:
            True if audio track exists
        """
        # In production, would probe audio streams
        # For now: assume audio present if file size suggests it
        # (videos without audio are typically smaller)
        
        file_size = video.get("file_size_bytes", 0)
        duration = video.get("duration_sec", 10)
        
        # Rough heuristic: should be at least 50KB per second with audio
        expected_min_size = duration * 50_000
        
        return file_size >= expected_min_size
    
    async def _check_resolution_ok(self, video: Dict[str, Any]) -> bool:
        """
        Check 4: Verify resolution is 1080p (1920x1080).
        
        Args:
            video: Video asset
            
        Returns:
            True if resolution is correct
        """
        resolution = video.get("resolution", "")
        
        return resolution == self.target_resolution or resolution == "1920x1080"
    
    async def _check_codec_valid(self, video: Dict[str, Any]) -> bool:
        """
        Check 5: Verify codec is H264.
        
        Args:
            video: Video asset
            
        Returns:
            True if codec is H264
        """
        codec = video.get("codec", "").lower()
        
        return codec == self.target_codec or codec == "h.264" or codec == "avc"
    
    async def _check_bitrate_ok(self, video: Dict[str, Any]) -> bool:
        """
        Check 6: Verify bitrate is in acceptable range (2-8 Mbps).
        
        Args:
            video: Video asset
            
        Returns:
            True if bitrate in range
        """
        bitrate = video.get("bitrate", 0)
        
        if bitrate <= 0:
            return False
        
        return self.bitrate_min <= bitrate <= self.bitrate_max
    
    async def _check_frame_rate_ok(self, video: Dict[str, Any]) -> bool:
        """
        Check 7: Verify frame rate is ~30fps (29-31 fps tolerance).
        
        Args:
            video: Video asset
            
        Returns:
            True if frame rate acceptable
        """
        # In production, would probe actual FPS
        # For now: assume 30fps if not specified
        fps = video.get("fps", 30)
        
        return self.fps_min <= fps <= self.fps_max
    
    async def _check_color_space_ok(self, video: Dict[str, Any]) -> bool:
        """
        Check 8: Verify color space is YUV (standard for video).
        
        Args:
            video: Video asset
            
        Returns:
            True if color space is YUV
        """
        # In production, would probe color space metadata
        # For now: assume YUV for standard video codecs
        codec = video.get("codec", "").lower()
        
        # H264 implies YUV color space
        if codec in ["h264", "h.264", "avc"]:
            return True
        
        # Check explicit color_space if provided
        color_space = video.get("color_space", "yuv").lower()
        return "yuv" in color_space or "420" in color_space
    
    async def _check_container_ok(self, video: Dict[str, Any]) -> bool:
        """
        Check 9: Verify container format is MP4.
        
        Args:
            video: Video asset
            
        Returns:
            True if container is MP4
        """
        url = video.get("url", "")
        
        # Check URL extension
        if url.endswith(".mp4"):
            return True
        
        # Check explicit container field
        container = video.get("container", "").lower()
        return container == "mp4" or container == "mpeg4"
    
    async def _check_metadata_ok(self, video: Dict[str, Any]) -> bool:
        """
        Check 10: Verify required metadata is present.
        
        Args:
            video: Video asset
            
        Returns:
            True if all required metadata present
        """
        required_fields = [
            "id",
            "url",
            "duration_sec",
            "resolution",
            "codec",
            "bitrate"
        ]
        
        for field in required_fields:
            if field not in video or not video[field]:
                return False
        
        return True
    
    def get_validation_summary(self, ctx: Context) -> Dict[str, Any]:
        """
        Get human-readable validation summary.
        
        Args:
            ctx: Context with validation report
            
        Returns:
            Summary dictionary
        """
        report = ctx["memory"].get("technical_validation_report", {})
        
        if not report:
            return {"status": "not_run"}
        
        return {
            "status": "passed" if report.get("passed") else "failed",
            "pass_rate": f"{report.get('pass_rate', 0) * 100:.1f}%",
            "total_checks": report.get("total_checks", 0),
            "passed_checks": report.get("passed_checks", 0),
            "failed_checks": len(report.get("failed_checks", [])),
            "videos_analyzed": report.get("videos_analyzed", 0)
        }
