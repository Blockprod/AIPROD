"""
Quality Specifications for AIPROD - Video Generation Standards

Defines guaranteed quality standards (specs) for each tier:
- GOOD: 1080p professional social media standard
- HIGH: 4K professional broadcast standard
- ULTRA: 4K@60fps cinematic HDR grade

Every generated video is certified against these specs.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class QualityTier(Enum):
    """Video quality tiers - QUALITY FIRST approach"""
    GOOD = "good"      # 1080p@24fps, stereo, auto WB
    HIGH = "high"      # 4K@30fps, 5.1 surround, professional grade
    ULTRA = "ultra"    # 4K@60fps, 7.1.4 Atmos, cinematic HDR


@dataclass
class VideoSpecification:
    """Video codec and resolution specification"""
    resolution: str         # "1920x1080", "3840x2160"
    fps: int                # 24, 30, 60
    codec: str              # "H.264", "H.265", "H.266"
    bitrate_kbps: int       # Video bitrate in kbps
    color_space: str        # "Rec.709 (SDR)", "Rec.2020 (HDR10)"
    
    def __str__(self) -> str:
        return f"{self.resolution}@{self.fps}fps, {self.codec}, {self.bitrate_kbps}kbps"


@dataclass
class AudioSpecification:
    """Audio format and processing specification"""
    format_name: str        # "Stereo", "5.1 Surround", "7.1.4 Atmos"
    channels: int           # 2, 6, 12
    codec: str              # "AAC-LC", "AC-3", "TrueHD"
    bitrate_kbps: int       # Audio bitrate
    sample_rate_khz: int    # 48
    
    # Audio processing pipeline
    loudness_lufs: float    # Target loudness in LUFS
    eq_bands: int           # Number of EQ bands used
    processing_steps: List[str] = None
    
    def __post_init__(self):
        if self.processing_steps is None:
            self.processing_steps = []
    
    def __str__(self) -> str:
        return f"{self.format_name}, {self.codec}, {self.bitrate_kbps}kbps"


@dataclass
class PostProductionSpecification:
    """Color grading, effects, and post-production standards"""
    color_grade_type: str   # "Auto WB", "3-point professional", "Cinematic DaVinci"
    effects: List[str]      # ["Transitions", "Motion blur", "Sharpness"]
    
    # Color grading details
    has_shadow_lift: bool
    has_midtone_curve: bool
    has_highlight_roll: bool
    shadow_range: str = None  # "-10 to +10%"
    midtone_range: str = None
    highlight_range: str = None
    
    # Beauty pass / quality enhancements
    has_beauty_pass: bool = False
    has_ai_upsampling: bool = False
    
    def __str__(self) -> str:
        return f"{self.color_grade_type} + {', '.join(self.effects) if self.effects else 'None'}"


@dataclass
class DeliverySpecification:
    """Output formats and delivery options"""
    output_formats: List[str]      # ["mp4", "webm", "mov"]
    estimated_delivery_time_sec: int
    estimated_file_size_mb: float


class GoodTierSpec:
    """
    GOOD Tier: Social Media Professional Standard
    
    Positioning: Modern social media standard (not "budget")
    Target: Content creators, small studios, social platforms
    """
    
    # Video specs
    video = VideoSpecification(
        resolution="1920x1080",
        fps=24,
        codec="H.264 (AVC)",
        bitrate_kbps=3500,
        color_space="Rec.709 (SDR)"
    )
    
    # Audio specs
    audio = AudioSpecification(
        format_name="Stereo (2.0)",
        channels=2,
        codec="AAC-LC",
        bitrate_kbps=128,
        sample_rate_khz=48,
        loudness_lufs=-23,
        eq_bands=1,
        processing_steps=[
            "Dialogue normalize (-23 LUFS)",
            "Basic limiter (-1dB ceiling)",
            "Fade in/out"
        ]
    )
    
    # Post-production specs
    postprod = PostProductionSpecification(
        color_grade_type="Automatic white balance correction",
        effects=[],
        has_shadow_lift=False,
        has_midtone_curve=False,
        has_highlight_roll=False,
        has_beauty_pass=False,
        has_ai_upsampling=False
    )
    
    # Delivery specs
    delivery = DeliverySpecification(
        output_formats=["mp4"],
        estimated_delivery_time_sec=35,
        estimated_file_size_mb=75  # 30 sec video approx
    )
    
    @classmethod
    def get_quality_guarantee(cls) -> str:
        return "Professional 1080p, conversation-clear dialogue, modern social media standard"
    
    @classmethod
    def get_sla(cls) -> str:
        return "Best-effort support (24h response)"
    
    @classmethod
    def get_use_cases(cls) -> List[str]:
        return ["TikTok", "Instagram Reels", "YouTube Shorts", "Web video", "Social content"]
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {
            "tier": "GOOD",
            "positioning": "Social Media Professional",
            "video": {
                "resolution": cls.video.resolution,
                "fps": cls.video.fps,
                "codec": cls.video.codec,
                "bitrate_kbps": cls.video.bitrate_kbps,
                "color_space": cls.video.color_space
            },
            "audio": {
                "format": cls.audio.format_name,
                "channels": cls.audio.channels,
                "codec": cls.audio.codec,
                "bitrate_kbps": cls.audio.bitrate_kbps,
                "loudness_lufs": cls.audio.loudness_lufs,
                "processing": cls.audio.processing_steps
            },
            "postprod": {
                "color_grading": cls.postprod.color_grade_type,
                "effects": cls.postprod.effects or "None"
            },
            "delivery_formats": cls.delivery.output_formats,
            "estimated_delivery_sec": cls.delivery.estimated_delivery_time_sec,
            "quality_guarantee": cls.get_quality_guarantee(),
            "sla": cls.get_sla()
        }


class HighTierSpec:
    """
    HIGH Tier: Professional 4K Broadcast Standard
    
    Positioning: Professional broadcast quality (Netflix/Vimeo ready)
    Target: Professional studios, streaming platforms, YouTube creators
    """
    
    # Video specs
    video = VideoSpecification(
        resolution="3840x2160",
        fps=30,
        codec="H.265 (HEVC)",
        bitrate_kbps=10000,
        color_space="Rec.709 (SDR)"
    )
    
    # Audio specs
    audio = AudioSpecification(
        format_name="5.1 Surround",
        channels=6,
        codec="AAC or AC-3",
        bitrate_kbps=320,
        sample_rate_khz=48,
        loudness_lufs=-23,
        eq_bands=3,
        processing_steps=[
            "Dialogue normalize (-23 LUFS)",
            "3-band EQ polish",
            "Surround object placement",
            "Dynamic range compression",
            "Fade in/out (smooth curves)"
        ]
    )
    
    # Post-production specs
    postprod = PostProductionSpecification(
        color_grade_type="3-point professional grade (shadows/mids/highlights)",
        effects=["Smooth transitions", "Motion blur", "Sharpness enhancement"],
        has_shadow_lift=True,
        has_midtone_curve=True,
        has_highlight_roll=True,
        shadow_range="-10 to +10% with rolloff",
        midtone_range="S-curve ±15%",
        highlight_range="-10% (protective, no clipping)",
        has_beauty_pass=False,
        has_ai_upsampling=False
    )
    
    # Delivery specs
    delivery = DeliverySpecification(
        output_formats=["mp4", "webm", "mov", "hls_adaptive"],
        estimated_delivery_time_sec=60,
        estimated_file_size_mb=250  # 30 sec 4K video approx
    )
    
    @classmethod
    def get_quality_guarantee(cls) -> str:
        return "Professional 4K broadcast quality with immersive surround audio and cinema-grade color"
    
    @classmethod
    def get_sla(cls) -> str:
        return "Standard support (24h response, business hours)"
    
    @classmethod
    def get_use_cases(cls) -> List[str]:
        return ["YouTube 4K", "Netflix", "Vimeo", "Professional content", "Broadcast television"]
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {
            "tier": "HIGH",
            "positioning": "Professional 4K Broadcast",
            "video": {
                "resolution": cls.video.resolution,
                "fps": cls.video.fps,
                "codec": cls.video.codec,
                "bitrate_kbps": cls.video.bitrate_kbps,
                "color_space": cls.video.color_space
            },
            "audio": {
                "format": cls.audio.format_name,
                "channels": cls.audio.channels,
                "codec": cls.audio.codec,
                "bitrate_kbps": cls.audio.bitrate_kbps,
                "loudness_lufs": cls.audio.loudness_lufs,
                "processing": cls.audio.processing_steps
            },
            "postprod": {
                "color_grading": cls.postprod.color_grade_type,
                "effects": cls.postprod.effects
            },
            "delivery_formats": cls.delivery.output_formats,
            "estimated_delivery_sec": cls.delivery.estimated_delivery_time_sec,
            "quality_guarantee": cls.get_quality_guarantee(),
            "sla": cls.get_sla()
        }


class UltraTierSpec:
    """
    ULTRA Tier: Cinematic HDR 4K@60fps Standard
    
    Positioning: Broadcast cinema quality with HDR mastery
    Target: Theatrical cinema, premium streaming, Hollywood studios, VR content
    """
    
    # Video specs
    video = VideoSpecification(
        resolution="3840x2160",
        fps=60,
        codec="H.266 (VVC) with HDR10",
        bitrate_kbps=30000,
        color_space="Rec.2020 (HDR10 @ 1000 nits)"
    )
    
    # Audio specs
    audio = AudioSpecification(
        format_name="7.1.4 Spatial (Dolby Atmos)",
        channels=12,
        codec="Dolby TrueHD (Atmos)",
        bitrate_kbps=768,
        sample_rate_khz=48,
        loudness_lufs=-24,  # Cinema standard
        eq_bands=5,
        processing_steps=[
            "Dialogue normalize (-24 LUFS cinema standard)",
            "5-band mastering EQ",
            "Full object audio placement (3D coordinates)",
            "Immersive effects spatialization",
            "Multiband compression",
            "Professional limiting (-0.1dB headroom)",
            "Cinema-grade fade curves",
            "Metadata: Loudness segmentation"
        ]
    )
    
    # Post-production specs
    postprod = PostProductionSpecification(
        color_grade_type="Cinematic HDR color grading (DaVinci workflow)",
        effects=["Cinematic transitions", "Advanced motion blur", "Sharpness enhancement"],
        has_shadow_lift=True,
        has_midtone_curve=True,
        has_highlight_roll=True,
        shadow_range="Graduated curve with rolloff",
        midtone_range="8-point custom curve (full creative range)",
        highlight_range="±15% (bright protection)",
        has_beauty_pass=True,  # Despeckle + noise reduction
        has_ai_upsampling=True  # Optional: 2x temporal coherence
    )
    
    # Delivery specs
    delivery = DeliverySpecification(
        output_formats=["mp4_hdr", "mkv_atmos", "prores", "dnxhr", "dcp"],
        estimated_delivery_time_sec=120,
        estimated_file_size_mb=800  # 30 sec 4K@60fps approx
    )
    
    @classmethod
    def get_quality_guarantee(cls) -> str:
        return "Broadcast cinema quality: 4K@60fps HDR, fully immersive 7.1.4 Atmos, professional DaVinci color"
    
    @classmethod
    def get_sla(cls) -> str:
        return "Priority support (4h response, 24/7 availability)"
    
    @classmethod
    def get_use_cases(cls) -> List[str]:
        return ["Theatrical cinema", "Premium streaming", "Sports broadcast", "VR content", "Studio work"]
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        return {
            "tier": "ULTRA",
            "positioning": "Cinematic HDR 4K@60fps",
            "video": {
                "resolution": cls.video.resolution,
                "fps": cls.video.fps,
                "codec": cls.video.codec,
                "bitrate_kbps": cls.video.bitrate_kbps,
                "color_space": cls.video.color_space
            },
            "audio": {
                "format": cls.audio.format_name,
                "channels": cls.audio.channels,
                "codec": cls.audio.codec,
                "bitrate_kbps": cls.audio.bitrate_kbps,
                "loudness_lufs": cls.audio.loudness_lufs,
                "processing": cls.audio.processing_steps
            },
            "postprod": {
                "color_grading": cls.postprod.color_grade_type,
                "effects": cls.postprod.effects,
                "beauty_pass": cls.postprod.has_beauty_pass,
                "ai_upsampling": cls.postprod.has_ai_upsampling
            },
            "delivery_formats": cls.delivery.output_formats,
            "estimated_delivery_sec": cls.delivery.estimated_delivery_time_sec,
            "quality_guarantee": cls.get_quality_guarantee(),
            "sla": cls.get_sla()
        }


class QualitySpecRegistry:
    """Registry of all quality tiers"""
    
    TIERS = {
        "good": GoodTierSpec,
        "high": HighTierSpec,
        "ultra": UltraTierSpec
    }
    
    @classmethod
    def get_tier_spec(cls, tier_name: str):
        """Get specification class for a tier"""
        tier_name = tier_name.lower().strip()
        if tier_name not in cls.TIERS:
            raise ValueError(f"Unknown tier: {tier_name}. Must be one of {list(cls.TIERS.keys())}")
        return cls.TIERS[tier_name]
    
    @classmethod
    def get_all_tiers(cls) -> List[Dict[str, Any]]:
        """Return all tier specifications as dicts"""
        return [spec.to_dict() for spec in cls.TIERS.values()]
    
    @classmethod
    def get_tier_details(cls, tier_name: str) -> Dict[str, Any]:
        """Get full details for a specific tier"""
        spec = cls.get_tier_spec(tier_name)
        return spec.to_dict()


if __name__ == "__main__":
    # Test: Print all tier specs
    logger.info("AIPROD Quality Specifications")
    logger.info("=" * 60)
    
    for tier_spec in QualitySpecRegistry.get_all_tiers():
        logger.info(f"\n📺 {tier_spec['tier']} - {tier_spec['positioning']}")
        logger.info(f"   Video: {tier_spec['video']['resolution']}@{tier_spec['video']['fps']}fps, {tier_spec['video']['codec']}")
        logger.info(f"   Audio: {tier_spec['audio']['format']}, {tier_spec['audio']['codec']}")
        logger.info(f"   Delivery: {', '.join(tier_spec['delivery_formats'])}")
        logger.info(f"   SLA: {tier_spec['sla']}")
