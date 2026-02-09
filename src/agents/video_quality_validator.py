"""
P1.1 - Video Quality Validator
Valide la qualité des vidéos générées

Métriques:
- Résolution (720p, 1080p, 2K, 4K)
- Codec Video (h264, h265, vp9)
- Frame Rate (24, 30, 60 fps)
- Durée totale
- File size
- Bitrate

Score: 0-100 (plus haut = meilleur)
"""

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoQualityTier(Enum):
    """Tiers de qualité vidéo"""
    GOOD = "good"  # 720p, h264, 24-30fps
    HIGH = "high"  # 1080p, h264/h265, 30fps
    ULTRA = "ultra"  # 2K-4K, h265, 60fps


@dataclass
class VideoQualityMetrics:
    """Métriques de qualité d'une vidéo"""
    resolution: str  # "1920x1080"
    width_px: int
    height_px: int
    codec: str  # "h264", "h265", "vp9"
    bitrate_kbps: int
    frame_rate: float  # 24, 30, 60
    duration_sec: float
    file_size_mb: float
    

    def to_dict(self) -> dict:
        return {
            "resolution": self.resolution,
            "dimensions": f"{self.width_px}x{self.height_px}",
            "codec": self.codec,
            "bitrate_kbps": self.bitrate_kbps,
            "fps": self.frame_rate,
            "duration_sec": self.duration_sec,
            "file_size_mb": self.file_size_mb,
        }


class VideoQualityValidator:
    """
    Valide la qualité des vidéos générées
    
    Cas d'usage:
    1. Parser metadata d'une vidéo
    2. Calculer un score de qualité 0-100
    3. Vérifier que la qualité atteint les standards
    """

    def __init__(self):
        """Initialize validator"""
        self.quality_thresholds = {
            "720p": {"width": 1280, "height": 720, "min_bitrate": 1000, "codec": ["h264", "vp9"]},
            "1080p": {"width": 1920, "height": 1080, "min_bitrate": 2500, "codec": ["h264", "h265"]},
            "2k": {"width": 2560, "height": 1440, "min_bitrate": 5000, "codec": ["h265", "vp9"]},
            "4k": {"width": 3840, "height": 2160, "min_bitrate": 8000, "codec": ["h265"]},
        }

    def extract_metrics(self, video_path: str) -> Optional[VideoQualityMetrics]:
        """
        Extrait les métriques d'une vidéo
        
        Args:
            video_path: Chemin vers le fichier vidéo
            
        Returns:
            VideoQualityMetrics ou None si erreur
        """
        try:
            import ffmpeg
            
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "video"), None
            )
            
            if not video_stream:
                logger.warning(f"No video stream found in {video_path}")
                return None
            
            # Extraire metadata
            width = video_stream.get("width", 0)
            height = video_stream.get("height", 0)
            codec = video_stream.get("codec_name", "unknown")
            bitrate = video_stream.get("bit_rate", 0)
            if bitrate:
                bitrate_kbps = int(bitrate) // 1000
            else:
                # Calculer depuis file size si bitrate pas disponible
                file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
                duration = float(probe["format"]["duration"])
                bitrate_kbps = int((file_size_mb * 8 * 1000) / duration)
            
            fps = eval(video_stream.get("r_frame_rate", "30/1"))
            duration = float(probe["format"]["duration"])
            file_size_mb = Path(video_path).stat().st_size / (1024 * 1024)
            
            resolution = f"{width}x{height}"
            
            return VideoQualityMetrics(
                resolution=resolution,
                width_px=width,
                height_px=height,
                codec=codec,
                bitrate_kbps=bitrate_kbps,
                frame_rate=fps,
                duration_sec=duration,
                file_size_mb=file_size_mb,
            )
            
        except Exception as e:
            logger.error(f"Error extracting metrics from {video_path}: {e}")
            return None

    def calculate_quality_score(self, metrics: VideoQualityMetrics) -> int:
        """
        Calcule score de qualité 0-100
        
        Critères:
        - Résolution (30 pts max)
        - Codec (20 pts max)
        - Bitrate (20 pts max)
        - FPS (15 pts max)
        - File size (15 pts max)
        
        Args:
            metrics: VideoQualityMetrics
            
        Returns:
            Score 0-100
        """
        score = 0
        resolution_score = 0
        codec_score = 0
        bitrate_score = 0
        fps_score = 0
        filesize_score = 0
        
        # === RÉSOLUTION (30 pts) ===
        pixels = metrics.width_px * metrics.height_px
        if pixels >= 3840 * 2160:  # 4K
            resolution_score = 30
        elif pixels >= 2560 * 1440:  # 2K
            resolution_score = 25
        elif pixels >= 1920 * 1080:  # 1080p
            resolution_score = 20
        elif pixels >= 1280 * 720:  # 720p
            resolution_score = 15
        else:
            resolution_score = 5
        
        # === CODEC (20 pts) ===
        if metrics.codec == "h265":
            codec_score = 20  # Modern, efficient
        elif metrics.codec == "h264":
            codec_score = 15  # Standard
        elif metrics.codec == "vp9":
            codec_score = 18  # Good
        else:
            codec_score = 5
        
        # === BITRATE (20 pts) ===
        # Optimal bitrate pour chaque résolution
        optimal_bitrate = {
            (1280, 720): 2000,
            (1920, 1080): 4000,
            (2560, 1440): 8000,
            (3840, 2160): 15000,
        }
        
        closest_res = min(
            optimal_bitrate.keys(),
            key=lambda r: abs(r[0] - metrics.width_px) + abs(r[1] - metrics.height_px)
        )
        optimal = optimal_bitrate.get(closest_res, 4000)
        
        if metrics.bitrate_kbps >= optimal:
            bitrate_score = 20
        elif metrics.bitrate_kbps >= optimal * 0.8:
            bitrate_score = 18
        elif metrics.bitrate_kbps >= optimal * 0.6:
            bitrate_score = 14
        elif metrics.bitrate_kbps >= optimal * 0.4:
            bitrate_score = 10
        else:
            bitrate_score = 5
        
        # === FPS (15 pts) ===
        if metrics.frame_rate >= 60:
            fps_score = 15
        elif metrics.frame_rate >= 30:
            fps_score = 12
        elif metrics.frame_rate >= 24:
            fps_score = 10
        else:
            fps_score = 5
        
        # === FILE SIZE (15 pts) ===
        # Ratio: file size vs duration (MB/second)
        size_ratio = metrics.file_size_mb / max(metrics.duration_sec, 1)
        if size_ratio >= 5:  # Good compression
            filesize_score = 15
        elif size_ratio >= 2:
            filesize_score = 12
        elif size_ratio >= 1:
            filesize_score = 10
        else:
            filesize_score = 5
        
        score = resolution_score + codec_score + bitrate_score + fps_score + filesize_score
        return min(100, score)

    def get_quality_tier(self, quality_score: int) -> VideoQualityTier:
        """
        Détermine le tier de qualité basé sur le score
        
        Args:
            quality_score: Score 0-100
            
        Returns:
            VideoQualityTier
        """
        if quality_score >= 75:
            return VideoQualityTier.ULTRA
        elif quality_score >= 50:
            return VideoQualityTier.HIGH
        else:
            return VideoQualityTier.GOOD

    def validate(self, video_path: str, min_score: int = 50) -> Dict[str, Any]:
        """
        Valide complètement une vidéo
        
        Args:
            video_path: Chemin vers le fichier vidéo
            min_score: Score minimum requis (0-100)
            
        Returns:
            {
                "valid": bool,
                "score": int (0-100),
                "tier": str (good/high/ultra),
                "metrics": {...},
                "issues": [list of problems],
                "recommendations": [list of improvements]
            }
        """
        metrics = self.extract_metrics(video_path)
        if not metrics:
            return {
                "valid": False,
                "score": 0,
                "tier": "unknown",
                "metrics": None,
                "issues": ["Could not extract video metadata"],
                "recommendations": [],
            }
        
        quality_score = self.calculate_quality_score(metrics)
        quality_tier = self.get_quality_tier(quality_score)
        
        issues = []
        recommendations = []
        
        # Check resolution
        pixels = metrics.width_px * metrics.height_px
        if pixels < 1280 * 720:
            issues.append(f"Resolution too low: {metrics.resolution} (recommend 720p+)")
            recommendations.append("Upscale video to at least 720p")
        
        # Check codec
        if metrics.codec not in ["h264", "h265", "vp9"]:
            issues.append(f"Codec not optimal: {metrics.codec}")
            recommendations.append("Re-encode with h265 or h264")
        
        # Check bitrate
        if metrics.bitrate_kbps < 1000:
            issues.append(f"Bitrate too low: {metrics.bitrate_kbps} kbps")
            recommendations.append("Increase bitrate for better quality")
        
        # Check FPS
        if metrics.frame_rate < 24:
            issues.append(f"Frame rate too low: {metrics.frame_rate} fps")
            recommendations.append("Generate at 30fps minimum")
        
        is_valid = quality_score >= min_score
        
        return {
            "valid": is_valid,
            "score": quality_score,
            "tier": quality_tier.value,
            "metrics": metrics.to_dict(),
            "issues": issues,
            "recommendations": recommendations,
        }
