"""Color Grading & Color Science Pipeline

Handles LUT application, color space conversions, HDR processing,
and automatic color matching across scenes.

Pipeline:
    input frame  → ColorSpaceConverter (to linear)
               → LUTManager.apply_lut (trilinear)
               → AutoGrader (AI curve)
               → SceneColorMatcher (histogram)
               → HDRProcessor (tone-map)
               → ColorSpaceConverter (to output)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────────

@dataclass
class ColorGradingConfig:
    """Color Grading Configuration."""
    input_color_space: str = "rec709"
    output_color_space: str = "rec709"
    lut_size: int = 33
    enable_hdr: bool = True
    hdr_format: str = "hdr10"     # hdr10 | dolby_vision | hlg
    peak_brightness: float = 1000.0  # nits
    enable_scene_matching: bool = True
    matching_strength: float = 0.7
    enable_auto_grade: bool = True
    style_preset: str = "cinematic"


@dataclass
class LUT:
    """3D Look-Up Table for color grading."""
    name: str
    data: torch.Tensor           # [S, S, S, 3]
    color_space: str = "rec709"
    description: str = ""


# ───────────────────────────────────────────────────────────────────────────
# Color-space conversion matrices  (3 × 3, linear light)
# ───────────────────────────────────────────────────────────────────────────

# Rec.709 → Rec.2020  (BT.2087 Annex 1)
_M_709_TO_2020 = torch.tensor([
    [0.6274, 0.3293, 0.0433],
    [0.0691, 0.9195, 0.0114],
    [0.0164, 0.0880, 0.8956],
], dtype=torch.float32)

# Rec.709 → DCI-P3
_M_709_TO_P3 = torch.tensor([
    [0.8225, 0.1774, 0.0000],
    [0.0332, 0.9669, 0.0000],
    [0.0171, 0.0724, 0.9108],
], dtype=torch.float32)

# Rec.2020 → Rec.709
_M_2020_TO_709 = torch.tensor([
    [1.6605, -0.5877, -0.0728],
    [-0.1246,  1.1330, -0.0084],
    [-0.0182, -0.1006,  1.1187],
], dtype=torch.float32)


class ColorSpaceConverter:
    """Converts between colour spaces using 3×3 matrix transforms."""

    @staticmethod
    def _apply_matrix(frame: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        """Apply 3×3 matrix to frame [..., 3]."""
        mat = mat.to(frame.device)
        return (frame @ mat.T).clamp(0.0, 1.0)

    @staticmethod
    def rec709_to_rec2020(frame: torch.Tensor) -> torch.Tensor:
        return ColorSpaceConverter._apply_matrix(frame, _M_709_TO_2020)

    @staticmethod
    def rec709_to_dci_p3(frame: torch.Tensor) -> torch.Tensor:
        return ColorSpaceConverter._apply_matrix(frame, _M_709_TO_P3)

    @staticmethod
    def rec2020_to_rec709(frame: torch.Tensor) -> torch.Tensor:
        return ColorSpaceConverter._apply_matrix(frame, _M_2020_TO_709)

    @staticmethod
    def linear_to_log(frame: torch.Tensor, cut: float = 0.010591) -> torch.Tensor:
        """ARRI LogC3 (EI 800) encoding."""
        a, b, c, d = 5.555556, 0.052272, 0.247190, 0.385537
        return torch.where(
            frame > cut,
            c * torch.log10(a * frame + b) + d,
            frame * 5.367655 + 0.092809,
        )

    @staticmethod
    def log_to_linear(frame: torch.Tensor, cut_log: float = 0.149658) -> torch.Tensor:
        """ARRI LogC3 (EI 800) decoding."""
        a, b, c, d = 5.555556, 0.052272, 0.247190, 0.385537
        return torch.where(
            frame > cut_log,
            (10.0 ** ((frame - d) / c) - b) / a,
            (frame - 0.092809) / 5.367655,
        )


# ───────────────────────────────────────────────────────────────────────────
# 3D LUT manager
# ───────────────────────────────────────────────────────────────────────────

class LUTManager:
    """Manages 3D LUT library and trilinear LUT application."""

    def __init__(self):
        self.luts: Dict[str, LUT] = {}
        self._init_builtin_luts()

    # ── Built-in procedural LUTs ──────────────────────────────────────

    def _init_builtin_luts(self) -> None:
        """Create a handful of procedural LUTs."""
        size = 33
        self._add_identity(size)
        self._add_warm_cinematic(size)
        self._add_cool_cinematic(size)

    def _add_identity(self, s: int) -> None:
        coords = torch.linspace(0, 1, s)
        r, g, b = torch.meshgrid(coords, coords, coords, indexing="ij")
        data = torch.stack([r, g, b], dim=-1)
        self.luts["identity"] = LUT("identity", data, description="Pass-through")

    def _add_warm_cinematic(self, s: int) -> None:
        base = self.luts.get("identity")
        if base is None:
            self._add_identity(s)
            base = self.luts["identity"]
        data = base.data.clone()
        data[..., 0] = (data[..., 0] * 1.08).clamp(0, 1)   # red lift
        data[..., 2] = (data[..., 2] * 0.90).clamp(0, 1)   # blue pull
        gamma = 0.95
        data = data.pow(gamma)
        self.luts["cinematic_warm"] = LUT("cinematic_warm", data, description="Warm cinema look")

    def _add_cool_cinematic(self, s: int) -> None:
        base = self.luts.get("identity")
        if base is None:
            self._add_identity(s)
            base = self.luts["identity"]
        data = base.data.clone()
        data[..., 0] = (data[..., 0] * 0.92).clamp(0, 1)
        data[..., 2] = (data[..., 2] * 1.06).clamp(0, 1)
        self.luts["cinematic_cool"] = LUT("cinematic_cool", data, description="Cool teal look")

    # ── 3D LUT application ────────────────────────────────────────────

    def apply_lut(self, frame: torch.Tensor, lut: LUT) -> torch.Tensor:
        """Apply 3D LUT via trilinear interpolation.

        Args:
            frame: [H, W, 3] RGB in [0, 1]
            lut:   LUT with data [S, S, S, 3]
        Returns:
            graded: [H, W, 3]
        """
        H, W, _ = frame.shape
        S = lut.data.shape[0]
        device = frame.device
        lut_data = lut.data.to(device)

        # Scale to LUT grid
        rgb = frame.clamp(0, 1) * (S - 1)
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        # Floor / ceil indices
        r0, g0, b0 = r.long().clamp(0, S - 2), g.long().clamp(0, S - 2), b.long().clamp(0, S - 2)
        r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1

        # Fractional parts
        fr, fg, fb = r - r0.float(), g - g0.float(), b - b0.float()
        fr = fr.unsqueeze(-1)
        fg = fg.unsqueeze(-1)
        fb = fb.unsqueeze(-1)

        # 8 corners of the trilinear cell
        c000 = lut_data[r0, g0, b0]
        c001 = lut_data[r0, g0, b1]
        c010 = lut_data[r0, g1, b0]
        c011 = lut_data[r0, g1, b1]
        c100 = lut_data[r1, g0, b0]
        c101 = lut_data[r1, g0, b1]
        c110 = lut_data[r1, g1, b0]
        c111 = lut_data[r1, g1, b1]

        # Trilinear interpolation
        c00 = c000 * (1 - fb) + c001 * fb
        c01 = c010 * (1 - fb) + c011 * fb
        c10 = c100 * (1 - fb) + c101 * fb
        c11 = c110 * (1 - fb) + c111 * fb
        c0 = c00 * (1 - fg) + c01 * fg
        c1 = c10 * (1 - fg) + c11 * fg
        result = c0 * (1 - fr) + c1 * fr

        return result.clamp(0, 1)

    # ── .cube file loader ─────────────────────────────────────────────

    def load_lut_file(self, filepath: str, lut_name: str) -> None:
        """Parse an Adobe .cube 3D LUT file."""
        size = 0
        values: list = []

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("TITLE"):
                    continue
                if line.startswith("LUT_3D_SIZE"):
                    size = int(line.split()[-1])
                    continue
                if line.startswith("DOMAIN_MIN") or line.startswith("DOMAIN_MAX"):
                    continue
                parts = line.split()
                if len(parts) == 3:
                    values.append([float(x) for x in parts])

        if size == 0:
            raise ValueError("Could not determine LUT size from .cube file")

        data = torch.tensor(values, dtype=torch.float32).reshape(size, size, size, 3)
        self.luts[lut_name] = LUT(lut_name, data, description=f"Loaded from {filepath}")


# ───────────────────────────────────────────────────────────────────────────
# HDR processor
# ───────────────────────────────────────────────────────────────────────────

class HDRProcessor(nn.Module):
    """HDR tone mapping and metadata handling."""

    def __init__(self, config: ColorGradingConfig):
        super().__init__()
        self.config = config

    def tone_map_sdr_to_hdr(
        self,
        sdr_video: torch.Tensor,       # [..., 3]  range [0, 1]
        peak_brightness: float = 1000.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Inverse tone-map SDR → HDR using Hable filmic curve.

        Returns (hdr_video, metadata).
        """
        # Hable / Uncharted-2 filmic curve (inverse for expansion)
        def hable(x: torch.Tensor) -> torch.Tensor:
            A, B, C, D, E, F_ = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
            return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F_)) - E / F_

        W = peak_brightness / 80.0  # normalise to SDR white (80 nits)
        white_scale = 1.0 / hable(torch.tensor(W))

        linear = sdr_video.clamp(0, 1)
        expanded = hable(linear * W) * white_scale
        hdr = expanded * peak_brightness

        metadata = self.add_hdr_metadata(hdr, peak_brightness)
        return hdr, metadata

    def add_hdr_metadata(
        self, video: torch.Tensor, peak: float = 1000.0,
    ) -> dict:
        """Create SMPTE 2086 / CTA-861 HDR10 metadata."""
        return {
            "format": self.config.hdr_format,
            "max_cll": round(video.max().item(), 1),
            "max_fall": round(video.mean().item(), 1),
            "peak_luminance": peak,
            "min_luminance": 0.005,
            "primaries": "rec2020",
            "white_point": "D65",
            "transfer": "pq" if self.config.hdr_format == "hdr10" else "hlg",
        }


# ───────────────────────────────────────────────────────────────────────────
# Auto Grader (learnable)
# ───────────────────────────────────────────────────────────────────────────

class AutoGrader(nn.Module):
    """Lightweight CNN that predicts per-pixel colour adjustments."""

    def __init__(self, config: ColorGradingConfig):
        super().__init__()
        self.config = config
        # Small U-Net-like net: 3→16→32→16→3  (residual)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(16, 3, 1)

    def grade(self, frame: torch.Tensor, style: str = "cinematic") -> torch.Tensor:
        """Auto-grade a single frame.

        Args:
            frame: [H, W, 3]  (0–1)
        Returns:
            graded: [H, W, 3]
        """
        x = frame.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        f1 = self.enc1(x)
        f2 = self.enc2(f1)
        d1 = self.dec1(f2) + f1          # skip
        adj = self.head(d1)              # [1, 3, H, W]  residual
        out = (x + adj).clamp(0, 1)
        return out.squeeze(0).permute(1, 2, 0)  # [H, W, 3]


# ───────────────────────────────────────────────────────────────────────────
# Scene Color Matcher
# ───────────────────────────────────────────────────────────────────────────

class SceneColorMatcher:
    """Automatic colour matching across scenes for consistency."""

    @staticmethod
    def compute_color_histogram(
        frame: torch.Tensor, bins: int = 256,
    ) -> torch.Tensor:
        """Compute per-channel histogram.

        Args:
            frame: [H, W, 3]  (0–1)
        Returns:
            hist: [3, bins]
        """
        hist = torch.zeros(3, bins, device=frame.device)
        for c in range(3):
            channel = frame[..., c].reshape(-1)
            idx = (channel * (bins - 1)).long().clamp(0, bins - 1)
            hist[c].scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        # Normalise
        hist = hist / hist.sum(dim=1, keepdim=True).clamp(min=1)
        return hist

    @staticmethod
    def match_histograms(
        src_frame: torch.Tensor, tgt_frame: torch.Tensor, bins: int = 256,
    ) -> torch.Tensor:
        """Match colour distribution of *src* to *tgt* via histogram matching.

        Args:
            src_frame, tgt_frame: [H, W, 3]
        Returns:
            matched: [H, W, 3]
        """
        matched = src_frame.clone()
        for c in range(3):
            src_ch = src_frame[..., c].reshape(-1)
            tgt_ch = tgt_frame[..., c].reshape(-1)

            # Build CDFs
            src_idx = (src_ch * (bins - 1)).long().clamp(0, bins - 1)
            tgt_idx = (tgt_ch * (bins - 1)).long().clamp(0, bins - 1)

            src_hist = torch.zeros(bins, device=src_ch.device)
            tgt_hist = torch.zeros(bins, device=tgt_ch.device)
            src_hist.scatter_add_(0, src_idx, torch.ones_like(src_idx, dtype=torch.float32))
            tgt_hist.scatter_add_(0, tgt_idx, torch.ones_like(tgt_idx, dtype=torch.float32))

            src_cdf = src_hist.cumsum(0)
            tgt_cdf = tgt_hist.cumsum(0)
            src_cdf = src_cdf / src_cdf[-1].clamp(min=1)
            tgt_cdf = tgt_cdf / tgt_cdf[-1].clamp(min=1)

            # Build mapping: for each src bin find closest tgt bin by CDF
            mapping = torch.zeros(bins, dtype=torch.long, device=src_ch.device)
            for s in range(bins):
                diff = (tgt_cdf - src_cdf[s]).abs()
                mapping[s] = diff.argmin()

            # Apply mapping
            new_vals = mapping[src_idx].float() / (bins - 1)
            matched[..., c] = new_vals.reshape(src_frame.shape[0], src_frame.shape[1])

        return matched.clamp(0, 1)

    @staticmethod
    def match_across_scene_sequence(
        frames_by_scene: List[torch.Tensor], strength: float = 0.7,
    ) -> List[torch.Tensor]:
        """Match colours consistently across multiple scenes.

        Uses the first scene as the reference and progressively matches
        subsequent scenes, blended with ``strength``.
        """
        if len(frames_by_scene) < 2:
            return frames_by_scene
        ref = frames_by_scene[0]
        result = [ref]
        for frame in frames_by_scene[1:]:
            matched = SceneColorMatcher.match_histograms(frame, ref)
            blended = (1.0 - strength) * frame + strength * matched
            result.append(blended.clamp(0, 1))
        return result


# ───────────────────────────────────────────────────────────────────────────
# Main pipeline
# ───────────────────────────────────────────────────────────────────────────

class ColorPipeline(nn.Module):
    """Main colour-grading pipeline orchestrating all sub-modules."""

    def __init__(self, config: ColorGradingConfig):
        super().__init__()
        self.config = config
        self.lut_manager = LUTManager()
        self.color_space_converter = ColorSpaceConverter()
        self.hdr_processor = HDRProcessor(config)
        self.auto_grader = AutoGrader(config)

    # ── Frame-level grading ────────────────────────────────────────────

    def grade_frame(
        self,
        frame: torch.Tensor,           # [H, W, 3]  (0–1)
        lut: Optional[LUT] = None,
        style: Optional[str] = None,
    ) -> torch.Tensor:
        """Apply full grading chain to a single frame."""
        # 1. Input colour-space → linear
        if self.config.input_color_space != "rec709":
            frame = self.color_space_converter.rec2020_to_rec709(frame)

        # 2. LUT
        if lut is not None:
            frame = self.lut_manager.apply_lut(frame, lut)

        # 3. Auto-grade
        if self.config.enable_auto_grade:
            frame = self.auto_grader.grade(frame, style or self.config.style_preset)

        # 4. Output colour-space
        if self.config.output_color_space == "rec2020":
            frame = self.color_space_converter.rec709_to_rec2020(frame)
        elif self.config.output_color_space == "dci_p3":
            frame = self.color_space_converter.rec709_to_dci_p3(frame)

        return frame

    # ── Video-level grading ────────────────────────────────────────────

    def grade_video(
        self,
        video: torch.Tensor,           # [B, 3, H, W, T]
        lut_name: Optional[str] = None,
        style: Optional[str] = None,
    ) -> torch.Tensor:
        """Grade every frame of a video tensor."""
        B, C, H, W, T = video.shape
        lut = self.lut_manager.luts.get(lut_name) if lut_name else None

        # Process frame-by-frame
        frames: List[torch.Tensor] = []
        for t_idx in range(T):
            frame = video[:, :, :, :, t_idx]       # [B, 3, H, W]
            frame = frame[0].permute(1, 2, 0)      # [H, W, 3]  — first batch
            graded = self.grade_frame(frame, lut=lut, style=style)
            frames.append(graded)

        # Scene matching
        if self.config.enable_scene_matching and len(frames) > 1:
            frames = SceneColorMatcher.match_across_scene_sequence(
                frames, strength=self.config.matching_strength,
            )

        # Reassemble [B, 3, H, W, T]
        stack = torch.stack(frames, dim=-1)       # [H, W, 3, T]
        stack = stack.permute(2, 0, 1, 3)         # [3, H, W, T]
        return stack.unsqueeze(0).expand(B, -1, -1, -1, -1)
