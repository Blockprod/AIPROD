"""
LPIPS (Learned Perceptual Image Patch Similarity) metric.

Measures perceptual image quality using learned features from a pre-trained network.
More aligned with human perception than pixel-level metrics like PSNR/SSIM.

Features:
- Per-frame perceptual distance computation
- Support for different backbone networks (AlexNet, VGG, SqueezeNet)
- Batch processing for efficiency
- Spatial pooling for frame-level and video-level scores

Typical ranges:
- 0.0-0.1: Very similar (excellent quality)
- 0.1-0.2: Quite similar (good quality)
- 0.2-0.5: Moderate difference (acceptable)
- 0.5+: Large difference (poor quality)
"""

from typing import Optional, Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LPIPSMetric:
    """LPIPS score for comparing two images/videos."""
    
    # Overall distance (lower = more similar)
    overall_distance: float
    
    # Per-frame distances
    per_frame_distances: torch.Tensor  # Shape: (num_frames,)
    
    # Min distance (best frame pair)
    min_distance: float
    
    # Max distance (worst frame pair)
    max_distance: float
    
    # Mean absolute difference across frames
    mean_distance: float
    
    # Standard deviation
    std_distance: float
    
    def __repr__(self) -> str:
        return (
            f"LPIPSMetric(overall={self.overall_distance:.4f}, "
            f"range=[{self.min_distance:.4f}, {self.max_distance:.4f}])"
        )


class LPIPSNet(nn.Module):
    """LPIPS network using learned perceptual metric.
    
    Extracts features from intermediate layers of a backbone network,
    computes patch-wise distances, and aggregates to image-level distance.
    """
    
    def __init__(
        self,
        net_type: str = "vgg",  # "vgg", "alex", "squeeze"
        normalize: bool = True,
    ):
        """
        Initialize LPIPS network.
        
        Args:
            net_type: Backbone network type
            normalize: Whether to normalize input to ImageNet statistics
        """
        super().__init__()
        self.net_type = net_type
        self.normalize = normalize
        
        if normalize:
            # ImageNet normalization
            self.register_buffer(
                "mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            )
        
        # Simplified backbone (in practice would load pre-trained weights)
        if net_type == "vgg":
            self.layers = self._build_vgg_features()
            self.layer_weights = [0.03, 0.1, 0.1, 0.2, 0.3]
        elif net_type == "alex":
            self.layers = self._build_alex_features()
            self.layer_weights = [0.1, 0.1, 0.1, 0.15, 0.35]
        else:  # squeeze
            self.layers = self._build_squeeze_features()
            self.layer_weights = [0.1, 0.1, 0.2, 0.3, 0.3]
    
    def _build_vgg_features(self) -> nn.ModuleList:
        """Build VGG feature extraction layers."""
        layers = nn.ModuleList()
        
        # Simplified VGG-like layer structure
        in_ch = 3
        out_channels = [64, 128, 256, 512, 512]
        
        for out_ch in out_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))
            in_ch = out_ch
        
        return layers
    
    def _build_alex_features(self) -> nn.ModuleList:
        """Build AlexNet-like feature extraction."""
        layers = nn.ModuleList()
        
        in_ch = 3
        configs = [(64, 11), (192, 5), (384, 3), (256, 3), (256, 3)]
        
        for out_ch, kernel in configs:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2, 2))
            in_ch = out_ch
        
        return layers
    
    def _build_squeeze_features(self) -> nn.ModuleList:
        """Build SqueezeNet-like feature extraction."""
        layers = nn.ModuleList()
        
        # Simplified squeeze module structure
        in_ch = 3
        out_channels = [64, 128, 256, 384, 512]
        
        for out_ch in out_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        
        return layers
    
    def forward(
        self,
        x: torch.Tensor,  # (batch, 3, height, width)
    ) -> List[torch.Tensor]:
        """
        Extract features at multiple layers.
        
        Args:
            x: Input images
            
        Returns:
            List of feature maps from different layers
        """
        if self.normalize:
            x = (x - self.mean) / self.std
        
        features = []
        
        # Extract features layer by layer
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.ReLU):
                if isinstance(layer, nn.ReLU):
                    features.append(x)
        
        return features


class LPIPSCalculator:
    """Computes LPIPS metric between image pairs.
    
    Handles:
    - Single image pair comparison
    - Frame-by-frame video comparison
    - Batch processing
    """
    
    def __init__(
        self,
        net_type: str = "vgg",
        device: str = "cuda",
        spatial_pooling: str = "mean",  # "mean", "max"
    ):
        """
        Initialize LPIPS calculator.
        
        Args:
            net_type: Backbone network type
            device: Device to compute on
            spatial_pooling: How to pool spatial dimensions
        """
        self.net = LPIPSNet(net_type=net_type, normalize=True)
        self.net.to(device)
        self.net.eval()
        self.device = device
        self.spatial_pooling = spatial_pooling
    
    def compute_distance(
        self,
        img1: torch.Tensor,  # (3, H, W)
        img2: torch.Tensor,  # (3, H, W)
    ) -> float:
        """
        Compute LPIPS distance between two images.
        
        Args:
            img1: First image
            img2: Second image
            
        Returns:
            LPIPS distance (lower = more similar)
        """
        # Batch dimension
        img1 = img1.unsqueeze(0).to(self.device)
        img2 = img2.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract features
            feats1 = self.net(img1)
            feats2 = self.net(img2)
            
            # Compute distances at each layer
            distances = []
            for f1, f2 in zip(feats1, feats2):
                # L2 distance between features
                diff = (f1 - f2) ** 2
                
                # Spatial pooling
                if self.spatial_pooling == "mean":
                    pooled = diff.mean(dim=[2, 3])
                else:  # max
                    pooled = diff.max(dim=2)[0].max(dim=2)[0]
                
                distances.append(pooled)
            
            # Weighted combination of layer distances
            layer_weights = torch.tensor(
                self.net.layer_weights,
                device=self.device,
            )
            combined = sum(
                w * d.mean() for w, d in zip(layer_weights, distances)
            )
        
        return combined.item()
    
    def compute_frame_distances(
        self,
        video1: torch.Tensor,  # (T, 3, H, W)
        video2: torch.Tensor,  # (T, 3, H, W)
    ) -> LPIPSMetric:
        """
        Compute per-frame LPIPS distances between two videos.
        
        Args:
            video1: First video
            video2: Second video
            
        Returns:
            LPIPSMetric with per-frame distances
        """
        num_frames = video1.shape[0]
        distances = []
        
        for t in range(num_frames):
            dist = self.compute_distance(video1[t], video2[t])
            distances.append(dist)
        
        distances = torch.tensor(distances)
        
        return LPIPSMetric(
            overall_distance=distances.mean().item(),
            per_frame_distances=distances,
            min_distance=distances.min().item(),
            max_distance=distances.max().item(),
            mean_distance=distances.mean().item(),
            std_distance=distances.std().item(),
        )
    
    def compute_batch_distances(
        self,
        video_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> List[LPIPSMetric]:
        """
        Compute LPIPS for batch of video pairs.
        
        Args:
            video_pairs: List of (video1, video2) tuples
            
        Returns:
            List of LPIPSMetric results
        """
        return [
            self.compute_frame_distances(v1, v2)
            for v1, v2 in video_pairs
        ]


def compute_lpips_batch_efficient(
    features1_batch: List[List[torch.Tensor]],  # per-video, per-layer features
    features2_batch: List[List[torch.Tensor]],
    layer_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Compute LPIPS distances from pre-computed features (most efficient).
    
    Args:
        features1_batch: Pre-computed features for video 1
        features2_batch: Pre-computed features for video 2
        layer_weights: Weights for each layer
        
    Returns:
        Per-frame LPIPS distances
    """
    distances = []
    
    for feats1, feats2 in zip(features1_batch, features2_batch):
        layer_distances = []
        
        for f1, f2 in zip(feats1, feats2):
            diff = (f1 - f2) ** 2
            pooled = diff.mean(dim=[2, 3])
            layer_distances.append(pooled)
        
        # Weighted combination
        combined = sum(
            w * d.mean() for w, d in zip(layer_weights, layer_distances)
        )
        distances.append(combined)
    
    return torch.stack(distances)
