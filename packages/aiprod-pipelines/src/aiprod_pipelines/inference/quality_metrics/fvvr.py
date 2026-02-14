"""
FVVR (Frame Visual Relevance Ratio) metric.

Measures how well video frames match the text prompt using CLIP embeddings.
Higher FVVR (0-1 scale) indicates better alignment with prompt semantics.

Formula:
    FVVR = mean(cosine_similarity(frame_embeddings, prompt_embedding))
    
Per-frame relevance enables:
- Identifying frames that diverge from prompt
- Detecting generation failures per frame
- Guiding adaptive guidance control
- Quality monitoring during inference

Typical scores:
- Excellent (0.85+): Strong prompt alignment
- Good (0.75-0.85): Acceptable alignment with minor artifacts
- Fair (0.65-0.75): Noticeable divergence
- Poor (<0.65): Major prompt misalignment
"""

from typing import Tuple, Optional, Dict, List, Any
import torch
import torch.nn.functional as F
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FVVRMetric:
    """FVVR score for a single video."""
    
    # Overall video FVVR (mean of all frames)
    overall_score: float
    
    # Per-frame scores (one per frame)
    per_frame_scores: torch.Tensor  # Shape: (num_frames,)
    
    # Frame-level statistics
    min_score: float
    max_score: float
    std_dev: float
    
    # Interpretation
    quality_grade: str  # "excellent", "good", "fair", "poor"
    
    def __repr__(self) -> str:
        return (
            f"FVVRMetric(overall={self.overall_score:.3f}, "
            f"grade={self.quality_grade}, "
            f"range=[{self.min_score:.3f}, {self.max_score:.3f}])"
        )


class FVVRCalculator:
    """Computes FVVR metric using CLIP embeddings.
    
    Requires pre-computed embeddings from CLIP vision encoder and text encoder.
    Can operate on latent space (more efficient) or pixel space (more accurate).
    """
    
    def __init__(
        self,
        text_encoder,
        image_encoder,
        embed_dim: int = 768,
        device: str = "cuda",
    ):
        """
        Initialize FVVR calculator.
        
        Args:
            text_encoder: Text encoder for prompt embeddings (e.g., CLIP text encoder)
            image_encoder: Image encoder for frame embeddings (e.g., CLIP image encoder)
            embed_dim: Embedding dimension (usually 768 for CLIP)
            device: Device to compute on ("cuda" or "cpu")
        """
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.embed_dim = embed_dim
        self.device = device
    
    def compute_prompt_embedding(
        self,
        prompt: str,
    ) -> torch.Tensor:
        """
        Compute text embedding for prompt.
        
        Args:
            prompt: Text prompt
            
        Returns:
            Normalized text embedding (embed_dim,)
        """
        # Tokenize and encode
        tokens = self.text_encoder.tokenize([prompt]).to(self.device)
        
        with torch.no_grad():
            embedding = self.text_encoder.encode_text(tokens)
        
        # Normalize
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding.squeeze(0)
    
    def compute_frame_embeddings(
        self,
        frames: torch.Tensor,  # (batch, channels, height, width)
    ) -> torch.Tensor:
        """
        Compute image embeddings for frames.
        
        Args:
            frames: Video frames (B, C, H, W) normalized to [0,1]
            
        Returns:
            Frame embeddings (B, embed_dim), normalized
        """
        # Ensure frames are on correct device
        frames = frames.to(self.device)
        
        with torch.no_grad():
            embeddings = self.image_encoder.encode_image(frames)
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def compute_fvvr(
        self,
        video_frames: torch.Tensor,  # (num_frames, channels, height, width)
        prompt: str,
    ) -> FVVRMetric:
        """
        Compute FVVR metric for entire video.
        
        Args:
            video_frames: Video frames (T, C, H, W)
            prompt: Text prompt
            
        Returns:
            FVVRMetric with overall + per-frame scores
        """
        # Get prompt embedding (single, used for all frames)
        prompt_emb = self.compute_prompt_embedding(prompt)  # (embed_dim,)
        
        # Get frame embeddings
        frame_embeds = self.compute_frame_embeddings(video_frames)  # (T, embed_dim)
        
        # Compute cosine similarity per frame
        # prompt_emb: (embed_dim,) -> (1, embed_dim)
        # frame_embeds: (T, embed_dim)
        # result: (T,)
        similarities = torch.nn.functional.cosine_similarity(
            prompt_emb.unsqueeze(0),
            frame_embeds,
            dim=1,
        )
        
        # Convert to [0,1] range (cosine similarity is in [-1,1])
        per_frame_scores = (similarities + 1) / 2
        
        overall_score = per_frame_scores.mean().item()
        
        # Compute statistics
        min_score = per_frame_scores.min().item()
        max_score = per_frame_scores.max().item()
        std_dev = per_frame_scores.std().item()
        
        # Grade assignment
        if overall_score >= 0.85:
            grade = "excellent"
        elif overall_score >= 0.75:
            grade = "good"
        elif overall_score >= 0.65:
            grade = "fair"
        else:
            grade = "poor"
        
        logger.info(
            f"FVVR computed: {overall_score:.3f} ({grade}) "
            f"[{min_score:.3f}, {max_score:.3f}] ± {std_dev:.3f}"
        )
        
        return FVVRMetric(
            overall_score=overall_score,
            per_frame_scores=per_frame_scores.cpu(),
            min_score=min_score,
            max_score=max_score,
            std_dev=std_dev,
            quality_grade=grade,
        )
    
    def compute_fvvr_batch(
        self,
        video_batch: List[torch.Tensor],  # List[T, C, H, W]
        prompts: List[str],
    ) -> List[FVVRMetric]:
        """
        Compute FVVR for batch of videos.
        
        Args:
            video_batch: List of video tensors
            prompts: List of prompts (one per video)
            
        Returns:
            List of FVVRMetric results
        """
        return [
            self.compute_fvvr(frames, prompt)
            for frames, prompt in zip(video_batch, prompts)
        ]


class FVVRTracker:
    """Tracks FVVR scores during inference for monitoring.
    
    Useful for:
    - Detecting when generation diverges
    - Early exit when quality converges
    - Adaptive guidance adjustment
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize tracker.
        
        Args:
            window_size: Rolling window for trend detection
        """
        self.window_size = window_size
        self.scores: List[float] = []
        self.history: Dict[int, FVVRMetric] = {}  # step -> metric
    
    def add_score(self, step: int, metric: FVVRMetric):
        """Add FVVR measurement.
        
        Args:
            step: Inference step number
            metric: FVVRMetric result
        """
        self.scores.append(metric.overall_score)
        self.history[step] = metric
    
    def get_trend(self) -> str:
        """Get trend in recent scores.
        
        Returns:
            "improving", "stable", or "declining"
        """
        if len(self.scores) < self.window_size:
            return "unknown"
        
        recent = self.scores[-self.window_size:]
        first_half_mean = sum(recent[:self.window_size//2]) / (self.window_size // 2)
        second_half_mean = sum(recent[self.window_size//2:]) / (len(recent) - self.window_size // 2)
        
        diff = second_half_mean - first_half_mean
        
        if diff > 0.02:
            return "improving"
        elif diff < -0.02:
            return "declining"
        else:
            return "stable"
    
    def should_early_exit(self, threshold: float = 0.85) -> bool:
        """Check if quality has converged and can exit early.
        
        Args:
            threshold: FVVR threshold for early exit
            
        Returns:
            True if last 3 steps above threshold
        """
        if len(self.scores) < 3:
            return False
        
        recent_three = self.scores[-3:]
        return all(score >= threshold for score in recent_three)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get overall statistics.
        
        Returns:
            Dictionary with min, max, mean, std of scores
        """
        if not self.scores:
            return {}
        
        scores_tensor = torch.tensor(self.scores)
        
        return {
            "min": scores_tensor.min().item(),
            "max": scores_tensor.max().item(),
            "mean": scores_tensor.mean().item(),
            "std": scores_tensor.std().item(),
        }


def compute_fvvr_efficient(
    latent_embeddings: torch.Tensor,  # (batch, embed_dim)
    prompt_embedding: torch.Tensor,   # (embed_dim,)
) -> torch.Tensor:
    """
    Compute FVVR directly from embeddings (most efficient).
    
    Assumes embeddings are already computed and normalized.
    
    Args:
        latent_embeddings: Pre-computed frame embeddings
        prompt_embedding: Pre-computed prompt embedding
        
    Returns:
        Per-frame FVVR scores (batch,)
    """
    # Cosine similarity
    similarities = F.cosine_similarity(
        prompt_embedding.unsqueeze(0),
        latent_embeddings,
        dim=1,
    )
    
    # Map from [-1, 1] to [0, 1]
    return (similarities + 1) / 2
