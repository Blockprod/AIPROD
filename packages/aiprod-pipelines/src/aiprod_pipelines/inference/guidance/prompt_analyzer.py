"""
Prompt analysis for adaptive guidance prediction.

Analyzes text prompts to determine complexity and predict optimal
baseline guidance scales for classifier-free guidance (CFG).

Classes:
  - GuidanceProfile: Dataclass holding analysis results
  - PromptAnalyzer: Neural network model for prompt analysis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GuidanceProfile:
    """Analysis results from prompt examination."""
    
    complexity: float
    """Prompt complexity score [0-1]. 0=simple (cat), 1=complex (girl with flowing hair in sunset)."""
    
    base_guidance: float
    """Predicted baseline guidance scale [4.0-10.0]. Higher for complex prompts."""
    
    semantic_components: Dict[str, str]
    """Extracted semantic parts: {subject, action, scene, style, etc}."""
    
    confidence: float
    """Confidence in prediction [0-1]. 1.0 = very confident."""
    
    prompt_length: int
    """Number of tokens in prompt."""


class PromptAnalyzer(nn.Module):
    """
    Neural network for analyzing prompts to predict guidance parameters.
    
    Uses text embeddings (e.g., from Gemma 3 or CLIP) to predict:
    - Prompt complexity (0-1)
    - Optimal baseline guidance scale (4-10)
    
    Architecture:
        Text Embeddings [seq_len, 768]
            ↓
        Attention Pool [768]
            ↓
        Linear Layers → [complexity, base_guidance, confidence]
    
    Size: ~10M parameters, ~40MB checkpoint
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        """
        Initialize PromptAnalyzer.
        
        Args:
            embedding_dim: Size of input embeddings (default: 768 for CLIP)
            hidden_dim: Hidden dimension for MLP (default: 512)
            num_heads: Attention heads for pooling (default: 8)
            num_layers: Transformer layers before pooling (default: 2)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Transformer layers for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Attention-based pooling
        self.pooling_attention = nn.MultiheadAttention(
            embedding_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.pool_query = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),  # [complexity, base_guidance, confidence]
        )
        
        # Output scaling parameters
        self.register_buffer("complexity_min", torch.tensor(0.0))
        self.register_buffer("complexity_max", torch.tensor(1.0))
        self.register_buffer("guidance_min", torch.tensor(4.0))
        self.register_buffer("guidance_max", torch.tensor(10.0))
    
    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Analyze prompt embeddings.
        
        Args:
            embeddings: Text embeddings [batch, seq_len, embedding_dim]
            attention_mask: Attention mask [batch, seq_len] (optional)
        
        Returns:
            Tuple of (complexity, base_guidance, confidence) tensors
            - complexity: [batch] floats in [0, 1]
            - base_guidance: [batch] floats in [4, 10]
            - confidence: [batch] floats in [0, 1]
        """
        # Transformer encoding
        encoded = self.transformer_encoder(embeddings, src_key_padding_mask=~attention_mask if attention_mask is not None else None)
        
        # Attention-based pooling
        pool_query = self.pool_query.expand(encoded.shape[0], -1, -1)
        pooled, _ = self.pooling_attention(pool_query, encoded, encoded)
        pooled = pooled.squeeze(1)  # [batch, embedding_dim]
        
        # Prediction head
        logits = self.head(pooled)  # [batch, 3]
        
        # Unbind and apply scaling
        complexity_logit = logits[..., 0]
        guidance_logit = logits[..., 1]
        confidence_logit = logits[..., 2]
        
        # Map to output ranges using sigmoid + scaling
        complexity = torch.sigmoid(complexity_logit)  # [0, 1]
        base_guidance = (torch.sigmoid(guidance_logit) * (self.guidance_max - self.guidance_min)) + self.guidance_min  # [4, 10]
        confidence = torch.sigmoid(confidence_logit)  # [0, 1]
        
        return complexity, base_guidance, confidence
    
    def analyze(
        self,
        embeddings: torch.Tensor,
        tokens: Optional[List[str]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> GuidanceProfile:
        """
        Analyze single prompt (convenience method).
        
        Args:
            embeddings: Text embeddings [1, seq_len, embedding_dim]
            tokens: Optional token list for semantic extraction
            attention_mask: Optional attention mask
        
        Returns:
            GuidanceProfile with analysis results
        """
        with torch.no_grad():
            complexity, base_guidance, confidence = self.forward(embeddings, attention_mask)
        
        # Extract semantic components (placeholder - would use more sophisticated method)
        semantic_components = self._extract_semantics(tokens) if tokens else {}
        
        return GuidanceProfile(
            complexity=complexity.item(),
            base_guidance=base_guidance.item(),
            semantic_components=semantic_components,
            confidence=confidence.item(),
            prompt_length=embeddings.shape[1],
        )
    
    def _extract_semantics(self, tokens: List[str]) -> Dict[str, str]:
        """
        Extract semantic components from tokens (placeholder).
        
        In production, this would use semantic role labeling or
        other NLP techniques to extract subjects, actions, scenes, etc.
        """
        # Placeholder: return empty dict
        # Production implementation would analyze tokens for:
        # - subject (who/what)
        # - action (what's happening)
        # - scene (where)
        # - style (how it looks)
        # - temporal (when)
        return {}


class PromptAnalyzerPredictor:
    """
    Wrapper for using PromptAnalyzer in inference.
    
    Provides high-level interface for analyzing prompts and getting guidance profiles.
    """
    
    def __init__(
        self,
        model: Optional[PromptAnalyzer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize predictor.
        
        Args:
            model: PromptAnalyzer model (creates if None)
            device: Device for inference
        """
        self.device = device
        
        if model is None:
            model = PromptAnalyzer()
        
        self.model = model.to(device).eval()
    
    def analyze(
        self,
        prompt: str,
        embeddings: torch.Tensor,
        tokens: Optional[List[str]] = None,
    ) -> GuidanceProfile:
        """
        Analyze prompt to get guidance profile.
        
        Args:
            prompt: Text prompt (for reference)
            embeddings: Text embeddings [1, seq_len, embedding_dim]
            tokens: Optional token list
        
        Returns:
            GuidanceProfile with analysis results
        """
        # Ensure embeddings are on correct device
        if embeddings.device != self.model.transformer_encoder[0].device:
            embeddings = embeddings.to(self.device)
        
        # Analyze
        profile = self.model.analyze(embeddings, tokens)
        
        return profile
    
    def batch_analyze(
        self,
        prompts: List[str],
        embeddings_list: List[torch.Tensor],
    ) -> List[GuidanceProfile]:
        """
        Analyze multiple prompts.
        
        Args:
            prompts: List of text prompts
            embeddings_list: List of embeddings (may have different seq_lens)
        
        Returns:
            List of GuidanceProfile objects
        """
        profiles = []
        
        for prompt, embeddings in zip(prompts, embeddings_list):
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(0)
            
            profile = self.analyze(prompt, embeddings)
            profiles.append(profile)
        
        return profiles
    
    def load(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def save(self, checkpoint_path: str) -> None:
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), checkpoint_path)
