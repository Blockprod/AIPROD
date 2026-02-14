"""AIPROD v2 Core Models

Hybrid architecture combining Attention (global context) + CNN (local efficiency)
for optimal performance on consumer GPUs (GTX 1070+).
"""

from .backbone import HybridBackbone
from .vae import VideoVAE
from .text_encoder import MultilingualTextEncoder

__all__ = ["HybridBackbone", "VideoVAE", "MultilingualTextEncoder"]
