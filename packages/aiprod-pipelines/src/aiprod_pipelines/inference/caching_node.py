"""
GraphNode wrappers for intelligent caching in AIPROD inference pipelines.

Provides composable nodes for implementing caching strategies within inference graphs,
with support for embedding caching, feature caching, and VAE output caching.

Components:
  1. CachingProfile: Configuration for caching integration
  2. CachingNode: Initialize and manage cache lifecycle
  3. CachedTextEncodeNode: Text encoding with embedding cache
  4. CachedVAEDecodeNode: VAE decode with output cache
  5. CacheVisualizationNode: Cache statistics and monitoring

Example Usage:
  >>> profile = CachingProfile(
  ...     enable_embedding_cache=True,
  ...     embed_cache_size_mb=256,
  ...     embed_enable=True
  ... )
  >>> context = GraphContext(...)
  >>> node = CachingNode(profile)
  >>> result = node.execute(context)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn

from .caching import InferenceCache, CacheConfig, CacheStatistics


@dataclass
class CachingProfile:
    """Configuration for caching integration into inference pipeline."""
    
    enable_caching: bool = True
    embed_cache_size_mb: int = 256
    feature_cache_size_mb: int = 512
    vae_cache_size_mb: int = 128
    enable_embedding_cache: bool = True
    enable_feature_cache: bool = True
    enable_vae_cache: bool = True
    eviction_policy: str = "lru"
    cache_warmup_samples: int = 0  # Warmup iterations before stats


class CachingNode:
    """GraphNode for cache initialization and lifecycle management.
    
    Creates and initializes InferenceCache with warmup if needed.
    
    Input keys:
      - "text_encoder": Text encoder model (for warmup)
      - Optional: "warmup_prompts": List of warmup prompt lists
      
    Output keys:
      - "cache": InferenceCache instance
      - "cache_initialized": bool
    """
    
    def __init__(self, profile: CachingProfile, device: str = "cuda"):
        """Initialize caching node.
        
        Args:
            profile: CachingProfile configuration
            device: Compute device
        """
        self.profile = profile
        self.device = device
        self.cache: Optional[InferenceCache] = None
    
    @property
    def input_keys(self) -> List[str]:
        """Expected input context keys."""
        keys = []
        if self.profile.enable_embedding_cache:
            keys.append("text_encoder")
        return keys
    
    @property
    def output_keys(self) -> List[str]:
        """Output context keys produced."""
        return ["cache", "cache_initialized"]
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cache initialization.
        
        Args:
            context: GraphContext
            
        Returns:
            Dict with cache and initialization status
        """
        if not self.profile.enable_caching:
            return {
                "cache": None,
                "cache_initialized": False
            }
        
        # Create cache
        config = CacheConfig(
            embed_cache_size_mb=self.profile.embed_cache_size_mb,
            feature_cache_size_mb=self.profile.feature_cache_size_mb,
            vae_cache_size_mb=self.profile.vae_cache_size_mb,
            enable_embedding_cache=self.profile.enable_embedding_cache,
            enable_feature_cache=self.profile.enable_feature_cache,
            enable_vae_cache=self.profile.enable_vae_cache,
            eviction_policy=self.profile.eviction_policy
        )
        
        self.cache = InferenceCache(config, device=self.device)
        
        # Warmup if requested
        if self.profile.cache_warmup_samples > 0 and "warmup_prompts" in context:
            warmup_prompts = context["warmup_prompts"]
            text_encoder = context.get("text_encoder")
            
            if text_encoder is not None:
                for i in range(min(self.profile.cache_warmup_samples, len(warmup_prompts))):
                    prompts = warmup_prompts[i]
                    _ = self.cache.get_embeddings(prompts, text_encoder, self.device)
        
        return {
            "cache": self.cache,
            "cache_initialized": True
        }


class CachedTextEncodeNode:
    """GraphNode wrapper for cached text encoding.
    
    Uses InferenceCache to store and retrieve text embeddings,
    avoiding redundant text encoder computation.
    
    Input keys:
      - "prompts": List[str] or List[List[str]] (batch of prompt lists)
      - "text_encoder": Text encoder model
      - "cache": InferenceCache instance
      
    Output keys:
      - "embeddings": torch.Tensor with encoded embeddings
      - "cache_hit": bool indicating cache hit
      - "cache_stats": CacheStatistics
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize cached text encode node."""
        self.device = device
        self.last_hit = False
    
    @property
    def input_keys(self) -> List[str]:
        """Expected input keys."""
        return ["prompts", "text_encoder", "cache"]
    
    @property
    def output_keys(self) -> List[str]:
        """Output keys produced."""
        return ["embeddings", "cache_hit", "cache_stats"]
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cached text encoding.
        
        Args:
            context: GraphContext with prompts, text_encoder, cache
            
        Returns:
            Dict with embeddings and cache status
        """
        prompts = context["prompts"]
        text_encoder = context["text_encoder"]
        cache = context.get("cache")
        
        if cache is None:
            # No cache - encode directly
            with torch.no_grad():
                embeddings = text_encoder(prompts)
            return {
                "embeddings": embeddings,
                "cache_hit": False,
                "cache_stats": CacheStatistics()
            }
        
        # Record hit/miss before
        hits_before = cache.hits
        
        # Get embeddings with caching
        embeddings = cache.get_embeddings(prompts, text_encoder, self.device)
        
        # Check if hit
        cache_hit = cache.hits > hits_before
        self.last_hit = cache_hit
        
        return {
            "embeddings": embeddings,
            "cache_hit": cache_hit,
            "cache_stats": cache.get_statistics()
        }


class CachedVAEDecodeNode:
    """GraphNode wrapper for cached VAE decoding.
    
    Caches VAE decoder outputs to avoid recomputation when
    decoding the same latents multiple times.
    
    Input keys:
      - "latents": torch.Tensor (latent codes)
      - "vae_decoder": VAE decoder model
      - "cache": InferenceCache instance
      
    Output keys:
      - "video": torch.Tensor (decoded video)
      - "cache_hit": bool
      - "cache_stats": CacheStatistics
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize cached VAE decode node."""
        self.device = device
    
    @property
    def input_keys(self) -> List[str]:
        """Expected input keys."""
        return ["latents", "vae_decoder", "cache"]
    
    @property
    def output_keys(self) -> List[str]:
        """Output keys produced."""
        return ["video", "cache_hit", "cache_stats"]
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cached VAE decoding.
        
        Args:
            context: GraphContext with latents, vae_decoder, cache
            
        Returns:
            Dict with decoded video and cache status
        """
        latents = context["latents"]
        vae_decoder = context["vae_decoder"]
        cache = context.get("cache")
        
        if cache is None:
            # No cache - decode directly
            with torch.no_grad():
                video = vae_decoder(latents)
            return {
                "video": video,
                "cache_hit": False,
                "cache_stats": CacheStatistics()
            }
        
        # Record hits before
        hits_before = cache.hits
        
        # Compute VAE output with caching
        def decode_fn(x):
            with torch.no_grad():
                return vae_decoder(x)
        
        video = cache.get_vae_output("decode", latents, decode_fn)
        
        # Check if hit
        cache_hit = cache.hits > hits_before
        
        return {
            "video": video,
            "cache_hit": cache_hit,
            "cache_stats": cache.get_statistics()
        }


class CacheMonitoringNode:
    """GraphNode for cache statistics and visualization.
    
    Tracks cache performance metrics and optionally clears cache
    based on memory/performance thresholds.
    
    Input keys:
      - "cache": InferenceCache instance
      
    Output keys:
      - "cache_stats": CacheStatistics
      - "should_clear_cache": bool
      - "monitoring_report": str
    """
    
    def __init__(
        self,
        memory_threshold_mb: float = 1024.0,
        hit_rate_threshold: float = 60.0,
        enable_auto_clear: bool = False
    ):
        """Initialize cache monitoring node.
        
        Args:
            memory_threshold_mb: Clear if cache exceeds this size
            hit_rate_threshold: Warn if hit rate below this percent
            enable_auto_clear: Auto-clear if thresholds exceeded
        """
        self.memory_threshold_mb = memory_threshold_mb
        self.hit_rate_threshold = hit_rate_threshold
        self.enable_auto_clear = enable_auto_clear
        self.history: List[CacheStatistics] = []
    
    @property
    def input_keys(self) -> List[str]:
        """Expected input keys."""
        return ["cache"]
    
    @property
    def output_keys(self) -> List[str]:
        """Output keys produced."""
        return ["cache_stats", "should_clear_cache", "monitoring_report"]
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cache monitoring.
        
        Args:
            context: GraphContext with cache
            
        Returns:
            Dict with statistics and recommendations
        """
        cache = context.get("cache")
        
        if cache is None:
            return {
                "cache_stats": CacheStatistics(),
                "should_clear_cache": False,
                "monitoring_report": "No cache active"
            }
        
        stats = cache.get_statistics()
        self.history.append(stats)
        
        # Check thresholds
        should_clear = False
        warning_messages = []
        
        if stats.total_cache_size_mb > self.memory_threshold_mb:
            warning_messages.append(
                f"Cache size {stats.total_cache_size_mb:.1f}MB exceeds threshold {self.memory_threshold_mb:.1f}MB"
            )
            should_clear = should_clear or self.enable_auto_clear
        
        if stats.hit_rate_percent < self.hit_rate_threshold and stats.total_hits > 0:
            warning_messages.append(
                f"Hit rate {stats.hit_rate_percent:.1f}% below threshold {self.hit_rate_threshold:.1f}%"
            )
        
        # Generate report
        report = (
            f"Cache Stats: "
            f"Hits={stats.total_hits}, Misses={stats.total_misses}, "
            f"HitRate={stats.hit_rate_percent:.1f}%, "
            f"Size={stats.total_cache_size_mb:.1f}MB, "
            f"Speedup={stats.speedup_factor:.2f}x"
        )
        
        if warning_messages:
            report += " | Warnings: " + "; ".join(warning_messages)
        
        return {
            "cache_stats": stats,
            "should_clear_cache": should_clear,
            "monitoring_report": report
        }
    
    def get_history(self) -> List[CacheStatistics]:
        """Get cache statistics history."""
        return self.history.copy()
