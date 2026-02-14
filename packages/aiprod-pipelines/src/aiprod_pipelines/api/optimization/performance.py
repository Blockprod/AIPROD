"""
Performance Optimization Layer - Multi-Tier Caching & Optimization
==================================================================

Implements performance optimizations:
- Multi-tier caching (Gemini, consistency, batch)
- Lazy loading strategies
- Predictive chunking at scene boundaries
- Prefetching for next state
- Adaptive batch sizing

PHASE 4 implementation (Weeks 11-13).
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime, timedelta
from cachetools import TTLCache, LRUCache
import hashlib
from ..schema.schemas import Context, State


logger = logging.getLogger(__name__)


class PerformanceOptimizationLayer:
    """
    Enhanced caching and performance optimization.
    
    Features:
    - 3-tier cache system (LLM results, consistency markers, adaptive batching)
    - Lazy loading for large assets
    - Predictive chunking at scene boundaries
    - State-aware prefetching
    - Cache analytics and hit rate tracking
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize performance optimization layer."""
        self.config = config or {}
        
        # Multi-tier caching
        self.gemini_cache = TTLCache(
            maxsize=self.config.get("gemini_cache_size", 5000),
            ttl=self.config.get("gemini_cache_ttl", 86400)  # 24h
        )
        
        self.consistency_cache = TTLCache(
            maxsize=self.config.get("consistency_cache_size", 1000),
            ttl=self.config.get("consistency_cache_ttl", 604800)  # 168h (7 days)
        )
        
        self.adaptive_batch_cache = LRUCache(
            maxsize=self.config.get("batch_cache_size", 500)
        )
        
        # Cache statistics
        self.cache_stats = {
            "gemini": {"hits": 0, "misses": 0},
            "consistency": {"hits": 0, "misses": 0},
            "batch": {"hits": 0, "misses": 0}
        }
        
        # Optimization settings
        self.lazy_loading_threshold = self.config.get("lazy_loading_threshold", 10_000_000)  # 10MB
        self.prefetch_enabled = self.config.get("prefetch_enabled", True)
        self.chunk_at_scene_boundaries = self.config.get("chunk_at_scene_boundaries", True)
        
        logger.info("PerformanceOptimizationLayer initialized")
    
    async def optimize_for_performance(self, ctx: Context) -> Context:
        """
        Apply all performance optimizations to context.
        
        Args:
            ctx: Current context
            
        Returns:
            Optimized context
        """
        # 1. Configure lazy loading
        ctx = await self._apply_lazy_loading(ctx)
        
        # 2. Predictive chunking
        if self.chunk_at_scene_boundaries:
            ctx = await self._apply_predictive_chunking(ctx)
        
        # 3. Prefetch next state inputs
        if self.prefetch_enabled:
            ctx = await self._prefetch_next_state(ctx)
        
        # 4. Optimize batch sizes
        ctx = await self._optimize_batch_sizes(ctx)
        
        logger.info("Performance optimizations applied",
                    state=ctx.get("state"),
                    cache_stats=self.get_cache_stats())
        
        return ctx
    
    async def _apply_lazy_loading(self, ctx: Context) -> Context:
        """
        Configure lazy loading for large assets.
        
        Args:
            ctx: Context
            
        Returns:
            Context with lazy loading configured
        """
        if "generated_assets" in ctx["memory"]:
            assets = ctx["memory"]["generated_assets"]
            
            for asset in assets:
                file_size = asset.get("file_size_bytes", 0)
                
                if file_size > self.lazy_loading_threshold:
                    # Mark for lazy loading
                    asset["loading_strategy"] = "lazy"
                    asset["load_on_demand"] = True
                else:
                    asset["loading_strategy"] = "eager"
                    asset["load_on_demand"] = False
        
        ctx["memory"]["lazy_loading_enabled"] = True
        
        return ctx
    
    async def _apply_predictive_chunking(self, ctx: Context) -> Context:
        """
        Split videos at natural scene boundaries for efficient processing.
        
        Args:
            ctx: Context with visual_translation
            
        Returns:
            Context with chunk boundaries
        """
        visual_translation = ctx["memory"].get("visual_translation", {})
        
        if not visual_translation:
            return ctx
        
        scenes = visual_translation.get("scenes", [])
        shots = visual_translation.get("shots", [])
        
        # Calculate chunk boundaries
        boundaries = await self._predict_chunk_boundaries(scenes, shots)
        
        ctx["memory"]["chunk_boundaries"] = boundaries
        ctx["memory"]["chunking_strategy"] = "scene_boundary"
        
        logger.info(f"Predicted {len(boundaries)} chunk boundaries")
        
        return ctx
    
    async def _predict_chunk_boundaries(
        self,
        scenes: List[Dict[str, Any]],
        shots: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Predict optimal chunk boundaries based on scene structure.
        
        Args:
            scenes: Scene list from creative direction
            shots: Shot list from visual translation
            
        Returns:
            List of frame boundaries
        """
        boundaries = []
        cumulative_frames = 0
        
        # Use scenes if available, otherwise shots
        segments = scenes if scenes else shots
        
        for segment in segments:
            duration = segment.get("duration_sec", segment.get("duration", 10))
            fps = segment.get("fps", 30)
            
            frames = int(duration * fps)
            cumulative_frames += frames
            boundaries.append(cumulative_frames)
        
        return boundaries
    
    async def _prefetch_next_state(self, ctx: Context) -> Context:
        """
        Prefetch inputs required for next state transition.
        
        Args:
            ctx: Current context
            
        Returns:
            Context with prefetched data
        """
        current_state = ctx.get("state")
        
        # Predict next state
        next_state = self._predict_next_state(current_state)
        
        if not next_state:
            return ctx
        
        # Prefetch inputs for next state
        prefetched = await self._prefetch_inputs_for_state(next_state, ctx)
        
        ctx["memory"]["prefetched"] = prefetched
        ctx["memory"]["prefetch_state"] = next_state
        
        logger.info(f"Prefetched inputs for {next_state}")
        
        return ctx
    
    def _predict_next_state(self, current_state: str) -> Optional[str]:
        """
        Predict next state transition.
        
        Args:
            current_state: Current state name
            
        Returns:
            Predicted next state name or None
        """
        # State transition map
        transitions = {
            State.INIT: State.ANALYSIS,
            State.ANALYSIS: State.CREATIVE_DIRECTION,
            State.CREATIVE_DIRECTION: State.VISUAL_TRANSLATION,  # Default path
            State.VISUAL_TRANSLATION: State.FINANCIAL_OPTIMIZATION,
            State.FINANCIAL_OPTIMIZATION: State.RENDER_EXECUTION,
            State.RENDER_EXECUTION: State.QA_TECHNICAL,
            State.QA_TECHNICAL: State.QA_SEMANTIC,
            State.QA_SEMANTIC: State.FINALIZE,
            State.FINALIZE: State.COMPLETE
        }
        
        return transitions.get(current_state)
    
    async def _prefetch_inputs_for_state(
        self,
        target_state: str,
        ctx: Context
    ) -> Dict[str, Any]:
        """
        Prefetch inputs required for target state.
        
        Args:
            target_state: State to prefetch for
            ctx: Current context
            
        Returns:
            Prefetched data dictionary
        """
        prefetched = {"target_state": target_state}
        
        # State-specific prefetching
        if target_state == State.CREATIVE_DIRECTION:
            # Prefetch Gemini cache
            prefetched["gemini_cache_warmed"] = True
        
        elif target_state == State.RENDER_EXECUTION:
            # Prefetch backend availability
            prefetched["backends_available"] = ["veo3", "runway_gen3"]
        
        elif target_state == State.QA_TECHNICAL:
            # Prefetch video metadata
            if "generated_assets" in ctx["memory"]:
                prefetched["video_count"] = len(ctx["memory"]["generated_assets"])
        
        return prefetched
    
    async def _optimize_batch_sizes(self, ctx: Context) -> Context:
        """
        Optimize batch sizes for current workload.
        
        Args:
            ctx: Context
            
        Returns:
            Context with optimized batch configuration
        """
        # Analyze workload
        workload = self._analyze_workload(ctx)
        
        # Determine optimal batch size
        optimal_batch_size = self._calculate_optimal_batch_size(workload)
        
        ctx["memory"]["batch_config"] = {
            "batch_size": optimal_batch_size,
            "workload_type": workload["type"],
            "estimated_throughput": workload["estimated_throughput"]
        }
        
        logger.info(f"Optimized batch size: {optimal_batch_size}")
        
        return ctx
    
    def _analyze_workload(self, ctx: Context) -> Dict[str, Any]:
        """
        Analyze current workload characteristics.
        
        Args:
            ctx: Context
            
        Returns:
            Workload analysis
        """
        complexity = ctx["memory"].get("complexity", 0.5)
        duration = ctx["memory"].get("duration_sec", 30)
        
        # Classify workload
        if complexity < 0.3 and duration < 30:
            workload_type = "light"
            estimated_throughput = 20  # videos/min
        elif complexity < 0.7 and duration < 120:
            workload_type = "medium"
            estimated_throughput = 10
        else:
            workload_type = "heavy"
            estimated_throughput = 5
        
        return {
            "type": workload_type,
            "complexity": complexity,
            "duration": duration,
            "estimated_throughput": estimated_throughput
        }
    
    def _calculate_optimal_batch_size(self, workload: Dict[str, Any]) -> int:
        """
        Calculate optimal batch size for workload.
        
        Args:
            workload: Workload analysis
            
        Returns:
            Optimal batch size
        """
        workload_type = workload["type"]
        
        # Batch size by workload type
        batch_sizes = {
            "light": 8,
            "medium": 4,
            "heavy": 2
        }
        
        return batch_sizes.get(workload_type, 4)
    
    # ========================================================================
    # Cache Management Methods
    # ========================================================================
    
    def cache_gemini_result(self, prompt: str, result: Any, ttl: Optional[int] = None):
        """
        Cache Gemini LLM result.
        
        Args:
            prompt: Input prompt
            result: LLM result
            ttl: Optional custom TTL in seconds
        """
        cache_key = self._hash_prompt(prompt)
        self.gemini_cache[cache_key] = {
            "result": result,
            "cached_at": datetime.now().isoformat()
        }
        
        logger.debug(f"Cached Gemini result: {cache_key[:8]}")
    
    def get_cached_gemini_result(self, prompt: str) -> Optional[Any]:
        """
        Retrieve cached Gemini result.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Cached result or None
        """
        cache_key = self._hash_prompt(prompt)
        
        if cache_key in self.gemini_cache:
            self.cache_stats["gemini"]["hits"] += 1
            cached = self.gemini_cache[cache_key]
            logger.debug(f"Gemini cache HIT: {cache_key[:8]}")
            return cached["result"]
        
        self.cache_stats["gemini"]["misses"] += 1
        logger.debug(f"Gemini cache MISS: {cache_key[:8]}")
        return None
    
    def cache_consistency_markers(self, shot_id: str, markers: Dict[str, Any]):
        """
        Cache consistency markers for shot.
        
        Args:
            shot_id: Shot identifier
            markers: Consistency markers
        """
        self.consistency_cache[shot_id] = markers
        logger.debug(f"Cached consistency markers: {shot_id}")
    
    def get_cached_consistency_markers(self, shot_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached consistency markers.
        
        Args:
            shot_id: Shot identifier
            
        Returns:
            Cached markers or None
        """
        if shot_id in self.consistency_cache:
            self.cache_stats["consistency"]["hits"] += 1
            return self.consistency_cache[shot_id]
        
        self.cache_stats["consistency"]["misses"] += 1
        return None
    
    def _hash_prompt(self, prompt: str) -> str:
        """
        Generate cache key from prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Cache key string
        """
        return hashlib.sha256(prompt.encode()).hexdigest()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        stats = {}
        
        for cache_name, cache_data in self.cache_stats.items():
            hits = cache_data["hits"]
            misses = cache_data["misses"]
            total = hits + misses
            
            hit_rate = hits / total if total > 0 else 0.0
            
            stats[cache_name] = {
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_rate": f"{hit_rate * 100:.1f}%"
            }
        
        return stats
    
    def clear_caches(self):
        """Clear all cache tiers."""
        self.gemini_cache.clear()
        self.consistency_cache.clear()
        self.adaptive_batch_cache.clear()
        
        logger.info("All caches cleared")
