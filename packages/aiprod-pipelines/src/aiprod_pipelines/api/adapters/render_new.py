"""
Render Executor Adapter - Video Generation with Retry & Fallback
================================================================

Executes video generation with intelligent retry logic, exponential backoff,
and multi-level fallback chain across sovereign backend tiers.

PHASE 1 implementation (Weeks 4-5 in execution plan).

Backend tiers (100% souverain — aucun appel réseau externe) :
- aiprod_shdt         : Standard SHDT diffusion (default)
- aiprod_shdt_fast    : Lower-step count, faster/cheaper
- aiprod_shdt_premium : Higher-step count, max quality
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
import time
import random
from .base import BaseAdapter
from ..schema.schemas import Context


class RenderExecutorAdapter(BaseAdapter):
    """
    Render executor with advanced retry logic and fallback chain.
    
    Features:
    - Batch processing (configurable batch size)
    - Exponential backoff retry strategy
    - Multi-backend fallback (primary → fallback chain)
    - Deterministic seeding for reproducibility
    - Rate limiting between batches
    - Failure tracking and logging
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize render executor."""
        super().__init__(config)
        
        # Backend clients (will be injected in PHASE 1)
        self.backends = config.get("backends", {}) if config else {}
        
        # Retry configuration
        self.batch_size = config.get("batch_size", 4) if config else 4
        self.max_retries = config.get("max_retries", 3) if config else 3
        self.base_backoff_delay = 1.0  # seconds
        self.max_backoff_delay = 30.0
        
        # Fallback chain: primary backend → chain of fallbacks (souverain)
        self.fallback_chain = ["aiprod_shdt", "aiprod_shdt_fast"]
        
        # Rate limiting
        self.rate_limit_delay = 2.0  # seconds between batches
        self.rate_limit_requests = 10  # requests per window
        self.rate_limit_window = 60  # seconds
        
        # Tracking
        self.batch_failure_count = {}
        self.request_times = []
    
    async def execute(self, ctx: Context) -> Context:
        """
        Generate video assets from shot list with retry + fallback.
        
        Args:
            ctx: Context with shot_list and cost_estimation
            
        Returns:
            Context with generated_assets
        """
        # Validate context
        if not self.validate_context(ctx, ["shot_list"]):
            raise ValueError("Missing shot_list in context")
        
        shot_list = ctx["memory"]["shot_list"]
        selected_backend = ctx["memory"].get("cost_estimation", {}).get("selected_backend", "aiprod_shdt")
        
        ctx["memory"]["render_start"] = time.time()
        
        # Batch processing
        batches = self._create_batches(shot_list, self.batch_size)
        generated_assets: List[Dict[str, Any]] = []
        failed_batches: List[Tuple[int, List[Dict]]] = []
        
        for batch_idx, batch in enumerate(batches):
            try:
                # Check rate limit before rendering
                await self._check_rate_limit()
                
                # Render batch with retry + fallback
                batch_results = await self._render_batch_with_retry(
                    batch=batch,
                    primary_backend=selected_backend,
                    batch_idx=batch_idx
                )
                
                if batch_results:
                    generated_assets.extend(batch_results)
                    self.log("info", f"Batch {batch_idx + 1} rendered successfully", 
                             size=len(batch), backend=selected_backend)
                else:
                    # Batch failed despite retries
                    failed_batches.append((batch_idx, batch))
                    self.log("error", f"Batch {batch_idx + 1} failed after all retries", size=len(batch))
                
                # Rate limiting delay
                if batch_idx < len(batches) - 1:
                    await asyncio.sleep(self.rate_limit_delay)
            
            except Exception as e:
                failed_batches.append((batch_idx, batch))
                self.log("error", f"Batch {batch_idx + 1} exception", error=str(e))
        
        # Check if we have at least some results
        if not generated_assets and failed_batches:
            raise Exception(f"All {len(batches)} batches failed during rendering")
        
        # Store results
        ctx["memory"]["generated_assets"] = generated_assets
        ctx["memory"]["render_duration_sec"] = time.time() - ctx["memory"]["render_start"]
        ctx["memory"]["render_stats"] = {
            "total_shots": len(shot_list),
            "batches_processed": len(batches),
            "assets_generated": len(generated_assets),
            "failed_batches": len(failed_batches),
            "success_rate": len(generated_assets) / len(shot_list) if shot_list else 0
        }
        
        self.log("info", "Rendering complete", 
                 assets=len(generated_assets), failures=len(failed_batches),
                 duration=ctx["memory"]["render_duration_sec"])
        
        return ctx
    
    async def _render_batch_with_retry(
        self,
        batch: List[Dict[str, Any]],
        primary_backend: str,
        batch_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Render batch with retry logic and fallback chain.
        
        Args:
            batch: List of shots to render
            primary_backend: Primary backend to try first
            batch_idx: Batch index for tracking
            
        Returns:
            List of generated assets or empty list if all failed
        """
        backends_to_try = [primary_backend] + self.fallback_chain
        
        for retry_attempt in range(self.max_retries):
            for fallback_backend in backends_to_try:
                try:
                    # Render with specific backend
                    results = await self._render_with_backend(
                        batch, fallback_backend, batch_idx, retry_attempt
                    )
                    
                    if results:
                        return results
                    
                except asyncio.TimeoutError:
                    # Timeout - try next backend
                    self.log("warning", f"Backend {fallback_backend} timeout", 
                             batch=batch_idx, retry=retry_attempt)
                    continue
                
                except Exception as e:
                    # Other error - try next backend
                    self.log("warning", f"Backend {fallback_backend} error: {str(e)}", 
                             batch=batch_idx, retry=retry_attempt)
                    continue
            
            # All backends tried - apply backoff and retry
            if retry_attempt < self.max_retries - 1:
                delay = self._calculate_backoff_delay(retry_attempt)
                self.log("info", f"Retry {retry_attempt + 1}/{self.max_retries} for batch {batch_idx}", 
                         delay=delay)
                await asyncio.sleep(delay)
        
        # All retries exhausted
        return []
    
    async def _render_with_backend(
        self,
        batch: List[Dict[str, Any]],
        backend: str,
        batch_idx: int,
        retry_attempt: int
    ) -> List[Dict[str, Any]]:
        """
        Render batch with specific backend.
        
        Args:
            batch: Shots to render
            backend: Backend to use
            batch_idx: Batch index
            retry_attempt: Retry attempt number
            
        Returns:
            Generated assets or None on failure
        """
        # In production, would call actual backend API
        # For now: Mock generation with configurable success rate
        
        # Simulate probability of failure (lower on retries)
        success_probability = 0.95 if retry_attempt == 0 else 0.98
        
        if random.random() > success_probability:
            raise Exception(f"Simulated {backend} failure")
        
        # Generate mock assets
        results = []
        for shot in batch:
            asset = {
                "id": shot["shot_id"],
                "url": f"/data/aiprod-assets/{shot['shot_id']}.mp4",
                "duration_sec": shot.get("duration_sec", 10),
                "resolution": "1080p",
                "codec": "h264",
                "bitrate": 5000000,
                "file_size_bytes": int(shot.get("duration_sec", 10) * 5000000 // 8),
                "thumbnail_url": f"/data/aiprod-assets/{shot['shot_id']}_thumb.jpg",
                "backend_used": backend,
                "seed": shot.get("seed", 0),
                "generated_at": time.time()
            }
            results.append(asset)
        
        return results
    
    def _create_batches(
        self, 
        shots: List[Dict[str, Any]], 
        batch_size: int
    ) -> List[List[Dict[str, Any]]]:
        """
        Partition shots into batches.
        
        Args:
            shots: List of shots
            batch_size: Shots per batch
            
        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(shots), batch_size):
            batches.append(shots[i:i + batch_size])
        return batches
    
    def _calculate_backoff_delay(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.
        
        Args:
            attempt: Retry attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential: 1s, 2s, 4s, 8s, ...
        delay = self.base_backoff_delay * (2 ** attempt)
        
        # Add jitter (±20%)
        jitter = delay * random.uniform(0.8, 1.2)
        
        # Cap at maximum
        return min(jitter, self.max_backoff_delay)
    
    async def _check_rate_limit(self) -> None:
        """
        Check and respect rate limiting.
        
        Tracks request timing and waits if necessary.
        """
        now = time.time()
        
        # Remove old entries outside window
        self.request_times = [t for t in self.request_times if now - t < self.rate_limit_window]
        
        # Check if limit exceeded
        if len(self.request_times) >= self.rate_limit_requests:
            # Need to wait
            oldest_time = min(self.request_times)
            wait_time = self.rate_limit_window - (now - oldest_time)
            
            if wait_time > 0:
                self.log("info", "Rate limit reached, waiting", delay=wait_time)
                await asyncio.sleep(wait_time)
                
                # Clear old entries
                self.request_times = []
        
        # Record this request
        self.request_times.append(now)
    
    def _get_deterministic_seed(self, prompt: str, shot_id: str) -> int:
        """
        Generate reproducible seed from prompt and shot.
        
        Args:
            prompt: Shot prompt
            shot_id: Shot identifier
            
        Returns:
            Deterministic integer seed
        """
        import hashlib
        combined = f"{shot_id}_{prompt}"
        hash_val = hashlib.sha256(combined.encode()).hexdigest()
        return int(hash_val, 16) % (2 ** 32)
