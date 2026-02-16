"""
Render Executor Adapter - Video Generation with Local GPU Worker
================================================================

Executes video generation on the local GPU via GPUWorker.
Retry + fallback architecture conservée, mais le backend est SOUVERAIN.

PHASE 2 implementation — Zéro mock, zéro cloud, 100% local.
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
import logging
import time
import random
from .base import BaseAdapter
from ..schema.schemas import Context

logger = logging.getLogger(__name__)

# Import souverain : GPUWorker local
_GPU_WORKER = None  # Singleton lazy-loaded


def _get_gpu_worker():
    """Lazy-load le GPUWorker (singleton pour réutiliser le pipeline chargé)."""
    global _GPU_WORKER
    if _GPU_WORKER is None:
        from aiprod_pipelines.api.gpu_worker import GPUWorker, WorkerConfig
        _GPU_WORKER = GPUWorker(config=WorkerConfig())
        try:
            _GPU_WORKER.load_pipeline()
        except Exception as e:
            logger.warning("Pipeline loading failed (stub mode): %s", e)
    return _GPU_WORKER


class RenderExecutorAdapter(BaseAdapter):
    """
    Render executor souverain avec GPU local.
    
    Features:
    - Batch processing (configurable batch size)
    - Exponential backoff retry strategy
    - GPU local via GPUWorker (zéro API cloud)
    - Deterministic seeding for reproducibility
    - Rate limiting between batches
    - Failure tracking and logging
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize render executor."""
        super().__init__(config)
        
        # Retry configuration
        self.batch_size = config.get("batch_size", 4) if config else 4
        self.max_retries = config.get("max_retries", 3) if config else 3
        self.base_backoff_delay = 1.0  # seconds
        self.max_backoff_delay = 30.0
        
        # Rate limiting
        self.rate_limit_delay = 0.5  # seconds between batches (local = rapide)
        self.rate_limit_requests = 50  # pas de throttling cloud
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
        selected_backend = ctx["memory"].get("cost_estimation", {}).get("selected_backend", "aiprod_sovereign")
        
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
        Render batch avec retry — backend local GPU uniquement.
        
        Args:
            batch: List of shots to render
            primary_backend: Ignoré (toujours local)
            batch_idx: Batch index for tracking
            
        Returns:
            List of generated assets or empty list if all failed
        """
        for retry_attempt in range(self.max_retries):
            try:
                results = await self._render_with_backend(
                    batch, "aiprod_sovereign", batch_idx, retry_attempt
                )
                
                if results:
                    return results
                
            except asyncio.TimeoutError:
                self.log("warning", f"GPU render timeout",
                         batch=batch_idx, retry=retry_attempt)
                continue
                
            except Exception as e:
                self.log("warning", f"GPU render error: {str(e)}",
                         batch=batch_idx, retry=retry_attempt)
                continue
            
            # Apply backoff and retry
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
        Render batch via le GPUWorker souverain local.
        
        Remplace le mock GCS — génère de VRAIS fichiers vidéo locaux.
        """
        from aiprod_pipelines.api.gpu_worker import JobRequest
        
        worker = _get_gpu_worker()
        results = []
        
        for shot in batch:
            # Créer la requête de génération
            request = JobRequest(
                prompt=shot.get("prompt", ""),
                job_id=shot.get("shot_id", ""),
                negative_prompt=shot.get("negative_prompt", ""),
                seed=shot.get("seed", self._get_deterministic_seed(
                    shot.get("prompt", ""), shot.get("shot_id", "")
                )),
                height=shot.get("height", worker.config.default_height),
                width=shot.get("width", worker.config.default_width),
                num_frames=shot.get("num_frames", worker.config.default_num_frames),
                fps=shot.get("fps", worker.config.default_fps),
                num_inference_steps=shot.get("num_inference_steps", worker.config.default_num_steps),
                guidance_scale=shot.get("guidance_scale", worker.config.default_guidance_scale),
                duration_sec=shot.get("duration_sec", 5.0),
            )
            
            # Exécuter la génération (synchrone — le worker fait le GPU work)
            job_result = await asyncio.to_thread(worker.process_job, request)
            
            if job_result.status.value == "completed":
                asset = {
                    "id": shot.get("shot_id", request.job_id),
                    "url": f"file://{job_result.output_path}",
                    "duration_sec": shot.get("duration_sec", 5.0),
                    "resolution": f"{request.width}x{request.height}",
                    "codec": "h264",
                    "bitrate": 5000000,
                    "file_size_bytes": self._get_file_size(job_result.output_path),
                    "backend_used": "aiprod_sovereign",
                    "seed": request.seed,
                    "generated_at": time.time(),
                    "generation_time_sec": job_result.duration_sec,
                    "output_path": job_result.output_path,
                }
                results.append(asset)
                logger.info("Shot %s generated in %.1fs → %s",
                           shot.get("shot_id", "?"), job_result.duration_sec,
                           job_result.output_path)
            else:
                logger.error("Shot %s failed: %s",
                           shot.get("shot_id", "?"), job_result.error)
        
        return results if results else None
    
    @staticmethod
    def _get_file_size(path: Optional[str]) -> int:
        """Retourne la taille du fichier ou 0."""
        if path:
            from pathlib import Path
            p = Path(path)
            if p.exists():
                return p.stat().st_size
        return 0
    
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
