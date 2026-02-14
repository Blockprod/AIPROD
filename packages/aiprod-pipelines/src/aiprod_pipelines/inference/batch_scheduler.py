"""
AIPROD Batch Scheduler
=======================

Dynamic inference batching for GPU throughput maximisation:
- Groups incoming requests by resolution bin + duration range
- Timeout-based dispatch (max 5 s wait before execution)
- Memory-aware batch sizing (fits within GPU VRAM budget)
- Priority-weighted ordering (Enterprise > Pro > Free)
- Integrates with existing ``dynamic_batch_sizing/`` adaptive batcher

The scheduler runs as an asyncio background task that drains request
queues and dispatches batches to GPU workers.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ResolutionBin(str, Enum):
    """Discrete resolution bins for batching compatibility."""

    SD = "512x288"
    HD = "768x432"
    FHD = "1280x720"
    QHD = "1920x1080"
    UHD = "3840x2160"

    @classmethod
    def from_dims(cls, width: int, height: int) -> "ResolutionBin":
        area = width * height
        if area <= 512 * 288:
            return cls.SD
        elif area <= 768 * 432:
            return cls.HD
        elif area <= 1280 * 720:
            return cls.FHD
        elif area <= 1920 * 1080:
            return cls.QHD
        return cls.UHD


@dataclass
class BatchConfig:
    """Batch scheduler configuration."""

    max_wait_sec: float = 5.0  # max time a request waits before dispatch
    max_batch_size: int = 8
    gpu_vram_budget_gb: float = 22.0  # e.g. RTX 4090 24 GB – reserve 2 GB
    duration_tolerance_sec: float = 2.0  # requests within ±2 s are batched together
    priority_weights: Dict[str, int] = field(default_factory=lambda: {
        "free": 1,
        "pro": 5,
        "enterprise": 20,
    })


# ---------------------------------------------------------------------------
# Request & Batch objects
# ---------------------------------------------------------------------------


@dataclass
class InferenceRequest:
    """A single inference request waiting to be batched."""

    request_id: str = ""
    tenant_id: str = ""
    priority: int = 1
    width: int = 768
    height: int = 432
    duration_sec: float = 5.0
    prompt: str = ""
    features: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    # Estimated VRAM needed (GB), set by memory estimator
    estimated_vram_gb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()
        if self.estimated_vram_gb <= 0:
            self.estimated_vram_gb = self._estimate_vram()

    def _estimate_vram(self) -> float:
        """Heuristic VRAM estimate (GB) based on resolution + duration."""
        pixels = self.width * self.height
        base = pixels / (1920 * 1080) * 4.0  # 4 GB baseline at 1080p
        duration_factor = min(self.duration_sec / 10.0, 3.0)
        return round(base * (0.5 + 0.5 * duration_factor), 2)

    @property
    def resolution_bin(self) -> ResolutionBin:
        return ResolutionBin.from_dims(self.width, self.height)


@dataclass
class Batch:
    """A grouped batch ready for GPU execution."""

    batch_id: str = ""
    resolution_bin: ResolutionBin = ResolutionBin.FHD
    requests: List[InferenceRequest] = field(default_factory=list)
    total_vram_gb: float = 0.0
    created_at: float = 0.0
    dispatched_at: float = 0.0
    completed_at: float = 0.0

    def __post_init__(self):
        if not self.batch_id:
            self.batch_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()

    @property
    def size(self) -> int:
        return len(self.requests)


# ---------------------------------------------------------------------------
# Memory-aware batch sizing
# ---------------------------------------------------------------------------


class MemoryAwareSizer:
    """
    Determines how many requests fit in one batch given VRAM budget.

    Uses per-request VRAM estimates and checks running total.
    """

    def __init__(self, vram_budget_gb: float = 22.0):
        self._budget = vram_budget_gb

    def fit(self, requests: List[InferenceRequest]) -> Tuple[List[InferenceRequest], List[InferenceRequest]]:
        """
        Split requests into (fits, overflow) based on VRAM budget.

        Returns:
            (selected, remaining) tuples.
        """
        selected: List[InferenceRequest] = []
        remaining: List[InferenceRequest] = []
        used = 0.0

        for req in requests:
            if used + req.estimated_vram_gb <= self._budget:
                selected.append(req)
                used += req.estimated_vram_gb
            else:
                remaining.append(req)

        return selected, remaining


# ---------------------------------------------------------------------------
# Batch Scheduler
# ---------------------------------------------------------------------------

# Type alias for the async dispatch callback
DispatchFn = Callable[[Batch], Coroutine[Any, Any, None]]


class BatchScheduler:
    """
    Async batch scheduler.

    Requests are submitted via ``submit()``.  A background loop groups them
    by resolution bin and duration similarity, then dispatches when:
    1. Batch reaches max size, OR
    2. Oldest request in queue exceeds ``max_wait_sec``

    Within each bin, requests are sorted by priority (descending) so
    Enterprise tenants are served first.

    Usage:
        scheduler = BatchScheduler(config, dispatch_fn=my_gpu_worker)
        scheduler.start()                # start background loop
        future = await scheduler.submit(req)  # submit request
        await scheduler.stop()           # graceful shutdown
    """

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        dispatch_fn: Optional[DispatchFn] = None,
    ):
        self._config = config or BatchConfig()
        self._dispatch_fn = dispatch_fn or self._default_dispatch
        self._sizer = MemoryAwareSizer(self._config.gpu_vram_budget_gb)

        # Queues keyed by (ResolutionBin, duration_bucket)
        self._queues: Dict[Tuple[ResolutionBin, int], List[InferenceRequest]] = {}
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._running = False

        # Stats
        self._stats = {
            "total_submitted": 0,
            "total_dispatched": 0,
            "total_batches": 0,
            "avg_batch_size": 0.0,
            "avg_wait_sec": 0.0,
        }

    # ---- Queue key helpers ------------------------------------------------

    def _duration_bucket(self, duration_sec: float) -> int:
        """Bucket duration into tolerance-wide groups."""
        tol = self._config.duration_tolerance_sec
        return int(duration_sec / tol) if tol > 0 else 0

    def _queue_key(self, req: InferenceRequest) -> Tuple[ResolutionBin, int]:
        return (req.resolution_bin, self._duration_bucket(req.duration_sec))

    # ---- Public API -------------------------------------------------------

    async def submit(self, request: InferenceRequest) -> str:
        """
        Submit an inference request for batching.

        Returns the request_id for tracking.
        """
        async with self._lock:
            key = self._queue_key(request)
            self._queues.setdefault(key, []).append(request)
            self._stats["total_submitted"] += 1
        return request.request_id

    def start(self) -> None:
        """Start the background scheduling loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._loop())

    async def stop(self) -> None:
        """Stop the scheduler and flush remaining requests."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # Flush remaining queues
        await self._flush_all()

    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    @property
    def pending_count(self) -> int:
        return sum(len(q) for q in self._queues.values())

    # ---- Background loop --------------------------------------------------

    async def _loop(self) -> None:
        """Main scheduling loop — checks every 100 ms."""
        while self._running:
            try:
                await self._tick()
            except Exception:
                pass  # log in production
            await asyncio.sleep(0.1)

    async def _tick(self) -> None:
        """Single scheduling tick — check all queues."""
        now = time.time()
        batches_to_dispatch: List[Batch] = []

        async with self._lock:
            keys_to_remove: List[Tuple[ResolutionBin, int]] = []

            for key, queue in self._queues.items():
                if not queue:
                    keys_to_remove.append(key)
                    continue

                # Sort by priority descending
                queue.sort(key=lambda r: r.priority, reverse=True)

                # Check dispatch conditions
                oldest_age = now - queue[0].timestamp
                should_dispatch = (
                    len(queue) >= self._config.max_batch_size
                    or oldest_age >= self._config.max_wait_sec
                )

                if should_dispatch:
                    # Memory-aware sizing
                    fits, overflow = self._sizer.fit(
                        queue[: self._config.max_batch_size]
                    )
                    if fits:
                        batch = Batch(
                            resolution_bin=key[0],
                            requests=fits,
                            total_vram_gb=sum(r.estimated_vram_gb for r in fits),
                        )
                        batches_to_dispatch.append(batch)

                    # Keep overflow + remaining
                    remaining = overflow + queue[self._config.max_batch_size :]
                    if remaining:
                        self._queues[key] = remaining
                    else:
                        keys_to_remove.append(key)

            for k in keys_to_remove:
                self._queues.pop(k, None)

        # Dispatch outside lock
        for batch in batches_to_dispatch:
            batch.dispatched_at = time.time()
            await self._dispatch_fn(batch)
            self._update_stats(batch)

    async def _flush_all(self) -> None:
        """Dispatch everything remaining (graceful shutdown)."""
        async with self._lock:
            all_requests: List[InferenceRequest] = []
            for queue in self._queues.values():
                all_requests.extend(queue)
            self._queues.clear()

        if all_requests:
            all_requests.sort(key=lambda r: r.priority, reverse=True)
            fits, _ = self._sizer.fit(all_requests)
            if fits:
                batch = Batch(
                    requests=fits,
                    total_vram_gb=sum(r.estimated_vram_gb for r in fits),
                )
                batch.dispatched_at = time.time()
                await self._dispatch_fn(batch)
                self._update_stats(batch)

    def _update_stats(self, batch: Batch) -> None:
        n = self._stats["total_batches"]
        self._stats["total_dispatched"] += batch.size
        self._stats["total_batches"] += 1
        # Running average batch size
        self._stats["avg_batch_size"] = (
            (self._stats["avg_batch_size"] * n + batch.size) / (n + 1)
        )
        # Running average wait
        now = time.time()
        avg_wait = sum(now - r.timestamp for r in batch.requests) / max(batch.size, 1)
        self._stats["avg_wait_sec"] = (
            (self._stats["avg_wait_sec"] * n + avg_wait) / (n + 1)
        )

    @staticmethod
    async def _default_dispatch(batch: Batch) -> None:
        """No-op dispatch for testing."""
        pass
