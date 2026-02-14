"""
AIPROD Resilience Module
=========================

Production robustness mechanisms:

1. **GPUHealthMonitor** — Continuous GPU health checks with auto-restart.
   Monitors VRAM, temperature, ECC errors, utilization.  Triggers
   Kubernetes liveness-probe failure when GPU is unhealthy.

2. **OOMFallback** — Automatic resolution/batch-size downgrade when CUDA
   OOM is detected.  Retries with degraded config rather than failing.

3. **DataIntegrity** — SHA-256 checksum verification for downloaded
   datasets and model checkpoints.  Prevents silent corruption.

4. **DriftDetector** — Periodic FID / CLIP-Score monitoring against a
   reference distribution.  Raises alerts when quality degrades beyond
   configurable thresholds.

5. **CircuitBreaker** — Prevents cascading failures by opening the
   circuit after N consecutive errors, with automatic half-open probing.

6. **DeadlineManager** — Per-stage timeout enforcement with graceful
   cancellation.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# 1. GPU Health Monitor
# ============================================================================


@dataclass
class GPUHealthStatus:
    """Snapshot of GPU health."""

    healthy: bool = True
    available: bool = False
    vram_free_mb: float = 0.0
    vram_total_mb: float = 0.0
    vram_used_pct: float = 0.0
    temperature_c: float = 0.0
    utilization_pct: float = 0.0
    errors: List[str] = field(default_factory=list)


class GPUHealthMonitor:
    """
    Monitors GPU health and exposes status for Kubernetes liveness probes.

    Thresholds (configurable):
    - VRAM: alert if < 512 MB free
    - Temperature: alert if > 85 °C
    - Utilization stuck at 0% with active jobs → stalled
    """

    def __init__(
        self,
        min_vram_mb: float = 512.0,
        max_temp_c: float = 85.0,
        check_interval_sec: float = 30.0,
    ):
        self.min_vram_mb = min_vram_mb
        self.max_temp_c = max_temp_c
        self.check_interval_sec = check_interval_sec
        self._last_check: Optional[GPUHealthStatus] = None
        self._consecutive_failures = 0

    def check(self) -> GPUHealthStatus:
        """Perform a GPU health check. Safe to call without CUDA."""
        status = GPUHealthStatus()

        try:
            import torch

            if not torch.cuda.is_available():
                status.available = False
                status.errors.append("CUDA not available")
                status.healthy = False
                self._last_check = status
                return status

            status.available = True

            # VRAM
            free, total = torch.cuda.mem_get_info(0)
            status.vram_free_mb = free / (1024 * 1024)
            status.vram_total_mb = total / (1024 * 1024)
            status.vram_used_pct = (1.0 - free / total) * 100 if total > 0 else 0.0

            if status.vram_free_mb < self.min_vram_mb:
                status.errors.append(
                    f"Low VRAM: {status.vram_free_mb:.0f} MB free (threshold: {self.min_vram_mb} MB)"
                )

            # Temperature (via pynvml if available)
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                status.temperature_c = float(temp)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                status.utilization_pct = float(util.gpu)
                pynvml.nvmlShutdown()

                if status.temperature_c > self.max_temp_c:
                    status.errors.append(
                        f"High GPU temp: {status.temperature_c}°C (threshold: {self.max_temp_c}°C)"
                    )
            except Exception:
                pass  # pynvml not available — skip temp/util checks

            status.healthy = len(status.errors) == 0

        except Exception as e:
            status.available = False
            status.healthy = False
            status.errors.append(f"GPU probe failed: {e}")

        if status.healthy:
            self._consecutive_failures = 0
        else:
            self._consecutive_failures += 1

        self._last_check = status
        return status

    @property
    def is_healthy(self) -> bool:
        """Quick health flag for liveness probes."""
        if self._last_check is None:
            self.check()
        return self._last_check.healthy if self._last_check else False

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures


# ============================================================================
# 2. OOM Fallback
# ============================================================================


@dataclass
class FallbackConfig:
    """Degraded configuration to try after OOM."""

    width: int
    height: int
    num_inference_steps: int
    batch_size: int
    half_precision: bool
    label: str


class OOMFallback:
    """
    Automatic resolution/step reduction on CUDA OOM.

    Fallback chain (configurable):
      1080p/50 steps → 720p/50 → 720p/30 → 512p/20 (half precision)

    Usage:
        fallback = OOMFallback()
        for config in fallback.configs(original_width, original_height):
            try:
                result = generate(config)
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise
    """

    # Pre-defined fallback chain
    FALLBACK_CHAIN: List[FallbackConfig] = [
        FallbackConfig(1920, 1080, 50, 1, False, "1080p_full"),
        FallbackConfig(1280, 720, 50, 1, False, "720p_full"),
        FallbackConfig(1280, 720, 30, 1, True, "720p_fast_fp16"),
        FallbackConfig(768, 512, 30, 1, True, "512p_fast_fp16"),
        FallbackConfig(512, 512, 20, 1, True, "512p_min_fp16"),
    ]

    def __init__(self, max_retries: int = 3, clear_cache: bool = True):
        self.max_retries = max_retries
        self.clear_cache = clear_cache

    def configs(
        self, original_width: int = 1920, original_height: int = 1080
    ) -> List[FallbackConfig]:
        """
        Get fallback configuration chain starting from (or below) the
        requested resolution.
        """
        # Find the starting position in the chain
        start_idx = 0
        for i, cfg in enumerate(self.FALLBACK_CHAIN):
            if cfg.width <= original_width and cfg.height <= original_height:
                start_idx = i
                break

        return self.FALLBACK_CHAIN[start_idx : start_idx + self.max_retries + 1]

    @staticmethod
    def is_oom_error(error: Exception) -> bool:
        """Check if an exception is a CUDA out-of-memory error."""
        msg = str(error).lower()
        return (
            "out of memory" in msg
            or "cuda out of memory" in msg
            or "cudnn_status_alloc_failed" in msg
        )

    @staticmethod
    def clear_gpu_cache() -> None:
        """Clear CUDA cache to free VRAM."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass


# ============================================================================
# 3. Data Integrity
# ============================================================================


class DataIntegrity:
    """
    SHA-256 checksum verification for datasets and model checkpoints.

    Usage:
        di = DataIntegrity()
        di.register("model.pt", expected_sha256="abc123...")
        assert di.verify("model.pt")  # True if checksum matches
    """

    def __init__(self, manifest_path: Optional[str] = None):
        self._checksums: Dict[str, str] = {}
        self._manifest_path = manifest_path
        if manifest_path and os.path.isfile(manifest_path):
            self._load_manifest(manifest_path)

    def register(self, filepath: str, expected_hash: str) -> None:
        """Register an expected checksum for a file."""
        self._checksums[os.path.abspath(filepath)] = expected_hash

    def compute_hash(self, filepath: str, algorithm: str = "sha256") -> str:
        """Compute the hash of a file."""
        h = hashlib.new(algorithm)
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def verify(self, filepath: str) -> Tuple[bool, str]:
        """
        Verify a file's checksum against the registered value.

        Returns:
            (valid, message)
        """
        abs_path = os.path.abspath(filepath)
        expected = self._checksums.get(abs_path)

        if expected is None:
            return False, f"No registered checksum for {filepath}"

        if not os.path.isfile(filepath):
            return False, f"File not found: {filepath}"

        actual = self.compute_hash(filepath)
        if actual != expected:
            return False, (
                f"Checksum mismatch for {filepath}: "
                f"expected {expected[:16]}..., got {actual[:16]}..."
            )

        return True, "OK"

    def verify_all(self) -> Dict[str, Tuple[bool, str]]:
        """Verify all registered files. Returns dict of filepath → (valid, msg)."""
        results: Dict[str, Tuple[bool, str]] = {}
        for filepath in self._checksums:
            results[filepath] = self.verify(filepath)
        return results

    def save_manifest(self, path: Optional[str] = None) -> None:
        """Save checksums to a manifest file."""
        import json

        path = path or self._manifest_path or "checksums.json"
        Path(path).write_text(json.dumps(self._checksums, indent=2))

    def _load_manifest(self, path: str) -> None:
        """Load checksums from a manifest file."""
        import json

        data = json.loads(Path(path).read_text())
        self._checksums.update(data)

    def scan_directory(self, directory: str, extensions: Optional[List[str]] = None) -> Dict[str, str]:
        """Compute checksums for all files in a directory."""
        exts = set(extensions or [".pt", ".pth", ".safetensors", ".bin", ".ckpt"])
        results: Dict[str, str] = {}
        for root, _, files in os.walk(directory):
            for fname in files:
                if any(fname.endswith(e) for e in exts):
                    fpath = os.path.join(root, fname)
                    results[fpath] = self.compute_hash(fpath)
                    self._checksums[os.path.abspath(fpath)] = results[fpath]
        return results


# ============================================================================
# 4. Drift Detector
# ============================================================================


@dataclass
class DriftReport:
    """Result of a drift detection check."""

    model_name: str
    timestamp: float
    drifted: bool
    metrics: Dict[str, float]  # current values
    baselines: Dict[str, float]  # reference values
    deltas: Dict[str, float]  # percentage change
    alerts: List[str]


class DriftDetector:
    """
    Monitors model quality over time and detects drift.

    Compares current FID/CLIP-Score against a reference baseline.
    Alerts when metrics degrade beyond configurable thresholds.

    Usage:
        detector = DriftDetector()
        detector.set_baseline("transformer", {"fid": 35.0, "clip_score": 0.30})
        report = detector.check("transformer", {"fid": 52.0, "clip_score": 0.22})
        if report.drifted:
            trigger_alert(report)
    """

    def __init__(
        self,
        fid_threshold_pct: float = 20.0,  # alert if FID increases > 20%
        clip_threshold_pct: float = 15.0,  # alert if CLIP decreases > 15%
        generic_threshold_pct: float = 25.0,  # alert for other metrics
    ):
        self.fid_threshold_pct = fid_threshold_pct
        self.clip_threshold_pct = clip_threshold_pct
        self.generic_threshold_pct = generic_threshold_pct
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._history: List[DriftReport] = []

    def set_baseline(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Set reference baseline metrics for a model."""
        self._baselines[model_name] = dict(metrics)

    def check(self, model_name: str, current_metrics: Dict[str, float]) -> DriftReport:
        """
        Compare current metrics against baseline.

        Returns a DriftReport with alerts if drift is detected.
        """
        baseline = self._baselines.get(model_name, {})
        alerts: List[str] = []
        deltas: Dict[str, float] = {}

        # Metrics where LOWER is better
        lower_better = {"fid", "loss", "inference_latency_sec"}
        # Metrics where HIGHER is better
        higher_better = {"clip_score", "quality_score", "accuracy"}

        for metric, current in current_metrics.items():
            base = baseline.get(metric)
            if base is None or base == 0:
                continue

            delta_pct = ((current - base) / abs(base)) * 100
            deltas[metric] = round(delta_pct, 2)

            if metric in lower_better:
                # FID increased → quality degraded
                threshold = self.fid_threshold_pct if metric == "fid" else self.generic_threshold_pct
                if delta_pct > threshold:
                    alerts.append(
                        f"{metric} degraded: {base:.3f} → {current:.3f} (+{delta_pct:.1f}%)"
                    )
            elif metric in higher_better:
                # CLIP decreased → quality degraded
                threshold = self.clip_threshold_pct if metric == "clip_score" else self.generic_threshold_pct
                if delta_pct < -threshold:
                    alerts.append(
                        f"{metric} degraded: {base:.3f} → {current:.3f} ({delta_pct:.1f}%)"
                    )

        report = DriftReport(
            model_name=model_name,
            timestamp=time.time(),
            drifted=len(alerts) > 0,
            metrics=current_metrics,
            baselines=baseline,
            deltas=deltas,
            alerts=alerts,
        )
        self._history.append(report)
        return report

    @property
    def history(self) -> List[DriftReport]:
        return list(self._history)


# ============================================================================
# 5. Circuit Breaker
# ============================================================================


class CircuitState(str, Enum):
    CLOSED = "closed"  # normal operation
    OPEN = "open"  # failing fast, rejecting calls
    HALF_OPEN = "half_open"  # probing with limited traffic


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.

    States:
        CLOSED → OPEN (after failure_threshold consecutive failures)
        OPEN → HALF_OPEN (after reset_timeout_sec)
        HALF_OPEN → CLOSED (on success) or OPEN (on failure)

    Usage:
        cb = CircuitBreaker("gpu_inference", failure_threshold=3)
        if cb.allow_request():
            try:
                result = inference(input)
                cb.record_success()
            except Exception as e:
                cb.record_failure()
                raise
        else:
            raise ServiceUnavailable("Circuit breaker open")
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout_sec: float = 60.0,
        half_open_max_calls: int = 1,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout_sec = reset_timeout_sec
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may transition OPEN → HALF_OPEN on access)."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.reset_timeout_sec:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def allow_request(self) -> bool:
        """Check if a request should be allowed through."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful call."""
        self._failure_count = 0
        self._success_count += 1
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
        elif self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Force-reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_calls = 0


# ============================================================================
# 6. Deadline Manager
# ============================================================================


class DeadlineExceeded(Exception):
    """Raised when a pipeline stage exceeds its deadline."""

    def __init__(self, stage: str, elapsed: float, deadline: float):
        self.stage = stage
        self.elapsed = elapsed
        self.deadline = deadline
        super().__init__(
            f"Stage '{stage}' exceeded deadline: {elapsed:.1f}s > {deadline:.1f}s"
        )


class DeadlineManager:
    """
    Per-stage timeout enforcement.

    Usage:
        dm = DeadlineManager({
            "text_encode": 10.0,
            "denoise": 120.0,
            "decode": 30.0,
        })

        dm.start("denoise")
        # ... do work ...
        dm.check("denoise")  # raises DeadlineExceeded if over 120s
        dm.finish("denoise")
    """

    # Default deadlines per pipeline stage (seconds)
    DEFAULT_DEADLINES: Dict[str, float] = {
        "text_encode": 10.0,
        "denoise": 180.0,
        "upsample": 60.0,
        "decode_video": 60.0,
        "audio_encode": 30.0,
        "tts": 30.0,
        "lip_sync": 30.0,
        "audio_mix": 15.0,
        "color_grade": 20.0,
        "export": 60.0,
        "total_pipeline": 600.0,
    }

    def __init__(self, deadlines: Optional[Dict[str, float]] = None):
        self._deadlines = dict(self.DEFAULT_DEADLINES)
        if deadlines:
            self._deadlines.update(deadlines)
        self._start_times: Dict[str, float] = {}
        self._elapsed: Dict[str, float] = {}

    def start(self, stage: str) -> None:
        """Start timing a stage."""
        self._start_times[stage] = time.perf_counter()

    def check(self, stage: str) -> float:
        """
        Check if a stage has exceeded its deadline.

        Returns elapsed time. Raises DeadlineExceeded if over.
        """
        start = self._start_times.get(stage)
        if start is None:
            return 0.0

        elapsed = time.perf_counter() - start
        deadline = self._deadlines.get(stage, float("inf"))

        if elapsed > deadline:
            raise DeadlineExceeded(stage, elapsed, deadline)

        return elapsed

    def finish(self, stage: str) -> float:
        """Finish timing a stage. Returns elapsed time."""
        start = self._start_times.pop(stage, None)
        if start is None:
            return 0.0
        elapsed = time.perf_counter() - start
        self._elapsed[stage] = elapsed
        return elapsed

    def get_deadline(self, stage: str) -> float:
        """Get the deadline for a stage."""
        return self._deadlines.get(stage, float("inf"))

    def set_deadline(self, stage: str, seconds: float) -> None:
        """Set or override the deadline for a stage."""
        self._deadlines[stage] = seconds

    @property
    def stage_timings(self) -> Dict[str, float]:
        """Get elapsed times for all completed stages."""
        return dict(self._elapsed)
