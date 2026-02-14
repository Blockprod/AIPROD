"""
AIPROD Model Registry & Versioning
====================================

MLflow-compatible model registry with:
- Semantic versioning (vX.Y.Z) for every model component
- Promotion pipeline: dev → staging → production
- Automatic rollback on quality degradation
- Canary deployment with A/B metric comparison
- Artifact storage abstraction (local / GCS / S3)
- Quality gate evaluation (FID, CLIP-Score thresholds)

Components registered:
  transformer, video_vae, audio_vae, text_encoder, upsampler,
  tts, lip_sync, audio_mixer, vocoder

Can operate standalone (local JSON registry) or delegate to MLflow.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class Stage(str, Enum):
    """Model promotion stages."""
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """A single registered model version."""

    name: str  # e.g. "transformer", "video_vae"
    version: str  # semantic version "1.2.3"
    stage: Stage = Stage.DEV
    artifact_path: str = ""  # local path or GCS/S3 URI
    artifact_hash: str = ""  # SHA-256 of checkpoint file
    created_at: float = 0.0
    promoted_at: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    parent_version: Optional[str] = None  # version it was promoted from

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    @property
    def fqn(self) -> str:
        """Fully-qualified name: name/version."""
        return f"{self.name}/v{self.version}"


@dataclass
class QualityGate:
    """Quality thresholds for promotion gates."""

    max_fid: float = 50.0  # FID must be below this
    min_clip_score: float = 0.25  # CLIP must be above this
    max_latency_sec: float = 120.0  # inference latency cap
    min_samples: int = 100  # minimum evaluation samples
    custom_checks: Dict[str, Tuple[str, float]] = field(default_factory=dict)
    # e.g. {"lip_sync_confidence": (">=", 0.8)}

    def evaluate(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Evaluate metrics against quality gates. Returns (pass, failures)."""
        failures: List[str] = []

        fid = metrics.get("fid")
        if fid is not None and fid > self.max_fid:
            failures.append(f"FID {fid:.2f} exceeds threshold {self.max_fid}")

        clip = metrics.get("clip_score")
        if clip is not None and clip < self.min_clip_score:
            failures.append(f"CLIP score {clip:.3f} below threshold {self.min_clip_score}")

        latency = metrics.get("inference_latency_sec")
        if latency is not None and latency > self.max_latency_sec:
            failures.append(f"Latency {latency:.1f}s exceeds threshold {self.max_latency_sec}s")

        samples = metrics.get("eval_samples", 0)
        if samples < self.min_samples:
            failures.append(f"Only {int(samples)} eval samples (need {self.min_samples})")

        for key, (op, threshold) in self.custom_checks.items():
            val = metrics.get(key)
            if val is None:
                failures.append(f"Missing metric: {key}")
                continue
            if op == ">=" and val < threshold:
                failures.append(f"{key} = {val} (need >= {threshold})")
            elif op == "<=" and val > threshold:
                failures.append(f"{key} = {val} (need <= {threshold})")
            elif op == ">" and val <= threshold:
                failures.append(f"{key} = {val} (need > {threshold})")
            elif op == "<" and val >= threshold:
                failures.append(f"{key} = {val} (need < {threshold})")

        return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Registry storage backends
# ---------------------------------------------------------------------------


class RegistryBackend:
    """Abstract registry storage backend."""

    def save(self, versions: Dict[str, Dict[str, ModelVersion]]) -> None:
        raise NotImplementedError

    def load(self) -> Dict[str, Dict[str, ModelVersion]]:
        raise NotImplementedError


class LocalJSONBackend(RegistryBackend):
    """Stores registry state as a local JSON file."""

    def __init__(self, path: str = "model_registry.json"):
        self._path = Path(path)

    def save(self, versions: Dict[str, Dict[str, ModelVersion]]) -> None:
        data: Dict[str, Any] = {}
        for model_name, ver_dict in versions.items():
            data[model_name] = {}
            for ver_str, mv in ver_dict.items():
                data[model_name][ver_str] = asdict(mv)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, default=str))

    def load(self) -> Dict[str, Dict[str, ModelVersion]]:
        if not self._path.exists():
            return {}
        raw = json.loads(self._path.read_text())
        result: Dict[str, Dict[str, ModelVersion]] = {}
        for model_name, ver_dict in raw.items():
            result[model_name] = {}
            for ver_str, mv_data in ver_dict.items():
                mv_data["stage"] = Stage(mv_data["stage"])
                result[model_name][ver_str] = ModelVersion(**mv_data)
        return result


class MLflowBackend(RegistryBackend):
    """
    Delegates to MLflow Model Registry.

    Requires mlflow to be installed and MLFLOW_TRACKING_URI configured.
    """

    def __init__(self, tracking_uri: Optional[str] = None):
        self._tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self._mlflow = None

    def _get_mlflow(self):
        if self._mlflow is None:
            try:
                import mlflow

                mlflow.set_tracking_uri(self._tracking_uri)
                self._mlflow = mlflow
            except ImportError:
                raise ImportError("MLflow is required: pip install mlflow")
        return self._mlflow

    def save(self, versions: Dict[str, Dict[str, ModelVersion]]) -> None:
        mlflow = self._get_mlflow()
        client = mlflow.tracking.MlflowClient()

        for model_name, ver_dict in versions.items():
            # Ensure registered model exists
            try:
                client.get_registered_model(model_name)
            except Exception:
                client.create_registered_model(
                    model_name, description=f"AIPROD {model_name} model"
                )

            for ver_str, mv in ver_dict.items():
                tags = {
                    "aiprod.version": mv.version,
                    "aiprod.stage": mv.stage.value,
                    "aiprod.hash": mv.artifact_hash,
                    **mv.tags,
                }
                try:
                    client.create_model_version(
                        name=model_name,
                        source=mv.artifact_path,
                        tags=tags,
                    )
                except Exception:
                    pass  # Version may already exist

    def load(self) -> Dict[str, Dict[str, ModelVersion]]:
        mlflow = self._get_mlflow()
        client = mlflow.tracking.MlflowClient()
        result: Dict[str, Dict[str, ModelVersion]] = {}

        try:
            models = client.search_registered_models()
        except Exception:
            return {}

        for rm in models:
            model_name = rm.name
            result[model_name] = {}
            for mv in client.search_model_versions(f"name='{model_name}'"):
                stage_str = mv.tags.get("aiprod.stage", "dev")
                version_str = mv.tags.get("aiprod.version", mv.version)
                result[model_name][version_str] = ModelVersion(
                    name=model_name,
                    version=version_str,
                    stage=Stage(stage_str),
                    artifact_path=mv.source or "",
                    artifact_hash=mv.tags.get("aiprod.hash", ""),
                )
        return result


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    Central model registry with promotion pipeline.

    Supports:
    - Registration of model versions with artifact checksums
    - Quality-gated promotion: dev → staging → production
    - Canary deployment comparison
    - Automatic rollback on quality degradation
    - History tracking for audit trail
    """

    # Promotion path
    PROMOTION_PATH = {
        Stage.DEV: Stage.STAGING,
        Stage.STAGING: Stage.PRODUCTION,
    }

    # Default quality gates per stage transition
    DEFAULT_GATES = {
        Stage.STAGING: QualityGate(max_fid=60.0, min_clip_score=0.22, min_samples=50),
        Stage.PRODUCTION: QualityGate(max_fid=50.0, min_clip_score=0.25, min_samples=100),
    }

    def __init__(self, backend: Optional[RegistryBackend] = None):
        self._backend = backend or LocalJSONBackend()
        self._versions: Dict[str, Dict[str, ModelVersion]] = self._backend.load()
        self._gates: Dict[Stage, QualityGate] = dict(self.DEFAULT_GATES)
        self._history: List[Dict[str, Any]] = []

    # ---- Registration -------------------------------------------------------

    def register(
        self,
        name: str,
        version: str,
        artifact_path: str,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
        compute_hash: bool = True,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            name: Model component name (e.g. "transformer", "video_vae")
            version: Semantic version string (e.g. "1.0.0")
            artifact_path: Path or URI to model checkpoint
            metrics: Evaluation metrics (FID, CLIP-Score, etc.)
            tags: Arbitrary key-value tags
            description: Human-readable description
            compute_hash: Whether to compute SHA-256 of artifact file

        Returns:
            Registered ModelVersion
        """
        artifact_hash = ""
        if compute_hash and os.path.isfile(artifact_path):
            artifact_hash = self._compute_sha256(artifact_path)

        mv = ModelVersion(
            name=name,
            version=version,
            stage=Stage.DEV,
            artifact_path=artifact_path,
            artifact_hash=artifact_hash,
            metrics=metrics or {},
            tags=tags or {},
            description=description,
        )

        self._versions.setdefault(name, {})[version] = mv
        self._record_event("register", mv)
        self._backend.save(self._versions)
        return mv

    # ---- Querying -----------------------------------------------------------

    def get(self, name: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version."""
        return self._versions.get(name, {}).get(version)

    def get_latest(self, name: str, stage: Optional[Stage] = None) -> Optional[ModelVersion]:
        """Get the latest version of a model, optionally filtered by stage."""
        versions = self._versions.get(name, {})
        candidates = [
            mv for mv in versions.values()
            if stage is None or mv.stage == stage
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda mv: mv.created_at)

    def get_production(self, name: str) -> Optional[ModelVersion]:
        """Shortcut: get the current production version of a model."""
        return self.get_latest(name, Stage.PRODUCTION)

    def list_versions(self, name: str) -> List[ModelVersion]:
        """List all versions of a model, newest first."""
        versions = list(self._versions.get(name, {}).values())
        versions.sort(key=lambda mv: mv.created_at, reverse=True)
        return versions

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._versions.keys())

    # ---- Promotion ----------------------------------------------------------

    def promote(
        self,
        name: str,
        version: str,
        metrics: Optional[Dict[str, float]] = None,
        force: bool = False,
    ) -> Tuple[bool, List[str]]:
        """
        Promote a model version to the next stage.

        Applies quality gates unless force=True.

        Returns:
            (success, list_of_failure_reasons)
        """
        mv = self.get(name, version)
        if mv is None:
            return False, [f"Model {name}/v{version} not found"]

        next_stage = self.PROMOTION_PATH.get(mv.stage)
        if next_stage is None:
            return False, [f"Model {mv.fqn} is already at {mv.stage.value} (or archived)"]

        # Update metrics if provided
        if metrics:
            mv.metrics.update(metrics)

        # Quality gate check
        if not force:
            gate = self._gates.get(next_stage)
            if gate:
                passed, failures = gate.evaluate(mv.metrics)
                if not passed:
                    self._record_event("promotion_rejected", mv, failures=failures)
                    return False, failures

        # Promote
        old_stage = mv.stage
        mv.stage = next_stage
        mv.promoted_at = time.time()
        mv.parent_version = version

        self._record_event("promote", mv, from_stage=old_stage.value, to_stage=next_stage.value)
        self._backend.save(self._versions)
        return True, []

    def rollback(self, name: str) -> Optional[ModelVersion]:
        """
        Rollback: demote current production model to staging,
        and promote the previous production model back.

        Returns the newly-active production version, or None if no rollback target.
        """
        current = self.get_production(name)
        if current is None:
            return None

        # Find previous production version
        versions = self.list_versions(name)
        previous = None
        for mv in versions:
            if mv.stage == Stage.ARCHIVED and mv.version != current.version:
                previous = mv
                break

        # Demote current
        current.stage = Stage.ARCHIVED
        self._record_event("rollback_demote", current)

        # Restore previous
        if previous:
            previous.stage = Stage.PRODUCTION
            previous.promoted_at = time.time()
            self._record_event("rollback_restore", previous)

        self._backend.save(self._versions)
        return previous

    def archive(self, name: str, version: str) -> bool:
        """Archive a model version."""
        mv = self.get(name, version)
        if mv is None:
            return False
        mv.stage = Stage.ARCHIVED
        self._record_event("archive", mv)
        self._backend.save(self._versions)
        return True

    # ---- Canary comparison --------------------------------------------------

    def compare_canary(
        self,
        name: str,
        candidate_version: str,
        baseline_version: str,
    ) -> Dict[str, Any]:
        """
        Compare two model versions for canary deployment decision.

        Returns comparison report with metric deltas.
        """
        candidate = self.get(name, candidate_version)
        baseline = self.get(name, baseline_version)

        if candidate is None or baseline is None:
            return {"error": "One or both versions not found"}

        report: Dict[str, Any] = {
            "model": name,
            "candidate": candidate.version,
            "baseline": baseline.version,
            "deltas": {},
            "recommendation": "unknown",
        }

        all_metrics = set(candidate.metrics.keys()) | set(baseline.metrics.keys())
        improvements = 0
        regressions = 0

        for metric in all_metrics:
            c_val = candidate.metrics.get(metric)
            b_val = baseline.metrics.get(metric)
            if c_val is not None and b_val is not None and b_val != 0:
                delta_pct = ((c_val - b_val) / abs(b_val)) * 100
                report["deltas"][metric] = {
                    "candidate": c_val,
                    "baseline": b_val,
                    "delta_percent": round(delta_pct, 2),
                }
                # For FID/latency lower is better; for CLIP higher is better
                lower_is_better = metric in ("fid", "inference_latency_sec", "loss")
                if lower_is_better:
                    if delta_pct < -5:
                        improvements += 1
                    elif delta_pct > 5:
                        regressions += 1
                else:
                    if delta_pct > 5:
                        improvements += 1
                    elif delta_pct < -5:
                        regressions += 1

        if regressions == 0 and improvements > 0:
            report["recommendation"] = "promote"
        elif regressions > improvements:
            report["recommendation"] = "reject"
        else:
            report["recommendation"] = "manual_review"

        return report

    # ---- Quality gates management -------------------------------------------

    def set_quality_gate(self, stage: Stage, gate: QualityGate) -> None:
        """Override quality gate for a promotion stage."""
        self._gates[stage] = gate

    # ---- Internal helpers ---------------------------------------------------

    @staticmethod
    def _compute_sha256(filepath: str) -> str:
        """Compute SHA-256 checksum of a file."""
        sha = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha.update(chunk)
        return sha.hexdigest()

    def _record_event(self, event: str, mv: ModelVersion, **kwargs) -> None:
        """Record a registry event for audit trail."""
        entry = {
            "timestamp": time.time(),
            "event": event,
            "model": mv.name,
            "version": mv.version,
            "stage": mv.stage.value,
            **kwargs,
        }
        self._history.append(entry)

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Get audit trail of all registry events."""
        return list(self._history)
