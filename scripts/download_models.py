#!/usr/bin/env python3
"""
AIPROD — Model Provisioning Script

Downloads all required pre-trained models from HuggingFace Hub to local storage.
All models are open-source (Apache 2.0 or MIT) and used as base weights
for AIPROD's sovereign fine-tuning pipeline.

After download, all inference and training code uses `local_files_only=True`
to ensure zero network dependency at runtime.

Usage:
    python scripts/download_models.py                  # Download all models
    python scripts/download_models.py --model text-encoder  # Download one model
    python scripts/download_models.py --list            # List models and status
    python scripts/download_models.py --verify          # Verify checksums

Requirements:
    pip install huggingface_hub
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("aiprod.download_models")

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CHECKSUMS_FILE = MODELS_DIR / "CHECKSUMS.sha256"

# ── Model registry ───────────────────────────────────────────────────────────
MODELS: list[dict] = [
    {
        "id": "text-encoder",
        "repo": "google/gemma-3-1b-pt",
        "destination": "models/text-encoder",
        "description": "Text encoder base weights (Apache 2.0)",
        "estimated_size_gb": 2.0,
        "license": "Apache-2.0",
        "usage": "aiprod_core.model.text_encoder.LLMBridge",
    },
    {
        "id": "scenarist",
        "repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "destination": "models/scenarist/mistral-7b",
        "description": "Scenarist LLM for storyboard generation (Apache 2.0)",
        "estimated_size_gb": 14.0,
        "license": "Apache-2.0",
        "usage": "aiprod_pipelines.inference.scenarist.ScenaristLLM",
    },
    {
        "id": "clip",
        "repo": "openai/clip-vit-large-patch14",
        "destination": "models/clip",
        "description": "CLIP ViT-L/14 for semantic QA scoring (MIT)",
        "estimated_size_gb": 1.7,
        "license": "MIT",
        "usage": "aiprod_pipelines.api.qa_semantic_local",
    },
    {
        "id": "captioning",
        "repo": "Qwen/Qwen2.5-Omni-7B",
        "destination": "models/captioning/qwen-omni-7b",
        "description": "Audio-visual captioning model (Apache 2.0)",
        "estimated_size_gb": 15.0,
        "license": "Apache-2.0",
        "usage": "aiprod_trainer.captioning.QwenOmniCaptioner",
    },
]


def _resolve_dest(model: dict) -> Path:
    """Resolve model destination as absolute path."""
    return PROJECT_ROOT / model["destination"]


def _is_provisioned(model: dict) -> bool:
    """Check if a model directory has actual weight files."""
    dest = _resolve_dest(model)
    if not dest.is_dir():
        return False
    # A model is "provisioned" if there's at least one .safetensors, .bin, or .model file
    weight_exts = {".safetensors", ".bin", ".model", ".gguf", ".pt"}
    return any(f.suffix in weight_exts for f in dest.rglob("*") if f.is_file())


def list_models() -> None:
    """Print status of all models."""
    total_gb = 0.0
    print("\n  AIPROD Model Provisioning Status")
    print("  " + "=" * 60)
    for m in MODELS:
        provisioned = _is_provisioned(m)
        status = "✅ Provisioned" if provisioned else "⬜ Not downloaded"
        print(f"  [{m['id']:15s}] {status:18s}  ~{m['estimated_size_gb']:.1f} GB  ({m['license']})")
        if not provisioned:
            total_gb += m["estimated_size_gb"]
    print("  " + "-" * 60)
    if total_gb > 0:
        print(f"  Total to download: ~{total_gb:.1f} GB")
    else:
        print("  All models provisioned ✅")
    print()


def download_model(model: dict, *, force: bool = False) -> bool:
    """Download a single model from HuggingFace Hub."""
    dest = _resolve_dest(model)

    if _is_provisioned(model) and not force:
        logger.info("%-15s already provisioned at %s — skipping", model["id"], dest)
        return True

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        )
        return False

    dest.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Downloading %s (~%.1f GB) from %s → %s",
        model["id"],
        model["estimated_size_gb"],
        model["repo"],
        dest,
    )

    try:
        snapshot_download(
            repo_id=model["repo"],
            local_dir=str(dest),
            local_dir_use_symlinks=False,
        )
        logger.info("✅ %s downloaded successfully", model["id"])
        return True
    except Exception as e:
        logger.error("❌ Failed to download %s: %s", model["id"], e)
        return False


def download_all(*, force: bool = False) -> bool:
    """Download all models. Returns True if all succeeded."""
    results = []
    for model in MODELS:
        ok = download_model(model, force=force)
        results.append(ok)
    return all(results)


def compute_checksums() -> None:
    """Compute SHA-256 checksums for all provisioned model files."""
    lines: list[str] = []

    # Keep existing checksums for non-model files
    if CHECKSUMS_FILE.exists():
        for line in CHECKSUMS_FILE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            # Keep lines not covered by our model directories
            parts = line.split("  ", 1)
            if len(parts) == 2:
                fpath = parts[1]
                is_model_file = any(
                    fpath.startswith(m["destination"].replace("/", "\\"))
                    or fpath.startswith(m["destination"])
                    for m in MODELS
                )
                if not is_model_file:
                    lines.append(line)

    # Compute new checksums for model weight files
    weight_exts = {".safetensors", ".bin", ".model", ".gguf", ".pt"}
    for model in MODELS:
        dest = _resolve_dest(model)
        if not dest.is_dir():
            continue
        for f in sorted(dest.rglob("*")):
            if f.is_file() and f.suffix in weight_exts:
                rel = f.relative_to(PROJECT_ROOT)
                logger.info("Computing SHA-256 for %s ...", rel)
                sha = hashlib.sha256(f.read_bytes()).hexdigest()
                lines.append(f"{sha}  {rel}")

    CHECKSUMS_FILE.write_text("\n".join(lines) + "\n")
    logger.info("Checksums written to %s (%d entries)", CHECKSUMS_FILE, len(lines))


def verify_checksums() -> bool:
    """Verify all checksums in CHECKSUMS.sha256."""
    if not CHECKSUMS_FILE.exists():
        logger.error("No checksums file found at %s", CHECKSUMS_FILE)
        return False

    ok_count = 0
    fail_count = 0
    for line in CHECKSUMS_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2:
            continue
        expected_sha, rel_path = parts
        full_path = PROJECT_ROOT / rel_path
        if not full_path.exists():
            logger.warning("MISSING: %s", rel_path)
            fail_count += 1
            continue
        actual_sha = hashlib.sha256(full_path.read_bytes()).hexdigest()
        if actual_sha == expected_sha:
            logger.info("OK: %s", rel_path)
            ok_count += 1
        else:
            logger.error("MISMATCH: %s (expected %s, got %s)", rel_path, expected_sha[:16], actual_sha[:16])
            fail_count += 1

    print(f"\nVerification: {ok_count} OK, {fail_count} failed")
    return fail_count == 0


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="AIPROD — Download and manage pre-trained model weights"
    )
    parser.add_argument(
        "--model",
        choices=[m["id"] for m in MODELS],
        help="Download a specific model only",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_models",
        help="List all models and their provisioning status",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify SHA-256 checksums of downloaded models",
    )
    parser.add_argument(
        "--checksums", action="store_true",
        help="Compute SHA-256 checksums for all provisioned models",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if model already exists",
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    if args.verify:
        ok = verify_checksums()
        sys.exit(0 if ok else 1)

    if args.checksums:
        compute_checksums()
        return

    if args.model:
        model = next(m for m in MODELS if m["id"] == args.model)
        ok = download_model(model, force=args.force)
        if ok:
            compute_checksums()
        sys.exit(0 if ok else 1)

    # Download all
    ok = download_all(force=args.force)
    if ok:
        compute_checksums()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
