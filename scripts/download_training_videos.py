#!/usr/bin/env python
"""
Download free training videos from Pexels API for AIPROD training.

Pexels provides royalty-free videos under the Pexels License
(free for commercial use, no attribution required).

Usage:
    # Download 100 videos (requires PEXELS_API_KEY env var)
    python scripts/download_training_videos.py --num-videos 100

    # Generate dummy data without downloading (for pipeline testing)
    python scripts/download_training_videos.py --dummy --num-videos 50

    # Full pipeline: download + preprocess
    python scripts/download_training_videos.py --num-videos 500 --preprocess

Get a free API key at: https://www.pexels.com/api/

Environment:
    PEXELS_API_KEY: Your Pexels API key (free tier: 200 requests/hour)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen, urlretrieve

import typer
from rich.console import Console
from rich.progress import track

console = Console()
logger = logging.getLogger(__name__)

app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Download free training videos for AIPROD.",
)

# Diverse query terms for varied training data
DEFAULT_QUERIES = [
    "nature landscape",
    "city timelapse",
    "ocean waves",
    "forest walk",
    "sunset sky",
    "people walking street",
    "aerial drone footage",
    "rain window",
    "cooking food",
    "cat dog pet",
    "flowers blooming",
    "traffic cars",
    "snow mountains",
    "underwater fish",
    "abstract motion",
    "studio portrait",
    "sports action",
    "fireworks night",
    "waterfall river",
    "clouds sky timelapse",
]


def pexels_search(
    api_key: str,
    query: str,
    per_page: int = 15,
    page: int = 1,
    min_width: int = 512,
    min_duration: int = 3,
    max_duration: int = 15,
) -> list[dict]:
    """Search Pexels video API.

    Returns list of dicts with keys: id, url, width, height, duration
    """
    url = (
        f"https://api.pexels.com/videos/search"
        f"?query={query.replace(' ', '+')}"
        f"&per_page={per_page}"
        f"&page={page}"
        f"&min_width={min_width}"
        f"&min_duration={min_duration}"
        f"&max_duration={max_duration}"
    )

    req = Request(url, headers={"Authorization": api_key})

    try:
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        logger.warning(f"Pexels API error for '{query}': {e}")
        return []

    results = []
    for video in data.get("videos", []):
        # Get best quality video file ≤ 720p
        best_file = None
        for vf in video.get("video_files", []):
            w = vf.get("width", 0)
            h = vf.get("height", 0)
            if w >= min_width and h >= min_width and w <= 1280:
                if best_file is None or w > best_file["width"]:
                    best_file = {
                        "download_url": vf["link"],
                        "width": w,
                        "height": h,
                        "file_type": vf.get("file_type", "video/mp4"),
                    }

        if best_file:
            results.append(
                {
                    "id": video["id"],
                    "download_url": best_file["download_url"],
                    "width": best_file["width"],
                    "height": best_file["height"],
                    "duration": video.get("duration", 0),
                    "query": query,
                }
            )

    return results


def download_video(url: str, output_path: Path, timeout: int = 120) -> bool:
    """Download a video file. Returns True on success."""
    try:
        req = Request(url, headers={"User-Agent": "AIPROD-Trainer/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
        return True
    except Exception as e:
        logger.warning(f"Download failed: {url} → {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def generate_dummy_videos(
    output_dir: Path,
    num_videos: int = 50,
    width: int = 512,
    height: int = 512,
    num_frames: int = 25,
    fps: int = 24,
) -> Path:
    """Generate dummy video data for pipeline testing.

    Creates .pt files with random tensors simulating preprocessed video data.
    """
    try:
        import torch
    except ImportError:
        console.print("[red]torch is required for dummy data generation[/red]")
        raise typer.Exit(code=1)

    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Also create preprocessed data directly
    precomputed = output_dir / "preprocessed" / ".precomputed"
    latents_dir = precomputed / "latents"
    conditions_dir = precomputed / "conditions"
    latents_dir.mkdir(parents=True, exist_ok=True)
    conditions_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"🔧 Generating {num_videos} dummy samples...")

    metadata = []
    for i in track(range(num_videos), description="Generating..."):
        # Latent dimensions (after VAE encoding)
        latent_c = 128
        latent_f = (num_frames - 1) // 8 + 1  # temporal compression
        latent_h = height // 32  # spatial compression
        latent_w = width // 32

        # Save latent
        latent_data = {
            "latents": torch.randn(latent_c, latent_f, latent_h, latent_w),
            "num_frames": latent_f,
            "height": latent_h,
            "width": latent_w,
            "fps": fps,
        }
        torch.save(latent_data, latents_dir / f"latent_{i:04d}.pt")

        # Save condition (text embedding)
        condition_data = {
            "prompt_embeds": torch.randn(256, 4096),
            "prompt_attention_mask": torch.ones(256, dtype=torch.bool),
        }
        torch.save(condition_data, conditions_dir / f"condition_{i:04d}.pt")

        metadata.append(
            {
                "id": i,
                "media_path": f"dummy_{i:04d}.mp4",
                "caption": f"A random test video sample {i}",
                "width": width,
                "height": height,
                "duration": num_frames / fps,
            }
        )

    # Save metadata
    meta_path = output_dir / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
        writer.writeheader()
        writer.writerows(metadata)

    console.print(f"✅ Dummy data generated:")
    console.print(f"   Latents: {latents_dir} ({num_videos} files)")
    console.print(f"   Conditions: {conditions_dir} ({num_videos} files)")
    console.print(f"   Metadata: {meta_path}")

    return output_dir


@app.command()
def main(
    num_videos: int = typer.Option(100, "--num-videos", "-n", help="Number of videos to download"),
    output_dir: str = typer.Option("data/training_videos", "--output", "-o", help="Output directory"),
    queries: Optional[list[str]] = typer.Option(None, "--query", "-q", help="Search queries (repeatable)"),
    dummy: bool = typer.Option(False, "--dummy", help="Generate dummy data instead of downloading"),
    preprocess: bool = typer.Option(
        False, "--preprocess", help="Run preprocessing pipeline after download"
    ),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Pexels API key (or set PEXELS_API_KEY env)"),
    resolution: int = typer.Option(512, "--resolution", "-r", help="Minimum video resolution"),
) -> None:
    """Download free training videos from Pexels."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if dummy:
        # Generate dummy data
        generate_dummy_videos(output_path, num_videos=num_videos, width=resolution, height=resolution)
        return

    # Get API key
    key = api_key or os.environ.get("PEXELS_API_KEY")
    if not key:
        console.print("[red]❌ Pexels API key required.[/red]")
        console.print("  Set PEXELS_API_KEY env var or use --api-key")
        console.print("  Get a free key at: https://www.pexels.com/api/")
        console.print()
        console.print("[yellow]💡 Or use --dummy to generate test data without API key[/yellow]")
        raise typer.Exit(code=1)

    search_queries = queries or DEFAULT_QUERIES

    # Search for videos
    console.print(f"\n🔍 Searching Pexels for {num_videos} videos...")
    all_videos = []
    videos_per_query = max(1, num_videos // len(search_queries)) + 1

    for query in search_queries:
        if len(all_videos) >= num_videos:
            break

        page = 1
        while len(all_videos) < num_videos:
            results = pexels_search(
                key,
                query,
                per_page=min(videos_per_query, 15),
                page=page,
                min_width=resolution,
            )
            if not results:
                break

            for r in results:
                if r["id"] not in {v["id"] for v in all_videos}:
                    all_videos.append(r)

            page += 1
            time.sleep(0.5)  # Rate limiting (200 req/hour)

            if len(results) < 15:
                break

        console.print(f"  '{query}': found {len(results)} videos (total: {len(all_videos)})")

    all_videos = all_videos[:num_videos]
    console.print(f"\n📦 {len(all_videos)} unique videos found")

    # Download videos
    videos_dir = output_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    downloaded = 0

    for video in track(all_videos, description="Downloading"):
        ext = ".mp4"
        filename = f"pexels_{video['id']}{ext}"
        filepath = videos_dir / filename

        if filepath.exists():
            downloaded += 1
            metadata.append(
                {
                    "media_path": str(filepath),
                    "caption": f"A video of {video['query']}",
                    "width": video["width"],
                    "height": video["height"],
                    "duration": video["duration"],
                    "source": "pexels",
                    "source_id": video["id"],
                }
            )
            continue

        if download_video(video["download_url"], filepath):
            downloaded += 1
            metadata.append(
                {
                    "media_path": str(filepath),
                    "caption": f"A video of {video['query']}",
                    "width": video["width"],
                    "height": video["height"],
                    "duration": video["duration"],
                    "source": "pexels",
                    "source_id": video["id"],
                }
            )

        time.sleep(0.3)  # Be nice to the API

    console.print(f"\n✅ Downloaded {downloaded}/{len(all_videos)} videos")

    # Save metadata
    meta_path = output_path / "metadata.csv"
    if metadata:
        with open(meta_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
            writer.writeheader()
            writer.writerows(metadata)
        console.print(f"   Metadata: {meta_path}")

    # Calculate total size
    total_size = sum(f.stat().st_size for f in videos_dir.rglob("*") if f.is_file())
    console.print(f"   Total size: {total_size / 1024**3:.1f} GB")

    # Optional: run preprocessing pipeline
    if preprocess and downloaded > 0:
        console.print(f"\n🔧 Running preprocessing pipeline...")
        import subprocess
        import sys

        model_path = "models/ltx2_research/ltx-2-19b-dev-fp8.safetensors"
        if not Path(model_path).exists():
            console.print(f"[yellow]⚠️ Model not found: {model_path}[/yellow]")
            console.print(f"   Preprocessing requires the LTX-2 model.")
            console.print(f"   Download it first, then run:")
            console.print(
                f"   python packages/aiprod-trainer/scripts/process_dataset.py "
                f"--dataset-file {meta_path} --model-path {model_path}"
            )
        else:
            subprocess.run(
                [
                    sys.executable,
                    "packages/aiprod-trainer/scripts/process_dataset.py",
                    "--dataset-file",
                    str(meta_path),
                    "--caption-column",
                    "caption",
                    "--video-column",
                    "media_path",
                    "--resolution-buckets",
                    f"{resolution}x{resolution}x25",
                    "--output-dir",
                    str(output_path / "preprocessed"),
                    "--model-path",
                    model_path,
                    "--batch-size",
                    "1",
                ],
                check=True,
            )
            console.print(f"✅ Preprocessing complete!")


if __name__ == "__main__":
    app()
