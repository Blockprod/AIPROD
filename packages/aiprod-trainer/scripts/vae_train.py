#!/usr/bin/env python
"""
Train AIPROD VAE models (Video HW-VAE or Audio NAC).

Usage:
    # Video VAE training
    python scripts/vae_train.py configs/train/vae_finetune.yaml --type video

    # Audio VAE training
    python scripts/vae_train.py configs/train/audio_vae.yaml --type audio

    # With dummy data for pipeline testing
    python scripts/vae_train.py configs/train/vae_finetune.yaml --type video --dummy-data
"""

from __future__ import annotations

from pathlib import Path

import torch
import typer
import yaml
from rich.console import Console

from aiprod_trainer.vae_trainer import VAETrainerConfig

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Train AIPROD VAE models (video or audio).",
)


@app.command()
def main(
    config_path: str = typer.Argument(..., help="Path to VAE training YAML config"),
    vae_type: str = typer.Option("video", "--type", "-t", help="VAE type: 'video' or 'audio'"),
    dummy_data: bool = typer.Option(
        False, "--dummy-data", help="Use dummy random data for pipeline testing"
    ),
) -> None:
    """Train an AIPROD VAE model."""
    from torch.utils.data import DataLoader, TensorDataset

    from aiprod_trainer.vae_trainer import (
        AudioVAETrainer,
        VAETrainerConfig,
        VideoVAETrainer,
    )

    # Load config
    config_file = Path(config_path)
    if not config_file.exists():
        typer.echo(f"Error: Config file not found: {config_file}")
        raise typer.Exit(code=1)

    with open(config_file) as f:
        raw = yaml.safe_load(f)

    # Build trainer config
    train_raw = raw.get("training", {})
    trainer_config = VAETrainerConfig(
        learning_rate=float(train_raw.get("learning_rate", 1e-4)),
        weight_decay=float(train_raw.get("weight_decay", 1e-2)),
        batch_size=int(train_raw.get("batch_size", 4)),
        num_epochs=int(train_raw.get("num_epochs", train_raw.get("epochs", 10))),
        warmup_steps=int(train_raw.get("warmup_steps", 1000)),
        beta_kl=float(raw.get("loss", {}).get("beta_kl", 0.1)),
        lambda_perceptual=float(raw.get("loss", {}).get("lambda_perceptual", 1.0)),
        use_perceptual_loss=bool(raw.get("loss", {}).get("use_perceptual_loss", True)),
        lambda_spectral=float(raw.get("loss", {}).get("lambda_spectral", 1.0)),
        gradient_accumulation_steps=int(train_raw.get("gradient_accumulation_steps", 1)),
        max_grad_norm=float(train_raw.get("max_grad_norm", 1.0)),
        mixed_precision=str(train_raw.get("mixed_precision", "bf16")),
        checkpoint_dir=Path(raw.get("output", {}).get("dir", "checkpoints/vae")),
        save_interval=raw.get("checkpoints", {}).get("save_interval", 500),
        eval_interval=raw.get("validation", {}).get("eval_interval", 100),
        use_wandb=raw.get("wandb", {}).get("enabled", False),
        project_name=raw.get("wandb", {}).get("project", "aiprod-vae"),
    )

    console.print(f"\n[bold]AIPROD {vae_type.upper()} VAE Training[/bold]")
    console.print(f"  Config: {config_file}")
    console.print(f"  Epochs: {trainer_config.num_epochs}")
    console.print(f"  Batch size: {trainer_config.batch_size}")
    console.print(f"  LR: {trainer_config.learning_rate}")

    if vae_type == "video":
        _train_video_vae(raw, trainer_config, dummy_data)
    elif vae_type == "audio":
        _train_audio_vae(raw, trainer_config, dummy_data)
    else:
        typer.echo(f"Error: Unknown VAE type '{vae_type}'. Use 'video' or 'audio'.")
        raise typer.Exit(code=1)


def _train_video_vae(raw: dict, config: VAETrainerConfig, dummy_data: bool) -> None:
    """Train video VAE."""
    from torch.utils.data import DataLoader, TensorDataset

    from aiprod_trainer.vae_trainer import VideoVAETrainer

    # Build model
    model_raw = raw.get("model", {})

    try:
        from aiprod_core.model.video_vae import HWVAEConfig

        vae_config = HWVAEConfig(
            in_channels=model_raw.get("in_channels", 3),
            latent_channels=model_raw.get("latent_channels", model_raw.get("latent_dim", 64)),
            encoder_channels=tuple(model_raw.get("encoder_channels", [128, 256, 384, 512])),
            decoder_channels=tuple(model_raw.get("decoder_channels", [512, 384, 256, 128])),
            wavelet=model_raw.get("wavelet", "haar"),
            kl_weight=model_raw.get("kl_weight", 1e-5),
            perceptual_weight=model_raw.get("perceptual_weight", 1.0),
        )
        from aiprod_core.model.video_vae import HWVAEEncoder

        model = HWVAEEncoder(vae_config)
    except (ImportError, AttributeError):
        console.print("[yellow]Using placeholder model (HWVAEEncoder import failed)[/yellow]")
        # Fallback: simple autoencoder for testing
        import torch.nn as nn

        model = nn.Sequential(
            nn.Conv3d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 3, 3, padding=1),
        )

    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Model parameters: {total_params:,}")

    # Build dataloader
    if dummy_data:
        console.print("[yellow]Using DUMMY data (random tensors)[/yellow]")
        # Random video tensors: [B, C, T, H, W]
        data = torch.randn(100, 3, 16, 128, 128)
        dataset = TensorDataset(data)
        train_dl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    else:
        data_path = raw.get("data", {}).get("path", "data/training_videos/videos")
        if not Path(data_path).exists():
            console.print(f"[red]❌ Data not found: {data_path}[/red]")
            console.print("  Use --dummy-data for testing, or download training videos first:")
            console.print("  python scripts/download_training_videos.py --num-videos 100")
            raise typer.Exit(code=1)

        # Load video dataset
        from aiprod_trainer.video_utils import read_video

        videos = []
        for vf in sorted(Path(data_path).glob("*.mp4"))[:200]:
            try:
                video = read_video(str(vf), num_frames=16, height=128, width=128)
                videos.append(video)
            except Exception:
                continue

        if not videos:
            console.print(f"[red]❌ No valid videos found in {data_path}[/red]")
            raise typer.Exit(code=1)

        data = torch.stack(videos)
        dataset = TensorDataset(data)
        train_dl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        console.print(f"  Loaded {len(videos)} videos from {data_path}")

    # Train
    trainer = VideoVAETrainer(
        model=model,
        config=config,
        train_dataloader=train_dl,
    )

    results = trainer.train()
    console.print(f"\n✅ Video VAE training complete: {results}")


def _train_audio_vae(raw: dict, config: VAETrainerConfig, dummy_data: bool) -> None:
    """Train audio VAE."""
    from torch.utils.data import DataLoader, TensorDataset

    from aiprod_trainer.vae_trainer import AudioVAETrainer

    # Build model
    try:
        from aiprod_core.model.audio_vae import AudioDecoder, AudioEncoder, NACConfig

        nac_config = NACConfig()
        model = AudioEncoder(nac_config)
    except (ImportError, AttributeError):
        console.print("[yellow]Using placeholder model (AudioEncoder import failed)[/yellow]")
        import torch.nn as nn

        model = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(64, 1, 7, padding=3),
        )

    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Model parameters: {total_params:,}")

    # Build dataloader
    if dummy_data:
        console.print("[yellow]Using DUMMY data (random tensors)[/yellow]")
        # Random audio: [B, samples] at 24kHz, 3 seconds
        data = torch.randn(200, 24000 * 3)
        dataset = TensorDataset(data)
        train_dl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    else:
        data_path = raw.get("data", {}).get("path", "data/audio")
        if not Path(data_path).exists():
            console.print(f"[red]❌ Audio data not found: {data_path}[/red]")
            console.print("  Use --dummy-data for testing")
            raise typer.Exit(code=1)

        import torchaudio

        audios = []
        for af in sorted(Path(data_path).glob("*.wav"))[:500]:
            try:
                waveform, sr = torchaudio.load(str(af))
                if sr != 24000:
                    waveform = torchaudio.functional.resample(waveform, sr, 24000)
                # Trim/pad to 3 seconds
                target_len = 24000 * 3
                if waveform.shape[-1] > target_len:
                    waveform = waveform[..., :target_len]
                else:
                    waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[-1]))
                audios.append(waveform.squeeze(0))
            except Exception:
                continue

        if not audios:
            console.print(f"[red]❌ No valid audio found in {data_path}[/red]")
            raise typer.Exit(code=1)

        data = torch.stack(audios)
        dataset = TensorDataset(data)
        train_dl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
        console.print(f"  Loaded {len(audios)} audio clips from {data_path}")

    # Train
    trainer = AudioVAETrainer(
        model=model,
        config=config,
        train_dataloader=train_dl,
    )

    results = trainer.train()
    console.print(f"\n✅ Audio VAE training complete: {results}")


if __name__ == "__main__":
    app()
