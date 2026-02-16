#!/usr/bin/env python
"""
Train AIPROD TTS model using configuration from YAML file.

3-phase training:
  Phase 1: TextFrontend + MelDecoder — mel spectrogram generation (200 epochs)
  Phase 2: VocoderTTS / HiFi-GAN — waveform synthesis (500 epochs)
  Phase 3: ProsodyModeler — duration, pitch, energy (100 epochs)

Usage:
    python scripts/tts_train.py configs/train/tts_training.yaml
    python scripts/tts_train.py configs/train/tts_training.yaml --dummy-data
"""

from pathlib import Path

import torch
import typer
import yaml
from rich.console import Console

console = Console()
app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=True,
    help="Train AIPROD TTS model (3-phase).",
)


@app.command()
def main(
    config_path: str = typer.Argument(..., help="Path to TTS training YAML config"),
    dummy_data: bool = typer.Option(
        False,
        "--dummy-data",
        help="Use dummy random data for testing the pipeline",
    ),
    phase: int = typer.Option(
        0,
        "--phase",
        help="Run only a specific phase (1, 2, or 3). 0 = all phases",
    ),
) -> None:
    """Train the AIPROD TTS model."""
    from torch.utils.data import DataLoader

    from aiprod_core.model.tts import TTSConfig, TTSModel
    from aiprod_trainer.tts_trainer import (
        DummyTTSDataset,
        TTSDataset,
        TTSPhaseConfig,
        TTSTrainer,
        TTSTrainerConfig,
        _collate_tts,
    )

    # Load config
    config_path = Path(config_path)
    if not config_path.exists():
        typer.echo(f"Error: Config file not found: {config_path}")
        raise typer.Exit(code=1)

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Build trainer config from YAML
    trainer_config = TTSTrainerConfig()

    # Loss weights
    if "loss" in raw:
        for key, val in raw["loss"].items():
            if hasattr(trainer_config, key):
                setattr(trainer_config, key, val)

    # Training phases
    if "training" in raw:
        tc = raw["training"]
        if "phase1" in tc:
            trainer_config.phase1 = TTSPhaseConfig(**tc["phase1"])
        if "phase2" in tc:
            trainer_config.phase2 = TTSPhaseConfig(**tc["phase2"])
        if "phase3" in tc:
            trainer_config.phase3 = TTSPhaseConfig(**tc["phase3"])
        for key in ["weight_decay", "gradient_accumulation_steps", "mixed_precision", "max_grad_norm"]:
            if key in tc:
                setattr(trainer_config, key, tc[key])

    # Validation
    if "validation" in raw:
        vc = raw["validation"]
        if "interval_epochs" in vc:
            trainer_config.eval_interval_epochs = vc["interval_epochs"]
        if "num_samples" in vc:
            trainer_config.num_eval_samples = vc["num_samples"]
        if "prompts" in vc:
            trainer_config.eval_prompts = vc["prompts"]

    # Checkpoints
    if "checkpoints" in raw:
        cc = raw["checkpoints"]
        if "save_interval_epochs" in cc:
            trainer_config.save_interval_epochs = cc["save_interval_epochs"]
        if "keep_last_n" in cc:
            trainer_config.keep_last_n = cc["keep_last_n"]

    # Output
    if "output" in raw:
        oc = raw["output"]
        if "dir" in oc:
            trainer_config.output_dir = oc["dir"]
            trainer_config.checkpoint_dir = Path(oc["dir"])
        if "final" in oc:
            trainer_config.final_output = oc["final"]

    # Wandb
    if "wandb" in raw:
        wc = raw["wandb"]
        trainer_config.use_wandb = wc.get("enabled", False)
        if "project" in wc:
            trainer_config.project_name = wc["project"]

    # Seed
    if "seed" in raw:
        trainer_config.seed = raw["seed"]

    console.print(f"\n[bold]AIPROD TTS Training[/bold]")
    console.print(f"  Config: {config_path}")
    console.print(f"  Output: {trainer_config.output_dir}")
    console.print(f"  Wandb: {'enabled' if trainer_config.use_wandb else 'disabled'}")

    # Build model
    model_raw = raw.get("model", {})
    tts_config = TTSConfig(
        encoder_hidden=model_raw.get("mel_decoder", {}).get("hidden_dim", 512),
        encoder_layers=model_raw.get("mel_decoder", {}).get("num_layers", 6) // 2,
        encoder_heads=model_raw.get("mel_decoder", {}).get("num_heads", 8),
        encoder_ff_dim=model_raw.get("mel_decoder", {}).get("ff_dim", 2048),
        decoder_hidden=model_raw.get("mel_decoder", {}).get("hidden_dim", 512),
        decoder_layers=model_raw.get("mel_decoder", {}).get("num_layers", 6),
        decoder_heads=model_raw.get("mel_decoder", {}).get("num_heads", 8),
        decoder_ff_dim=model_raw.get("mel_decoder", {}).get("ff_dim", 2048),
        vocab_size=model_raw.get("text_frontend", {}).get("vocab_size", 256),
        num_mels=model_raw.get("mel_decoder", {}).get("n_mels", 80),
    )

    console.print(f"  Model params: encoder={tts_config.encoder_hidden}d, {tts_config.decoder_layers} layers")
    model = TTSModel(tts_config)
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Total parameters: {total_params:,}")

    # Build dataloader
    if dummy_data:
        console.print(f"\n[yellow]Using DUMMY data (random) for pipeline testing[/yellow]")
        dataset = DummyTTSDataset(num_samples=200, n_mels=tts_config.num_mels)
    else:
        # Use phase1 data path by default
        data_path = trainer_config.phase1.data
        console.print(f"\n  Data: {data_path}")
        dataset = TTSDataset(data_path)

    dataloader = DataLoader(
        dataset,
        batch_size=trainer_config.phase1.batch_size,
        shuffle=True,
        collate_fn=_collate_tts,
        num_workers=2,
        pin_memory=True,
    )

    # Filter phases if --phase specified
    if phase > 0:
        console.print(f"\n[cyan]Running ONLY phase {phase}[/cyan]")

    # Create trainer and run
    trainer = TTSTrainer(
        model=model,
        config=trainer_config,
        train_dataloader=dataloader,
    )

    if phase == 0:
        # Run all phases
        results = trainer.train()
    else:
        # Run single phase
        phase_configs = {1: trainer_config.phase1, 2: trainer_config.phase2, 3: trainer_config.phase3}
        if phase not in phase_configs:
            typer.echo(f"Error: Invalid phase {phase}. Must be 1, 2, or 3.")
            raise typer.Exit(code=1)
        results = {f"phase_{phase}": trainer._train_phase(phase, phase_configs[phase])}
        if trainer.accelerator.is_main_process:
            trainer._save_final_model()

    # Summary
    console.print(f"\n{'='*60}")
    console.print(f"[bold green]TTS TRAINING COMPLETE[/bold green]")
    for phase_name, result in results.items():
        if isinstance(result, dict) and not result.get("skipped"):
            console.print(
                f"  {phase_name}: {result.get('epochs', '?')} epochs, "
                f"loss={result.get('final_loss', '?'):.4f}, "
                f"params={result.get('trainable_params', '?'):,}"
            )
    console.print(f"  Output: {trainer_config.final_output}")
    console.print(f"{'='*60}")


if __name__ == "__main__":
    app()
