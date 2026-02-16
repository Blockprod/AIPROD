# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
TTS Trainer — Text-to-Speech training loop (3-phase)

Implements training for the AIPROD TTS pipeline:
- Phase 1: TextFrontend + MelDecoder (mel spectrogram generation)
- Phase 2: VocoderTTS / HiFi-GAN (waveform synthesis)
- Phase 3: ProsodyModeler (duration, pitch, energy prediction)

Each phase trains different components while freezing the rest.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# wandb: optional dependency (souveraineté)
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False

from accelerate import Accelerator
from safetensors.torch import save_file
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ─── TTS LOSSES ──────────────────────────────────────────────────────────


class MelSpectrogramLoss(nn.Module):
    """Multi-scale mel spectrogram reconstruction loss."""

    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def forward(
        self, predicted_mel: torch.Tensor, target_mel: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predicted_mel: [B, n_mels, T]
            target_mel: [B, n_mels, T]
        """
        # L1 loss on mel spectrograms
        l1_loss = F.l1_loss(predicted_mel, target_mel)
        # MSE loss for stability
        mse_loss = F.mse_loss(predicted_mel, target_mel)

        return l1_loss + 0.5 * mse_loss


class DurationPredictionLoss(nn.Module):
    """MSE loss on log-duration predictions."""

    def forward(
        self, predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            predicted: [B, T] predicted log-durations
            target: [B, T] target durations (in frames)
            mask: [B, T] optional mask for valid tokens
        """
        target_log = torch.log(target.float().clamp(min=1))
        loss = F.mse_loss(predicted, target_log, reduction="none")
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        return loss


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss for vocoder discriminator training."""

    def forward(
        self,
        real_features: list[torch.Tensor],
        fake_features: list[torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=real_features[0].device)
        for r, f in zip(real_features, fake_features):
            loss = loss + F.l1_loss(f, r.detach())
        return loss / len(real_features)


# ─── TTS TRAINER CONFIG ─────────────────────────────────────────────────


@dataclass
class TTSPhaseConfig:
    """Config for a single training phase."""

    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    warmup_steps: int = 1000
    data: str = "data/ljspeech"
    components: list[str] = field(default_factory=lambda: ["text_frontend", "mel_decoder"])


@dataclass
class TTSTrainerConfig:
    """Configuration for TTS training."""

    # Phase configs
    phase1: TTSPhaseConfig = field(
        default_factory=lambda: TTSPhaseConfig(
            epochs=200,
            batch_size=16,
            learning_rate=1e-3,
            warmup_steps=4000,
            data="data/ljspeech",
            components=["text_frontend", "text_encoder", "mel_decoder"],
        )
    )
    phase2: TTSPhaseConfig = field(
        default_factory=lambda: TTSPhaseConfig(
            epochs=500,
            batch_size=16,
            learning_rate=2e-4,
            warmup_steps=1000,
            data="data/ljspeech",
            components=["vocoder"],
        )
    )
    phase3: TTSPhaseConfig = field(
        default_factory=lambda: TTSPhaseConfig(
            epochs=100,
            batch_size=32,
            learning_rate=5e-5,
            warmup_steps=500,
            data="data/libritts",
            components=["prosody", "speaker"],
        )
    )

    # Optimization
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    mixed_precision: str = "bf16"
    max_grad_norm: float = 5.0

    # Loss weights
    mel_reconstruction_weight: float = 1.0
    duration_prediction_weight: float = 1.0
    pitch_prediction_weight: float = 0.1
    energy_prediction_weight: float = 0.1
    adversarial_weight: float = 1.0
    feature_matching_weight: float = 2.0
    mel_spectral_weight: float = 45.0

    # Validation
    eval_interval_epochs: int = 20
    num_eval_samples: int = 4
    eval_prompts: list[str] = field(
        default_factory=lambda: [
            "The quick brown fox jumps over the lazy dog.",
            "In a quiet village nestled between mountains.",
            "Welcome to the future of speech synthesis.",
        ]
    )

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints/aiprod_tts_v1")
    save_interval_epochs: int = 50
    keep_last_n: int = 2

    # Output
    output_dir: str = "checkpoints/aiprod_tts_v1"
    final_output: str = "models/aiprod-sovereign/aiprod-tts-v1.safetensors"

    # Logging
    log_interval: int = 10
    use_wandb: bool = False
    project_name: str = "aiprod-sovereign-tts"

    # Reproducibility
    seed: int = 42


# ─── TTS DATASET ─────────────────────────────────────────────────────────


class TTSDataset(torch.utils.data.Dataset):
    """Dataset for TTS training.

    Expects pre-processed data with:
    - phoneme_ids.pt: [seq_len] tensor of phoneme indices
    - mel.pt: [n_mels, T] mel spectrogram
    - duration.pt: [seq_len] duration of each phoneme in frames
    - f0.pt: [T] fundamental frequency contour
    - energy.pt: [T] energy contour
    - speaker_id.pt: scalar speaker index
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"TTS data directory not found: {data_dir}")

        # Discover all samples
        self.sample_dirs = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir() and (d / "phoneme_ids.pt").exists()]
        )

        if not self.sample_dirs:
            # Fallback: look for numbered .pt files
            self.sample_dirs = []
            phoneme_files = sorted(self.data_dir.glob("phoneme_ids_*.pt"))
            for pf in phoneme_files:
                idx = pf.stem.split("_")[-1]
                self.sample_dirs.append((self.data_dir, idx))

        logger.info(f"TTS dataset: {len(self)} samples from {data_dir}")

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.sample_dirs[idx]

        if isinstance(item, tuple):
            data_dir, sample_idx = item
            return {
                "phoneme_ids": torch.load(data_dir / f"phoneme_ids_{sample_idx}.pt", weights_only=True),
                "mel": torch.load(data_dir / f"mel_{sample_idx}.pt", weights_only=True),
                "duration": torch.load(data_dir / f"duration_{sample_idx}.pt", weights_only=True),
                "f0": torch.load(data_dir / f"f0_{sample_idx}.pt", weights_only=True),
                "energy": torch.load(data_dir / f"energy_{sample_idx}.pt", weights_only=True),
                "speaker_id": torch.load(data_dir / f"speaker_id_{sample_idx}.pt", weights_only=True),
            }
        else:
            sample_dir = item
            return {
                "phoneme_ids": torch.load(sample_dir / "phoneme_ids.pt", weights_only=True),
                "mel": torch.load(sample_dir / "mel.pt", weights_only=True),
                "duration": torch.load(sample_dir / "duration.pt", weights_only=True),
                "f0": torch.load(sample_dir / "f0.pt", weights_only=True),
                "energy": torch.load(sample_dir / "energy.pt", weights_only=True),
                "speaker_id": torch.load(sample_dir / "speaker_id.pt", weights_only=True),
            }


class DummyTTSDataset(torch.utils.data.Dataset):
    """Dummy TTS dataset for testing and benchmarking."""

    def __init__(
        self,
        num_samples: int = 200,
        max_phonemes: int = 128,
        n_mels: int = 80,
        max_mel_frames: int = 512,
        vocab_size: int = 76,
    ):
        self.num_samples = num_samples
        self.max_phonemes = max_phonemes
        self.n_mels = n_mels
        self.max_mel_frames = max_mel_frames
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_len = torch.randint(20, self.max_phonemes, (1,)).item()
        mel_len = seq_len * 4  # ~4 frames per phoneme

        return {
            "phoneme_ids": torch.randint(0, self.vocab_size, (seq_len,)),
            "mel": torch.randn(self.n_mels, mel_len),
            "duration": torch.randint(2, 8, (seq_len,)),
            "f0": torch.randn(mel_len) * 50 + 200,
            "energy": torch.randn(mel_len).abs(),
            "speaker_id": torch.tensor(0),
        }


def _collate_tts(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for TTS data with variable-length padding."""
    max_phoneme_len = max(b["phoneme_ids"].shape[0] for b in batch)
    max_mel_len = max(b["mel"].shape[1] for b in batch)
    n_mels = batch[0]["mel"].shape[0]

    padded = {
        "phoneme_ids": torch.zeros(len(batch), max_phoneme_len, dtype=torch.long),
        "phoneme_mask": torch.zeros(len(batch), max_phoneme_len, dtype=torch.bool),
        "mel": torch.zeros(len(batch), n_mels, max_mel_len),
        "duration": torch.zeros(len(batch), max_phoneme_len, dtype=torch.long),
        "f0": torch.zeros(len(batch), max_mel_len),
        "energy": torch.zeros(len(batch), max_mel_len),
        "speaker_id": torch.stack([b["speaker_id"] for b in batch]),
    }

    for i, b in enumerate(batch):
        p_len = b["phoneme_ids"].shape[0]
        m_len = b["mel"].shape[1]
        padded["phoneme_ids"][i, :p_len] = b["phoneme_ids"]
        padded["phoneme_mask"][i, :p_len] = True
        padded["mel"][i, :, :m_len] = b["mel"]
        padded["duration"][i, :p_len] = b["duration"]
        padded["f0"][i, :m_len] = b["f0"]
        padded["energy"][i, :m_len] = b["energy"]

    return padded


# ─── TTS TRAINER ─────────────────────────────────────────────────────────


class TTSTrainer:
    """Three-phase TTS trainer.

    Phase 1: Train TextFrontend + TextEncoder + MelDecoder (mel prediction)
    Phase 2: Train VocoderTTS / HiFi-GAN (waveform generation from mel)
    Phase 3: Train ProsodyModeler (duration, pitch, energy prediction)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TTSTrainerConfig,
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
        checkpoint_path: Path | None = None,
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.start_epoch = 0

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Loss functions
        self.mel_loss = MelSpectrogramLoss()
        self.duration_loss = DurationPredictionLoss()

        # Initialize wandb
        self._wandb_run = None
        if config.use_wandb and _WANDB_AVAILABLE and self.accelerator.is_main_process:
            self._wandb_run = wandb.init(
                project=config.project_name,
                config=config.__dict__,
            )

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def _freeze_all(self) -> None:
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def _unfreeze_components(self, component_names: list[str]) -> int:
        """Unfreeze specific model components. Returns trainable param count."""
        trainable = 0
        for name in component_names:
            component = getattr(self.model, name, None)
            if component is None:
                # Try common aliases
                aliases = {
                    "text_frontend": "text_frontend",
                    "text_encoder": "text_encoder",
                    "mel_decoder": "mel_decoder",
                    "vocoder": "vocoder",
                    "prosody": "prosody",
                    "speaker": "speaker",
                    "speaker_proj": "speaker_proj",
                }
                attr_name = aliases.get(name, name)
                component = getattr(self.model, attr_name, None)

            if component is not None:
                component.requires_grad_(True)
                trainable += sum(p.numel() for p in component.parameters() if p.requires_grad)
            else:
                logger.warning(f"Component '{name}' not found in model — skipping")

        return trainable

    def train(self) -> dict:
        """Run full 3-phase training.

        Returns:
            dict with training summary
        """
        results = {}

        phases = [
            (1, self.config.phase1),
            (2, self.config.phase2),
            (3, self.config.phase3),
        ]

        for phase_num, phase_config in phases:
            logger.info(f"\n{'='*60}")
            logger.info(f"PHASE {phase_num} — Components: {phase_config.components}")
            logger.info(f"  Epochs: {phase_config.epochs}, LR: {phase_config.learning_rate}")
            logger.info(f"  Data: {phase_config.data}")
            logger.info(f"{'='*60}")

            phase_result = self._train_phase(phase_num, phase_config)
            results[f"phase_{phase_num}"] = phase_result

        # Save final model
        if self.accelerator.is_main_process:
            self._save_final_model()

        if self._wandb_run is not None:
            self._wandb_run.finish()

        return results

    def _train_phase(self, phase_num: int, phase_config: TTSPhaseConfig) -> dict:
        """Train a single phase."""
        # Freeze all, then unfreeze target components
        self._freeze_all()
        trainable_params = self._unfreeze_components(phase_config.components)

        if self.accelerator.is_main_process:
            logger.info(f"Phase {phase_num}: {trainable_params:,} trainable parameters")

        # Create optimizer for this phase
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            logger.warning(f"Phase {phase_num}: No trainable parameters, skipping")
            return {"skipped": True}

        optimizer = AdamW(
            params,
            lr=phase_config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Cosine annealing scheduler
        total_steps = phase_config.epochs * max(1, len(self.train_dataloader))
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=phase_config.learning_rate * 0.01,
        )

        # Warmup scheduler (linear)
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=phase_config.warmup_steps,
        )
        combined_scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[phase_config.warmup_steps],
        )

        # Prepare with accelerator
        self.model, optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, optimizer, self.train_dataloader
        )
        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        # Training loop
        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(phase_config.epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch in self.train_dataloader:
                with self.accelerator.accumulate(self.model):
                    loss, loss_dict = self._training_step(batch, phase_num)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(params, self.config.max_grad_norm)

                    optimizer.step()
                    combined_scheduler.step()
                    optimizer.zero_grad()

                    epoch_loss += loss.item()
                    num_batches += 1
                    global_step += 1

                    # Log
                    if global_step % self.config.log_interval == 0 and self.accelerator.is_main_process:
                        lr = optimizer.param_groups[0]["lr"]
                        logger.info(
                            f"Phase {phase_num} | Epoch {epoch+1}/{phase_config.epochs} | "
                            f"Step {global_step} | Loss: {loss.item():.4f} | LR: {lr:.2e}"
                        )
                        if self._wandb_run is not None:
                            wandb.log(
                                {f"phase{phase_num}/{k}": v for k, v in loss_dict.items()}
                                | {
                                    f"phase{phase_num}/lr": lr,
                                    f"phase{phase_num}/epoch": epoch,
                                    "global_step": global_step,
                                },
                                step=global_step,
                            )

            avg_loss = epoch_loss / max(num_batches, 1)

            # Validation
            if (
                self.val_dataloader is not None
                and (epoch + 1) % self.config.eval_interval_epochs == 0
            ):
                val_loss = self._validate(phase_num)
                if self.accelerator.is_main_process:
                    logger.info(
                        f"Phase {phase_num} | Epoch {epoch+1} | "
                        f"Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}"
                    )
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

            # Save checkpoint
            if (
                (epoch + 1) % self.config.save_interval_epochs == 0
                and self.accelerator.is_main_process
            ):
                self.save_checkpoint(phase_num, epoch)

        # Save end-of-phase checkpoint
        if self.accelerator.is_main_process:
            self.save_checkpoint(phase_num, phase_config.epochs - 1, is_final=True)

        return {
            "epochs": phase_config.epochs,
            "final_loss": avg_loss,
            "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
            "trainable_params": trainable_params,
        }

    def _training_step(
        self, batch: dict[str, torch.Tensor], phase_num: int
    ) -> tuple[torch.Tensor, dict]:
        """Single training step, loss depends on phase."""
        phoneme_ids = batch["phoneme_ids"]
        speaker_id = batch["speaker_id"]
        target_mel = batch["mel"]
        target_duration = batch["duration"]
        target_f0 = batch["f0"]
        target_energy = batch["energy"]
        mask = batch.get("phoneme_mask")

        if phase_num == 1:
            # Phase 1: Train mel prediction (TextEncoder + MelDecoder)
            outputs = self.model(
                phoneme_ids=phoneme_ids,
                speaker_id=speaker_id,
                target_mel=target_mel,
                target_durations=target_duration,
                target_f0=target_f0,
                target_energy=target_energy,
            )

            mel_pred = outputs.get("mel_output", outputs.get("mel"))
            if mel_pred is None:
                raise ValueError("Model did not return 'mel_output' or 'mel' key")

            # Truncate to matching length
            min_len = min(mel_pred.shape[-1], target_mel.shape[-1])
            mel_pred = mel_pred[..., :min_len]
            target_mel = target_mel[..., :min_len]

            mel_loss = self.mel_loss(mel_pred, target_mel)

            # Duration prediction loss (if model returns it)
            dur_pred = outputs.get("duration_prediction")
            dur_loss = torch.tensor(0.0, device=mel_loss.device)
            if dur_pred is not None:
                dur_loss = self.duration_loss(dur_pred, target_duration, mask)

            # Pitch + energy prediction loss
            f0_pred = outputs.get("f0_prediction")
            f0_loss = torch.tensor(0.0, device=mel_loss.device)
            if f0_pred is not None:
                min_t = min(f0_pred.shape[-1], target_f0.shape[-1])
                f0_loss = F.mse_loss(f0_pred[..., :min_t], target_f0[..., :min_t])

            energy_pred = outputs.get("energy_prediction")
            energy_loss = torch.tensor(0.0, device=mel_loss.device)
            if energy_pred is not None:
                min_t = min(energy_pred.shape[-1], target_energy.shape[-1])
                energy_loss = F.mse_loss(energy_pred[..., :min_t], target_energy[..., :min_t])

            total_loss = (
                self.config.mel_reconstruction_weight * mel_loss
                + self.config.duration_prediction_weight * dur_loss
                + self.config.pitch_prediction_weight * f0_loss
                + self.config.energy_prediction_weight * energy_loss
            )

            return total_loss, {
                "loss/total": total_loss.item(),
                "loss/mel": mel_loss.item(),
                "loss/duration": dur_loss.item(),
                "loss/f0": f0_loss.item(),
                "loss/energy": energy_loss.item(),
            }

        elif phase_num == 2:
            # Phase 2: Train vocoder (mel → waveform)
            # Generate mel from model in eval mode for encoder, train vocoder
            with torch.no_grad():
                outputs = self.model(
                    phoneme_ids=phoneme_ids,
                    speaker_id=speaker_id,
                    target_mel=target_mel,
                    target_durations=target_duration,
                    target_f0=target_f0,
                    target_energy=target_energy,
                )

            mel_pred = outputs.get("mel_output", target_mel)

            # Forward through vocoder only
            waveform = self.model.vocoder(mel_pred)

            # Mel spectrogram of generated waveform (for mel spectral loss)
            # Simple L1 loss on waveform vs vocoder output from target mel
            target_waveform = self.model.vocoder(target_mel)

            # Waveform reconstruction loss
            min_len = min(waveform.shape[-1], target_waveform.shape[-1])
            wav_loss = F.l1_loss(
                waveform[..., :min_len], target_waveform[..., :min_len].detach()
            )

            # Mel spectral consistency
            mel_spectral_loss = self.mel_loss(mel_pred, target_mel)

            total_loss = wav_loss + self.config.mel_spectral_weight * mel_spectral_loss

            return total_loss, {
                "loss/total": total_loss.item(),
                "loss/waveform": wav_loss.item(),
                "loss/mel_spectral": mel_spectral_loss.item(),
            }

        elif phase_num == 3:
            # Phase 3: Train prosody modeler
            outputs = self.model(
                phoneme_ids=phoneme_ids,
                speaker_id=speaker_id,
                target_durations=target_duration,
                target_f0=target_f0,
                target_energy=target_energy,
            )

            dur_pred = outputs.get("duration_prediction")
            f0_pred = outputs.get("f0_prediction")
            energy_pred = outputs.get("energy_prediction")

            dur_loss = torch.tensor(0.0, device=phoneme_ids.device)
            f0_loss = torch.tensor(0.0, device=phoneme_ids.device)
            energy_loss = torch.tensor(0.0, device=phoneme_ids.device)

            if dur_pred is not None:
                dur_loss = self.duration_loss(dur_pred, target_duration, mask)
            if f0_pred is not None:
                min_t = min(f0_pred.shape[-1], target_f0.shape[-1])
                f0_loss = F.mse_loss(f0_pred[..., :min_t], target_f0[..., :min_t])
            if energy_pred is not None:
                min_t = min(energy_pred.shape[-1], target_energy.shape[-1])
                energy_loss = F.mse_loss(energy_pred[..., :min_t], target_energy[..., :min_t])

            total_loss = (
                self.config.duration_prediction_weight * dur_loss
                + self.config.pitch_prediction_weight * f0_loss
                + self.config.energy_prediction_weight * energy_loss
            )

            return total_loss, {
                "loss/total": total_loss.item(),
                "loss/duration": dur_loss.item(),
                "loss/f0": f0_loss.item(),
                "loss/energy": energy_loss.item(),
            }

        else:
            raise ValueError(f"Unknown phase: {phase_num}")

    @torch.no_grad()
    def _validate(self, phase_num: int) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            loss, _ = self._training_step(batch, phase_num)
            total_loss += loss.item()
            num_batches += 1

        self.model.train()
        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, phase: int, epoch: int, is_final: bool = False) -> None:
        """Save training checkpoint."""
        ckpt_dir = Path(self.config.checkpoint_dir) / f"phase_{phase}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        suffix = "final" if is_final else f"epoch_{epoch:04d}"
        checkpoint = {
            "phase": phase,
            "epoch": epoch,
            "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
        }

        ckpt_path = ckpt_dir / f"checkpoint_{suffix}.pt"
        self.accelerator.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        # Cleanup old checkpoints
        if not is_final and self.config.keep_last_n > 0:
            all_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
            while len(all_ckpts) > self.config.keep_last_n:
                old = all_ckpts.pop(0)
                old.unlink()
                logger.debug(f"Removed old checkpoint: {old}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load a checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.accelerator.unwrap_model(self.model).load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(
            f"Loaded checkpoint from phase {checkpoint.get('phase', '?')}, "
            f"epoch {checkpoint.get('epoch', '?')}"
        )

    def _save_final_model(self) -> None:
        """Save the final model as safetensors."""
        output_path = Path(self.config.final_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        # Convert all to bfloat16 for storage
        bf16_dict = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state_dict.items()}

        save_file(bf16_dict, str(output_path))
        size_mb = output_path.stat().st_size / 1024**2
        logger.info(f"Final TTS model saved: {output_path} ({size_mb:.0f} MB)")


__all__ = [
    "TTSTrainer",
    "TTSTrainerConfig",
    "TTSPhaseConfig",
    "TTSDataset",
    "DummyTTSDataset",
    "MelSpectrogramLoss",
    "DurationPredictionLoss",
    "FeatureMatchingLoss",
]
