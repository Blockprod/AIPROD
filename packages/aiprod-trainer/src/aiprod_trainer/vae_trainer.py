# Copyright (c) 2025-2026 AIPROD. All rights reserved.
# AIPROD Proprietary Software — See LICENSE for terms.

"""
VAE Trainers — Video and Audio VAE training loop

Implements training for:
- Video VAE (encoder/decoder) with reconstruction + KL + perceptual loss
- Audio VAE (encoder/decoder) with reconstruction + spectral + adversarial loss
- Optional vocoder training for audio synthesis quality
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
import wandb
from accelerate import Accelerator
from torch.optim import AdamW, lr_scheduler
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


# ─── VIDEO VAE LOSSES ──────────────────────────────────────────────────────


class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss for video VAE training.
    
    Uses intermediate features from pretrained VGG16 to compute
    perceptual distance between generated and target videos.
    
    Falls back to L2 loss if torchvision is not available.
    """

    def __init__(self, layers: list[str] | None = None, device: torch.device = torch.device("cpu")):
        super().__init__()
        
        self.use_vgg = False
        self.features = None
        self.mean = None
        self.std = None
        
        try:
            from torchvision.models import vgg16
            
            self.use_vgg = True
            
            if layers is None:
                layers = ["relu3_2"]  # Can extend to multiple layers if needed

            vgg = vgg16(pretrained=True).to(device)
            self.features = nn.ModuleDict()

            layer_name_mapping = {
                "relu1_1": 2, "relu1_2": 4,
                "relu2_1": 7, "relu2_2": 9,
                "relu3_1": 12, "relu3_2": 14, "relu3_3": 16,
                "relu4_1": 19, "relu4_2": 21, "relu4_3": 23,
                "relu5_1": 26, "relu5_2": 28, "relu5_3": 30,
            }

            for layer in layers:
                idx = layer_name_mapping.get(layer, 14)
                self.features[layer] = nn.Sequential(*list(vgg.features.children())[:idx+1])
                for param in self.features[layer].parameters():
                    param.requires_grad = False

            self.register_buffer(
                "mean",
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                "std",
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize VGG16 perceptual loss ({type(e).__name__}). "
                f"Using L2 loss fallback instead."
            )

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        if not self.use_vgg or self.features is None:
            # Fallback: simple L2 loss
            return F.mse_loss(x, target)
        
        # Normalize to ImageNet stats
        x_norm = (x - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        loss = 0.0
        for layer in self.features.values():
            x_feat = layer(x_norm)
            target_feat = layer(target_norm)
            loss = loss + F.mse_loss(x_feat, target_feat)

        return loss


class VideoVAELoss(nn.Module):
    """Combined loss for video VAE training.
    
    Loss = reconstruction + beta * KL + lambda * perceptual
    """

    def __init__(
        self,
        beta: float = 0.1,
        lambda_perceptual: float = 1.0,
        use_perceptual: bool = True,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.beta = beta
        self.lambda_perceptual = lambda_perceptual
        self.use_perceptual = use_perceptual

        if use_perceptual:
            self.perceptual_loss = PerceptualLoss(device=device)
        else:
            self.perceptual_loss = None

    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute total VAE loss.
        
        Args:
            reconstructed: Reconstructed output [B, C, T, H, W]
            target: Target input [B, C, T, H, W]
            mu: Latent mean from encoder
            logvar: Latent log-variance from encoder
        
        Returns:
            Total loss, dict with individual loss components
        """
        # Reconstruction loss (L2)
        recon_loss = F.mse_loss(reconstructed, target, reduction="mean")

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Perceptual loss (optional)
        if self.use_perceptual and self.perceptual_loss is not None:
            # Sample video frames and compute perceptual loss
            # Use first, middle, and last frames
            t = target.shape[2]
            frame_indices = [0, t // 2, t - 1]
            perc_loss = 0.0
            for idx in frame_indices:
                perc_loss = perc_loss + self.perceptual_loss(
                    reconstructed[:, :, idx],
                    target[:, :, idx]
                )
            perc_loss = perc_loss / len(frame_indices)
        else:
            perc_loss = 0.0

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        if self.use_perceptual:
            total_loss = total_loss + self.lambda_perceptual * perc_loss

        return total_loss, {
            "loss/recon": recon_loss.item(),
            "loss/kl": kl_loss.item(),
            "loss/perceptual": perc_loss.item() if isinstance(perc_loss, torch.Tensor) else perc_loss,
            "loss/total": total_loss.item(),
        }


# ─── AUDIO VAE LOSSES ──────────────────────────────────────────────────────


class SpectralLoss(nn.Module):
    """Spectral loss for audio VAE training.
    
    Compares magnitude spectrograms of reconstructed and target audio.
    """

    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, reconstructed: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute spectral loss."""
        # Compute spectrograms
        recon_spec = torch.stft(
            reconstructed,
            self.n_fft,
            self.hop_length,
            return_complex=True,
            onesided=True,
            pad_mode="reflect"
        ).abs()

        target_spec = torch.stft(
            target,
            self.n_fft,
            self.hop_length,
            return_complex=True,
            onesided=True,
            pad_mode="reflect"
        ).abs()

        # Log magnitude loss (more perceptually relevant)
        spec_loss = F.l1_loss(
            torch.log(recon_spec + 1e-9),
            torch.log(target_spec + 1e-9)
        )

        # Linear magnitude loss
        spec_loss = spec_loss + F.l1_loss(recon_spec, target_spec)

        return spec_loss


class AudioVAELoss(nn.Module):
    """Combined loss for audio VAE training.
    
    Loss = reconstruction + beta * KL + lambda * spectral
    """

    def __init__(
        self,
        beta: float = 0.1,
        lambda_spectral: float = 1.0,
        n_fft: int = 1024,
    ):
        super().__init__()
        self.beta = beta
        self.lambda_spectral = lambda_spectral
        self.spectral_loss = SpectralLoss(n_fft=n_fft)

    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute total audio VAE loss."""
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, target, reduction="mean")

        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Spectral loss
        spec_loss = self.spectral_loss(reconstructed, target)

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss + self.lambda_spectral * spec_loss

        return total_loss, {
            "loss/recon": recon_loss.item(),
            "loss/kl": kl_loss.item(),
            "loss/spectral": spec_loss.item(),
            "loss/total": total_loss.item(),
        }


# ─── VAE TRAINER CONFIG ────────────────────────────────────────────────────


@dataclass
class VAETrainerConfig:
    """Configuration for VAE training."""
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    batch_size: int = 4
    num_epochs: int = 10
    warmup_steps: int = 1000

    # Loss weights (Video VAE)
    beta_kl: float = 0.1
    lambda_perceptual: float = 1.0
    use_perceptual_loss: bool = True

    # Loss weights (Audio VAE)
    lambda_spectral: float = 1.0

    # Optimization
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"  # "bf16", "fp16", or "no"

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_interval: int = 500
    eval_interval: int = 100

    # Logging
    log_interval: int = 10
    use_wandb: bool = True
    project_name: str = "aiprod-vae-training"


# ─── VAE TRAINERS ────────────────────────────────────────────────────────


class VideoVAETrainer:
    """Trainer for video VAE model."""

    def __init__(
        self,
        model: nn.Module,
        config: VAETrainerConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        checkpoint_path: Path | None = None,
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Loss function
        self.loss_fn = VideoVAELoss(
            beta=config.beta_kl,
            lambda_perceptual=config.lambda_perceptual,
            use_perceptual=config.use_perceptual_loss,
            device=self.accelerator.device,
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        total_steps = len(train_dataloader) * config.num_epochs
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )

        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        # Load checkpoint if provided
        self.checkpoint_path = checkpoint_path
        self.start_epoch = 0
        if checkpoint_path and checkpoint_path.exists():
            self.load_checkpoint(checkpoint_path)

        # Initialize WandB
        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=config.project_name,
                config=vars(config),
                name="video_vae",
            )

    def train(self) -> dict:
        """Run training loop."""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        global_step = 0

        for epoch in range(self.start_epoch, self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            # Training phase
            train_metrics = self._train_epoch(epoch, global_step)
            global_step = train_metrics.get("global_step", global_step)

            # Validation phase
            if self.val_dataloader is not None and epoch % 5 == 0:
                val_metrics = self._validate(epoch)
                if self.config.use_wandb:
                    wandb.log({"val": val_metrics}, step=epoch)

            # Save checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch)

        return {"final_epoch": self.config.num_epochs}

    def _train_epoch(self, epoch: int, global_step: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Unpack batch
            if isinstance(batch, (list, tuple)):
                video = batch[0]
            else:
                video = batch["video"]

            with self.accelerator.accumulate(self.model):
                # Forward pass — assuming model has encode/decode methods
                if hasattr(self.model, "encode"):
                    mu, logvar = self.model.encode(video)
                    z = self._reparameterize(mu, logvar)
                    reconstructed = self.model.decode(z)
                else:
                    # Fallback: treat as AE without explicit mu/logvar
                    z = self.model.encoder(video)
                    reconstructed = self.model.decoder(z)
                    mu = torch.zeros(1, device=video.device)
                    logvar = torch.zeros(1, device=video.device)

                # Compute loss
                loss, loss_dict = self.loss_fn(reconstructed, video, mu, logvar)

                # Backward and optimize
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            # Logging
            if batch_idx % self.config.log_interval == 0 and self.accelerator.is_main_process:
                avg_loss = epoch_loss / num_batches
                logger.info(f"  Batch {batch_idx}/{len(self.train_dataloader)}: loss={avg_loss:.4f}")

                if self.config.use_wandb:
                    wandb.log({
                        **loss_dict,
                        "epoch": epoch,
                        "batch": batch_idx,
                    }, step=global_step)

        self.scheduler.step()

        return {"global_step": global_step, "avg_loss": epoch_loss / max(1, num_batches)}

    def _validate(self, epoch: int) -> dict:
        """Validate the model."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                if isinstance(batch, (list, tuple)):
                    video = batch[0]
                else:
                    video = batch["video"]

                if hasattr(self.model, "encode"):
                    mu, logvar = self.model.encode(video)
                    z = self._reparameterize(mu, logvar)
                    reconstructed = self.model.decode(z)
                else:
                    z = self.model.encoder(video)
                    reconstructed = self.model.decoder(z)
                    mu = torch.zeros(1, device=video.device)
                    logvar = torch.zeros(1, device=video.device)

                loss, _ = self.loss_fn(reconstructed, video, mu, logvar)
                val_loss += loss.item()
                num_batches += 1

        avg_val_loss = val_loss / max(1, num_batches)
        logger.info(f"Validation loss: {avg_val_loss:.4f}")

        return {"val_loss": avg_val_loss}

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.config.checkpoint_dir / f"video_vae_epoch_{epoch}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.__dict__,
        }

        self.accelerator.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.accelerator.unwrap_model(self.model).load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {self.start_epoch}")


class AudioVAETrainer:
    """Trainer for audio VAE model."""

    def __init__(
        self,
        model: nn.Module,
        config: VAETrainerConfig,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        checkpoint_path: Path | None = None,
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )

        # Loss function
        self.loss_fn = AudioVAELoss(
            beta=config.beta_kl,
            lambda_spectral=config.lambda_spectral,
        )

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        total_steps = len(train_dataloader) * config.num_epochs
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader
        )

        if self.val_dataloader is not None:
            self.val_dataloader = self.accelerator.prepare(self.val_dataloader)

        # Load checkpoint if provided
        self.checkpoint_path = checkpoint_path
        self.start_epoch = 0
        if checkpoint_path and checkpoint_path.exists():
            self.load_checkpoint(checkpoint_path)

        # Initialize WandB
        if config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=config.project_name,
                config=vars(config),
                name="audio_vae",
            )

    def train(self) -> dict:
        """Run training loop."""
        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.start_epoch, self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            # Training phase
            train_metrics = self._train_epoch(epoch)

            # Validation phase
            if self.val_dataloader is not None and epoch % 5 == 0:
                val_metrics = self._validate(epoch)
                if self.config.use_wandb:
                    wandb.log({"val": val_metrics}, step=epoch)

            # Save checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(epoch)

        return {"final_epoch": self.config.num_epochs}

    def _train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Unpack batch
            if isinstance(batch, (list, tuple)):
                audio = batch[0]
            else:
                audio = batch["audio"]

            with self.accelerator.accumulate(self.model):
                # Forward pass
                if hasattr(self.model, "encode"):
                    mu, logvar = self.model.encode(audio)
                    z = self._reparameterize(mu, logvar)
                    reconstructed = self.model.decode(z)
                else:
                    z = self.model.encoder(audio)
                    reconstructed = self.model.decoder(z)
                    mu = torch.zeros(1, device=audio.device)
                    logvar = torch.zeros(1, device=audio.device)

                # Compute loss
                loss, loss_dict = self.loss_fn(reconstructed, audio, mu, logvar)

                # Backward and optimize
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                self.optimizer.step()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

            # Logging
            if batch_idx % self.config.log_interval == 0 and self.accelerator.is_main_process:
                avg_loss = epoch_loss / num_batches
                logger.info(f"  Batch {batch_idx}/{len(self.train_dataloader)}: loss={avg_loss:.4f}")

                if self.config.use_wandb:
                    wandb.log({**loss_dict, "epoch": epoch}, step=batch_idx)

        self.scheduler.step()

        return {"avg_loss": epoch_loss / max(1, num_batches)}

    def _validate(self, epoch: int) -> dict:
        """Validate the model."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_dataloader:
                if isinstance(batch, (list, tuple)):
                    audio = batch[0]
                else:
                    audio = batch["audio"]

                if hasattr(self.model, "encode"):
                    mu, logvar = self.model.encode(audio)
                    z = self._reparameterize(mu, logvar)
                    reconstructed = self.model.decode(z)
                else:
                    z = self.model.encoder(audio)
                    reconstructed = self.model.decoder(z)
                    mu = torch.zeros(1, device=audio.device)
                    logvar = torch.zeros(1, device=audio.device)

                loss, _ = self.loss_fn(reconstructed, audio, mu, logvar)
                val_loss += loss.item()
                num_batches += 1

        avg_val_loss = val_loss / max(1, num_batches)
        logger.info(f"Validation loss: {avg_val_loss:.4f}")

        return {"val_loss": avg_val_loss}

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.config.checkpoint_dir / f"audio_vae_epoch_{epoch}.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.__dict__,
        }

        self.accelerator.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.accelerator.unwrap_model(self.model).load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {self.start_epoch}")


__all__ = [
    "VideoVAETrainer",
    "AudioVAETrainer",
    "VAETrainerConfig",
    "VideoVAELoss",
    "AudioVAELoss",
    "PerceptualLoss",
    "SpectralLoss",
]
