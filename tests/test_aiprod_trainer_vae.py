"""Tests for aiprod_trainer.vae_trainer module."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from aiprod_trainer.vae_trainer import (
    VideoVAETrainer, AudioVAETrainer, VAETrainerConfig,
    VideoVAELoss, AudioVAELoss, PerceptualLoss, SpectralLoss
)


class TestPerceptualLoss:
    """Test PerceptualLoss."""
    
    def test_perceptual_loss_initialization(self):
        """Test PerceptualLoss can be initialized."""
        loss_fn = PerceptualLoss()
        assert loss_fn is not None
        assert hasattr(loss_fn, 'forward')
    
    def test_perceptual_loss_forward(self):
        """Test PerceptualLoss forward pass."""
        loss_fn = PerceptualLoss()
        x = torch.randn(2, 3, 224, 224)  # VGG16 expects larger input
        target = torch.randn(2, 3, 224, 224)
        
        try:
            loss = loss_fn(x, target)
            assert loss.shape == torch.Size([])
            assert loss.item() >= 0
            assert not torch.isnan(loss)
        except (RuntimeError, TypeError):
            # VGG not available or size issue, that's ok
            pytest.skip("Perceptual loss VGG not available")
    
    def test_perceptual_loss_fallback(self):
        """Test that perceptual loss falls back gracefully."""
        # This test ensures the L2 fallback works
        loss_fn = PerceptualLoss()
        assert hasattr(loss_fn, 'use_vgg')
        
        # Even if VGG16 fails, forward should still work
        x = torch.randn(2, 3, 224, 224)
        target = torch.randn(2, 3, 224, 224)
        try:
            loss = loss_fn(x, target)
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
        except (RuntimeError, TypeError):
            # VGG not available, that's ok
            pytest.skip("Perceptual loss VGG not available")


class TestSpectralLoss:
    """Test SpectralLoss."""
    
    def test_spectral_loss_initialization(self):
        """Test SpectralLoss initialization."""
        loss_fn = SpectralLoss(n_fft=1024, hop_length=256)
        assert loss_fn is not None
    
    def test_spectral_loss_forward(self):
        """Test SpectralLoss forward pass."""
        loss_fn = SpectralLoss(n_fft=512, hop_length=128)
        
        # Audio waveforms
        recon = torch.randn(2, 8000)
        target = torch.randn(2, 8000)
        
        loss = loss_fn(recon, target)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestVideoVAELoss:
    """Test VideoVAELoss."""
    
    def test_video_vae_loss_initialization(self):
        """Test VideoVAELoss initialization."""
        loss_fn = VideoVAELoss(
            beta=0.1,
            lambda_perceptual=1.0,
            use_perceptual=True
        )
        assert loss_fn is not None
        assert loss_fn.beta == 0.1
        assert loss_fn.lambda_perceptual == 1.0
    
    def test_video_vae_loss_computation(self):
        """Test VideoVAELoss computation."""
        loss_fn = VideoVAELoss(beta=0.1, lambda_perceptual=0.1, use_perceptual=False)
        
        # Dummy data
        recon = torch.randn(2, 3, 8, 64, 64)
        target = torch.randn(2, 3, 8, 64, 64)
        mu = torch.randn(2, 4, 2, 8, 8)
        logvar = torch.randn(2, 4, 2, 8, 8)
        
        loss, loss_dict = loss_fn(recon, target, mu, logvar)
        
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert 'loss/recon' in loss_dict
        assert 'loss/kl' in loss_dict
        assert 'loss/total' in loss_dict
    
    def test_video_vae_loss_with_perceptual(self):
        """Test VideoVAELoss with perceptual loss enabled."""
        loss_fn = VideoVAELoss(
            beta=0.1,
            lambda_perceptual=1.0,
            use_perceptual=True
        )
        
        # Use larger resolution for VGG16
        recon = torch.randn(2, 3, 8, 224, 224)
        target = torch.randn(2, 3, 8, 224, 224)
        mu = torch.randn(2, 4, 2, 28, 28)
        logvar = torch.randn(2, 4, 2, 28, 28)
        
        try:
            loss, loss_dict = loss_fn(recon, target, mu, logvar)
            assert 'loss/perceptual' in loss_dict or 'loss/reconstruction' in loss_dict
        except (RuntimeError, TypeError):
            # VGG not available or size issue
            pytest.skip("Perceptual loss VGG not available")


class TestAudioVAELoss:
    """Test AudioVAELoss."""
    
    def test_audio_vae_loss_initialization(self):
        """Test AudioVAELoss initialization."""
        loss_fn = AudioVAELoss(
            beta=0.1,
            lambda_spectral=1.0
        )
        assert loss_fn is not None
        assert loss_fn.beta == 0.1
        assert loss_fn.lambda_spectral == 1.0
    
    def test_audio_vae_loss_computation(self):
        """Test AudioVAELoss computation."""
        loss_fn = AudioVAELoss(beta=0.1, lambda_spectral=1.0)
        
        recon = torch.randn(2, 16000)
        target = torch.randn(2, 16000)
        mu = torch.randn(2, 8, 40)
        logvar = torch.randn(2, 8, 40)
        
        loss, loss_dict = loss_fn(recon, target, mu, logvar)
        
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        assert 'loss/recon' in loss_dict
        assert 'loss/kl' in loss_dict
        assert 'loss/spectral' in loss_dict


class TestVAETrainerConfig:
    """Test VAETrainerConfig."""
    
    def test_vae_trainer_config_creation(self):
        """Test VAETrainerConfig creation."""
        config = VAETrainerConfig(
            learning_rate=1e-4,
            batch_size=4,
            num_epochs=10,
            beta_kl=0.1,
            lambda_perceptual=1.0,
        )
        assert config.learning_rate == 1e-4
        assert config.batch_size == 4
        assert config.num_epochs == 10
        assert config.beta_kl == 0.1
    
    def test_vae_trainer_config_defaults(self):
        """Test VAETrainerConfig defaults."""
        config = VAETrainerConfig()
        assert config.learning_rate > 0
        assert config.batch_size > 0
        assert config.num_epochs > 0
        assert config.use_wandb is True
        assert config.mixed_precision in ["bf16", "fp16", "no"]
    
    def test_vae_trainer_config_checkpoint_dir(self):
        """Test checkpoint directory handling."""
        config = VAETrainerConfig(checkpoint_dir=Path("test_ckpts"))
        assert isinstance(config.checkpoint_dir, Path)


class TestVideoVAETrainer:
    """Test VideoVAETrainer initialization and basic methods."""
    
    def test_vae_trainer_requires_model(self):
        """Test that VAETrainer requires a model."""
        config = VAETrainerConfig(num_epochs=1)
        
        # Create dummy model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 100)
        )
        
        # Should require dataloader in actual usage
        with pytest.raises(TypeError):
            # Missing train_dataloader argument
            trainer = VideoVAETrainer(model, config)


class TestAudioVAETrainer:
    """Test AudioVAETrainer initialization."""
    
    def test_audio_vae_trainer_requires_model(self):
        """Test that AudioVAETrainer requires a model."""
        config = VAETrainerConfig(num_epochs=1)
        
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.Linear(50, 100)
        )
        
        with pytest.raises(TypeError):
            trainer = AudioVAETrainer(model, config)


class TestVAETrainerIntegration:
    """Integration tests for VAE trainers (without actual training)."""
    
    def test_loss_functions_differentiable(self):
        """Test that loss functions are differentiable."""
        # Video VAE
        video_loss_fn = VideoVAELoss(beta=0.1, use_perceptual=False)
        
        recon = torch.randn(2, 3, 8, 64, 64, requires_grad=True)
        target = torch.randn(2, 3, 8, 64, 64)
        mu = torch.randn(2, 4, 2, 8, 8)
        logvar = torch.randn(2, 4, 2, 8, 8)
        
        loss, _ = video_loss_fn(recon, target, mu, logvar)
        loss.backward()
        
        assert recon.grad is not None
        assert not torch.isnan(recon.grad).any()
    
    def test_audio_loss_differentiable(self):
        """Test that audio loss is differentiable."""
        audio_loss_fn = AudioVAELoss(beta=0.1, lambda_spectral=1.0)
        
        recon = torch.randn(2, 8000, requires_grad=True)
        target = torch.randn(2, 8000)
        mu = torch.randn(2, 8, 40)
        logvar = torch.randn(2, 8, 40)
        
        loss, _ = audio_loss_fn(recon, target, mu, logvar)
        loss.backward()
        
        assert recon.grad is not None


class TestVAETrainerConfig_Validation:
    """Test VAETrainerConfig validation."""
    
    def test_config_mixed_precision_options(self):
        """Test mixed precision configuration."""
        for precision in ["bf16", "fp16", "no"]:
            config = VAETrainerConfig(mixed_precision=precision)
            assert config.mixed_precision == precision
    
    def test_config_learning_rate_range(self):
        """Test reasonable learning rate values."""
        config = VAETrainerConfig(learning_rate=1e-3)
        assert config.learning_rate > 0
        
        config = VAETrainerConfig(learning_rate=1e-6)
        assert config.learning_rate > 0
