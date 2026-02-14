"""
Unit tests for latent distillation core components.

Coverage:
  - LatentMetrics dataclass
  - LatentCompressionConfig
  - LatentEncoder forward pass and quantization
  - LatentDecoder reconstruction
  - LatentDistillationEngine compression/decompression
  - Metrics computation
"""

import pytest
import torch
import torch.nn.functional as F

from aiprod_pipelines.inference.latent_distillation import (
    LatentMetrics,
    LatentCompressionConfig,
    LatentEncoder,
    LatentDecoder,
    LatentDistillationEngine,
)


class TestLatentMetrics:
    """Tests for LatentMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating LatentMetrics."""
        metrics = LatentMetrics(
            original_size_mb=8.0,
            compressed_size_mb=1.0,
            compression_ratio=8.0,
            reconstruction_mse=0.01,
            reconstruction_ssim=0.95,
            quality_retention_percent=98.0,
            compression_time_ms=5.0,
            decompression_time_ms=3.0,
        )
        
        assert metrics.original_size_mb == 8.0
        assert metrics.compression_ratio == 8.0
        assert metrics.memory_saved_mb == 7.0
    
    def test_metrics_ranges(self):
        """Test metrics have reasonable values."""
        metrics = LatentMetrics(
            original_size_mb=8.0,
            compressed_size_mb=1.0,
            compression_ratio=8.0,
            reconstruction_mse=0.01,
            reconstruction_ssim=0.95,
            quality_retention_percent=98.0,
            compression_time_ms=5.0,
            decompression_time_ms=3.0,
        )
        
        assert metrics.compression_ratio >= 1.0
        assert 0 <= metrics.reconstruction_ssim <= 1
        assert 0 <= metrics.quality_retention_percent <= 100
        assert metrics.compression_time_ms >= 0
        assert metrics.memory_saved_mb >= 0


class TestLatentCompressionConfig:
    """Tests for LatentCompressionConfig."""
    
    def test_config_defaults(self):
        """Test default configuration."""
        config = LatentCompressionConfig()
        
        assert config.codebook_size == 512
        assert config.embedding_dim == 64
        assert config.num_quantizers == 4
        assert config.ema_decay == 0.99
        assert config.commitment_loss_weight == 0.25
    
    def test_config_custom(self):
        """Test custom configuration."""
        config = LatentCompressionConfig(
            codebook_size=1024,
            embedding_dim=128,
            num_quantizers=8,
            ema_decay=0.95,
        )
        
        assert config.codebook_size == 1024
        assert config.embedding_dim == 128
        assert config.num_quantizers == 8
        assert config.ema_decay == 0.95


class TestLatentEncoderNetwork:
    """Tests for LatentEncoder neural network."""
    
    def test_encoder_init(self, compression_config):
        """Test encoder initialization."""
        encoder = LatentEncoder(compression_config)
        
        assert encoder.config is not None
        assert len(encoder.codebooks) == compression_config.num_quantizers
    
    def test_encoder_forward_shape(self, sample_latents, compression_config):
        """Test encoder forward pass output shape."""
        encoder = LatentEncoder(compression_config)
        encoder.eval()
        
        with torch.no_grad():
            codes, metrics = encoder(sample_latents)
        
        batch_size, channels, height, width = sample_latents.shape
        
        # Codes shape: [num_quantizers, batch, height, width]
        assert codes.shape == (
            compression_config.num_quantizers,
            batch_size,
            height,
            width,
        )
        
        # Check ranges
        assert (codes >= 0).all() and (codes < compression_config.codebook_size).all()
    
    def test_encoder_metrics(self, sample_latents, compression_config):
        """Test encoder outputs metrics."""
        encoder = LatentEncoder(compression_config)
        encoder.eval()
        
        with torch.no_grad():
            codes, metrics = encoder(sample_latents)
        
        assert "loss" in metrics
        assert "perplexity" in metrics
        assert "used_codes" in metrics
        
        assert metrics["loss"].shape == torch.Size([])
        assert isinstance(metrics["perplexity"], float)
        assert isinstance(metrics["used_codes"], int)
    
    def test_encoder_gradient_flow(self, sample_latents, compression_config):
        """Test gradient flow through encoder."""
        encoder = LatentEncoder(compression_config)
        encoder.train()
        
        codes, metrics = encoder(sample_latents)
        
        loss = metrics["loss"]
        loss.backward()
        
        # Check gradients exist
        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_encoder_different_input_sizes(self, compression_config):
        """Test encoder with different input sizes."""
        encoder = LatentEncoder(compression_config)
        encoder.eval()
        
        for height, width in [(16, 16), (32, 32), (64, 64)]:
            x = torch.randn(2, 4, height, width)
            
            with torch.no_grad():
                codes, _ = encoder(x)
            
            assert codes.shape == (compression_config.num_quantizers, 2, height, width)
    
    def test_encoder_batch_processing(self, compression_config):
        """Test encoder with different batch sizes."""
        encoder = LatentEncoder(compression_config)
        encoder.eval()
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 4, 32, 32)
            
            with torch.no_grad():
                codes, _ = encoder(x)
            
            assert codes.shape[1] == batch_size


class TestLatentDecoderNetwork:
    """Tests for LatentDecoder neural network."""
    
    def test_decoder_init(self, compression_config):
        """Test decoder initialization."""
        decoder = LatentDecoder(compression_config)
        
        assert decoder.config is not None
        assert len(decoder.codebooks) == compression_config.num_quantizers
    
    def test_decoder_forward_shape(self, compression_config):
        """Test decoder forward pass output shape."""
        decoder = LatentDecoder(compression_config)
        decoder.eval()
        
        # Create dummy codes
        codes = torch.randint(0, compression_config.codebook_size, (4, 2, 32, 32))
        
        with torch.no_grad():
            latents = decoder(codes)
        
        # Should output [batch, 4, height, width]
        assert latents.shape == (2, 4, 32, 32)
    
    def test_decoder_with_shared_codebooks(self, compression_config):
        """Test decoder using shared codebooks from encoder."""
        encoder = LatentEncoder(compression_config)
        decoder = LatentDecoder(compression_config)
        
        # Share codebooks
        decoder.codebooks = encoder.codebooks
        
        # Create dummy codes
        codes = torch.randint(0, compression_config.codebook_size, (4, 2, 32, 32))
        
        with torch.no_grad():
            latents = decoder(codes, encoder.codebooks)
        
        assert latents.shape == (2, 4, 32, 32)
    
    def test_decoder_different_output_sizes(self, compression_config):
        """Test decoder with different spatial sizes."""
        decoder = LatentDecoder(compression_config)
        decoder.eval()
        
        for height, width in [(16, 16), (32, 32), (64, 64)]:
            codes = torch.randint(0, compression_config.codebook_size, (4, 2, height, width))
            
            with torch.no_grad():
                latents = decoder(codes)
            
            assert latents.shape == (2, 4, height, width)


class TestLatentDistillationEngine:
    """Tests for LatentDistillationEngine wrapper."""
    
    def test_engine_init(self):
        """Test engine initialization."""
        engine = LatentDistillationEngine()
        
        assert engine.encoder is not None
        assert engine.decoder is not None
    
    def test_engine_compress(self, sample_latents):
        """Test compression."""
        engine = LatentDistillationEngine()
        engine.to(torch.device("cpu"))
        
        codes = engine.compress(sample_latents)
        
        # Codes should be smaller
        assert codes.dtype == torch.int64 or codes.dtype == torch.long
        assert codes.numel() < sample_latents.numel()
    
    def test_engine_decompress(self, sample_latents):
        """Test decompression."""
        engine = LatentDistillationEngine()
        engine.to(torch.device("cpu"))
        
        codes = engine.compress(sample_latents)
        recon = engine.decompress(codes)
        
        # Reconstruction should match original shape
        assert recon.shape == sample_latents.shape
        assert recon.dtype == torch.float32
    
    def test_engine_round_trip(self, sample_latents):
        """Test compress → decompress round trip."""
        engine = LatentDistillationEngine()
        engine.to(torch.device("cpu"))
        
        codes = engine.compress(sample_latents)
        recon = engine.decompress(codes)
        
        # Check reconstruction quality
        mse = F.mse_loss(sample_latents, recon).item()
        
        # Should have some reconstruction error but not huge
        assert mse < 1.0  # Reasonable quality
    
    def test_engine_compression_ratio(self, sample_latents):
        """Test compression achieves target ratio."""
        engine = LatentDistillationEngine()
        engine.to(torch.device("cpu"))
        
        codes = engine.compress(sample_latents)
        metrics = engine.compute_metrics(sample_latents, codes)
        
        # Should achieve at least 2x compression
        assert metrics.compression_ratio >= 2.0
    
    def test_engine_quality_retention(self, sample_latents):
        """Test reconstruction quality."""
        engine = LatentDistillationEngine()
        engine.to(torch.device("cpu"))
        
        codes = engine.compress(sample_latents)
        metrics = engine.compute_metrics(sample_latents, codes)
        
        # Should maintain >90% quality
        assert metrics.quality_retention_percent >= 90.0
    
    def test_engine_on_different_latents(self):
        """Test engine on different types of latents."""
        engine = LatentDistillationEngine()
        engine.to(torch.device("cpu"))
        
        # Test on various latent types
        for latents in [
            torch.randn(1, 4, 16, 16),  # Small
            torch.randn(8, 4, 32, 32),  # Larger batch
            torch.randn(2, 4, 64, 64),  # Larger spatial
            torch.ones(2, 4, 32, 32),   # Uniform
        ]:
            codes = engine.compress(latents)
            recon = engine.decompress(codes)
            
            assert recon.shape == latents.shape
    
    def test_engine_checkpointing(self, tmp_path, sample_latents):
        """Test saving and loading checkpoints."""
        engine1 = LatentDistillationEngine()
        engine1.to(torch.device("cpu"))
        
        checkpoint_path = str(tmp_path / "engine.pt")
        engine1.save_checkpoint(checkpoint_path)
        
        # Load into new engine
        engine2 = LatentDistillationEngine()
        engine2.load_checkpoint(checkpoint_path)
        
        # Should produce same compression
        codes1 = engine1.compress(sample_latents)
        codes2 = engine2.compress(sample_latents)
        
        assert torch.allclose(codes1.float(), codes2.float())


class TestCompressionMetrics:
    """Tests for compression metrics computation."""
    
    def test_metrics_computation(self, sample_latents):
        """Test metrics are computed correctly."""
        engine = LatentDistillationEngine()
        engine.to(torch.device("cpu"))
        
        codes = engine.compress(sample_latents)
        metrics = engine.compute_metrics(sample_latents, codes)
        
        # Check all metrics present
        assert isinstance(metrics.original_size_mb, float)
        assert isinstance(metrics.compressed_size_mb, float)
        assert isinstance(metrics.compression_ratio, float)
        assert isinstance(metrics.reconstruction_mse, float)
        assert isinstance(metrics.reconstruction_ssim, float)
        assert isinstance(metrics.quality_retention_percent, float)
    
    def test_metrics_value_ranges(self, sample_latents):
        """Test metrics have reasonable value ranges."""
        engine = LatentDistillationEngine()
        engine.to(torch.device("cpu"))
        
        codes = engine.compress(sample_latents)
        metrics = engine.compute_metrics(sample_latents, codes)
        
        # Sizes should be positive
        assert metrics.original_size_mb > 0
        assert metrics.compressed_size_mb > 0
        
        # Ratio should be > 1 (compression)
        assert metrics.compression_ratio > 1.0
        
        # MSE should be small
        assert metrics.reconstruction_mse >= 0
        
        # SSIM should be in [0, 1]
        assert 0 <= metrics.reconstruction_ssim <= 1
        
        # Quality should be reasonable
        assert 0 <= metrics.quality_retention_percent <= 100
    
    def test_ssim_computation(self):
        """Test SSIM is computed correctly."""
        engine = LatentDistillationEngine()
        
        # Identical tensors should have SSIM near 1
        x = torch.randn(1, 4, 32, 32)
        ssim1 = engine._compute_ssim(x, x)
        assert ssim1 > 0.99
        
        # Completely different should have low SSIM
        y = torch.randn(1, 4, 32, 32)
        ssim2 = engine._compute_ssim(x, y)
        assert 0 <= ssim2 < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
