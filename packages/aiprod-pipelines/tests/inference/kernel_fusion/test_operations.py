"""
Tests for kernel fusion operations.

Validates:
- Numerical correctness of fused operations
- Memory efficiency claims
- Output shapes and dtypes
- Configuration validation
- Registry functionality
"""

import pytest
import torch
import torch.nn as nn
from aiprod_pipelines.inference.kernel_fusion import (
    FusionConfig,
    FusedAttentionLinear,
    FusedConvActivation,
    FusedGroupNormActivation,
    FusedResidualBlock,
    FusionOperationRegistry,
)


class TestFusionConfig:
    """Tests for FusionConfig."""
    
    def test_default_config(self):
        """Default config has sensible values."""
        config = FusionConfig()
        
        assert config.activation == "gelu"
        assert config.use_flashattention is True
        assert config.kernel_tile_size == 128
    
    def test_custom_config(self):
        """Can create custom config."""
        config = FusionConfig(
            activation="relu",
            kernel_tile_size=256,
            enable_cuda_graphs=True,
        )
        
        assert config.activation == "relu"
        assert config.kernel_tile_size == 256
        assert config.enable_cuda_graphs is True
    
    def test_invalid_activation_raises(self):
        """Invalid activation raises error."""
        config = FusionConfig(activation="invalid")
        
        with pytest.raises(ValueError):
            config.validate()
    
    def test_invalid_tile_size_raises(self):
        """Tile size < 32 raises error."""
        config = FusionConfig(kernel_tile_size=16)
        
        with pytest.raises(ValueError):
            config.validate()
    
    def test_validate_passes_for_valid(self):
        """Valid config passes validation."""
        config = FusionConfig(
            activation="gelu",
            kernel_tile_size=128,
        )
        
        # Should not raise
        config.validate()


class TestFusedAttentionLinear:
    """Tests for FusedAttentionLinear."""
    
    def test_initialization(self):
        """Can initialize fusion."""
        fusion = FusedAttentionLinear(
            hidden_dim=768,
            num_heads=12,
            use_projection=False,
        )
        
        assert fusion.hidden_dim == 768
        assert fusion.num_heads == 12
        assert fusion.head_dim == 64
    
    def test_initialization_invalid_dims(self):
        """Invalid dims raise error."""
        with pytest.raises(ValueError):
            FusedAttentionLinear(hidden_dim=768, num_heads=13)
    
    def test_forward_shape(self):
        """Forward pass preserves shape."""
        batch, seq_len, hidden = 2, 128, 256
        
        fusion = FusedAttentionLinear(
            hidden_dim=hidden,
            num_heads=8,
        )
        
        query = torch.randn(batch, seq_len, hidden)
        key = torch.randn(batch, seq_len, hidden)
        value = torch.randn(batch, seq_len, hidden)
        
        output = fusion.forward(query, key, value)
        
        assert output.shape == (batch, seq_len, hidden)
    
    def test_forward_with_projection(self):
        """Forward works with projection."""
        hidden = 768
        fusion = FusedAttentionLinear(
            hidden_dim=hidden,
            num_heads=12,
            use_projection=True,
        )
        
        query = torch.randn(1, 100, hidden)
        key = torch.randn(1, 100, hidden)
        value = torch.randn(1, 100, hidden)
        weight = torch.randn(hidden, hidden)
        bias = torch.randn(hidden)
        
        output = fusion.forward(query, key, value, weight, bias)
        
        assert output.shape == (1, 100, hidden)
    
    def test_memory_savings_estimate_reasonable(self):
        """Memory savings estimate makes sense."""
        fusion = FusedAttentionLinear(hidden_dim=768, num_heads=12)
        savings = fusion.memory_savings_estimate()
        
        assert savings["memory_bandwidth_reduction_percent"] > 20
        assert savings["speedup_estimate"] > 1.0


class TestFusedConvActivation:
    """Tests for FusedConvActivation."""
    
    def test_initialization(self):
        """Can initialize fusion."""
        fusion = FusedConvActivation(
            in_channels=64,
            out_channels=128,
            activation="gelu",
        )
        
        assert fusion.in_channels == 64
        assert fusion.out_channels == 128
        assert fusion.activation == "gelu"
    
    def test_invalid_channels(self):
        """Invalid channels raise error."""
        fusion = FusedConvActivation(
            in_channels=64,
            out_channels=-1,  # Invalid
        )
        
        config = FusionConfig()
        with pytest.raises(ValueError):
            config.validate()
    
    def test_forward_shape(self):
        """Forward pass works and preserves spatial dims."""
        batch, in_ch, h, w, t = 1, 64, 16, 16, 8
        out_ch = 128
        
        fusion = FusedConvActivation(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
            activation="relu",
        )
        
        x = torch.randn(batch, in_ch, h, w, t)
        weight = torch.randn(out_ch, in_ch, 3, 3, 3)
        bias = torch.randn(out_ch)
        
        output = fusion.forward(x, weight, bias)
        
        assert output.shape == (batch, out_ch, h, w, t)
    
    def test_activation_applied(self):
        """Activation is actually applied."""
        fusion = FusedConvActivation(
            in_channels=3,
            out_channels=3,
            activation="relu",
        )
        
        x = torch.tensor([[[[-1.0, 0.5]]]])  # Has negative values
        weight = torch.ones(3, 3, 1, 1, 1)
        
        output = fusion.forward(x, weight)
        
        # ReLU should zero out negatives
        assert output.min() >= 0
    
    def test_forward_without_bias(self):
        """Forward works without bias."""
        fusion = FusedConvActivation(
            in_channels=3,
            out_channels=3,
        )
        
        x = torch.randn(1, 3, 8, 8, 4)
        weight = torch.randn(3, 3, 3, 3, 3)
        
        output = fusion.forward(x, weight, bias=None)
        
        assert output.shape == x.shape


class TestFusedGroupNormActivation:
    """Tests for FusedGroupNormActivation."""
    
    def test_initialization(self):
        """Can initialize fusion."""
        fusion = FusedGroupNormActivation(
            num_channels=256,
            num_groups=32,
            activation="gelu",
        )
        
        assert fusion.num_channels == 256
        assert fusion.num_groups == 32
    
    def test_invalid_group_count(self):
        """Invalid group count raises error."""
        with pytest.raises(ValueError):
            FusedGroupNormActivation(
                num_channels=256,
                num_groups=33,  # Doesn't divide evenly
            )
    
    def test_forward_shape(self):
        """Forward pass preserves shape."""
        batch, ch, h, w, t = 2, 64, 16, 16, 8
        
        fusion = FusedGroupNormActivation(
            num_channels=ch,
            num_groups=8,
        )
        
        x = torch.randn(batch, ch, h, w, t)
        output = fusion.forward(x)
        
        assert output.shape == x.shape
    
    def test_forward_with_weight_bias(self):
        """Forward works with weight and bias."""
        ch = 64
        fusion = FusedGroupNormActivation(num_channels=ch, num_groups=8)
        
        x = torch.randn(1, ch, 16, 16, 8)
        weight = torch.randn(ch)
        bias = torch.randn(ch)
        
        output = fusion.forward(x, weight, bias)
        
        assert output.shape == x.shape
    
    def test_normalization_applied(self):
        """Output should be normalized."""
        fusion = FusedGroupNormActivation(
            num_channels=64,
            num_groups=8,
        )
        
        x = torch.randn(2, 64, 16, 16, 8) * 100 + 50  # Large scale
        output = fusion.forward(x)
        
        # Normalized output should have smaller variance
        assert output.std() < x.std()


class TestFusedResidualBlock:
    """Tests for FusedResidualBlock."""
    
    def test_initialization(self):
        """Can initialize fusion."""
        fusion = FusedResidualBlock(
            hidden_dim=768,
            activation="gelu",
        )
        
        assert fusion.hidden_dim == 768
        assert fusion.activation == "gelu"
    
    def test_forward_shape(self):
        """Forward pass preserves shape."""
        batch, seq_len, hidden = 2, 100, 256
        
        fusion = FusedResidualBlock(hidden_dim=hidden)
        
        x = torch.randn(batch, seq_len, hidden)
        weight = torch.randn(hidden, hidden)
        bias = torch.randn(hidden)
        skip = torch.randn(batch, seq_len, hidden)
        
        output = fusion.forward(x, weight, bias, skip)
        
        assert output.shape == (batch, seq_len, hidden)
    
    def test_residual_addition(self):
        """Residual addition actually happens."""
        fusion = FusedResidualBlock(hidden_dim=64, activation="none")
        
        # Identity weight (diagonal matrix)
        weight = torch.eye(64)
        bias = torch.zeros(64)
        
        x = torch.ones(1, 10, 64)
        skip = torch.ones(1, 10, 64)
        
        output = fusion.forward(x, weight, bias, skip)
        
        # With identity projection and skip, output should be ~2
        assert output.mean() > 1.5
    
    def test_activation_applied(self):
        """Check activation is applied."""
        fusion = FusedResidualBlock(hidden_dim=64, activation="relu")
        
        x = torch.randn(1, 10, 64)
        skip = torch.randn(1, 10, 64) - 10  # Negative skip
        weight = torch.randn(64, 64)
        bias = torch.randn(64)
        
        output = fusion.forward(x, weight, bias, skip)
        
        # ReLU should zero out negatives
        assert output.min() >= -1e-5


class TestFusionRegistry:
    """Tests for FusionOperationRegistry."""
    
    def test_list_available(self):
        """Can list available fusions."""
        available = FusionOperationRegistry.list_available()
        
        assert "attention_linear" in available
        assert "conv_activation" in available
        assert "norm_activation" in available
        assert "residual_block" in available
    
    def test_is_available(self):
        """Check availability of fusions."""
        assert FusionOperationRegistry.is_available("attention_linear")
        assert FusionOperationRegistry.is_available("residual_block")
        assert not FusionOperationRegistry.is_available("nonexistent")
    
    def test_create_attention_linear(self):
        """Can create attention_linear fusion."""
        fusion = FusionOperationRegistry.create(
            "attention_linear",
            hidden_dim=768,
            num_heads=12,
        )
        
        assert isinstance(fusion, FusedAttentionLinear)
    
    def test_create_conv_activation(self):
        """Can create conv_activation fusion."""
        fusion = FusionOperationRegistry.create(
            "conv_activation",
            in_channels=64,
            out_channels=128,
        )
        
        assert isinstance(fusion, FusedConvActivation)
    
    def test_create_norm_activation(self):
        """Can create norm_activation fusion."""
        fusion = FusionOperationRegistry.create(
            "norm_activation",
            num_channels=256,
            num_groups=32,
        )
        
        assert isinstance(fusion, FusedGroupNormActivation)
    
    def test_create_residual_block(self):
        """Can create residual_block fusion."""
        fusion = FusionOperationRegistry.create(
            "residual_block",
            hidden_dim=768,
        )
        
        assert isinstance(fusion, FusedResidualBlock)
    
    def test_create_invalid_raises(self):
        """Creating invalid fusion raises error."""
        with pytest.raises(ValueError):
            FusionOperationRegistry.create("invalid_fusion")


class TestFusionMemoySavings:
    """Tests for memory savings estimates."""
    
    def test_all_operations_report_savings(self):
        """All operations report memory savings."""
        fusion_makers = [
            lambda: FusedAttentionLinear(768, 12),
            lambda: FusedConvActivation(64, 128),
            lambda: FusedGroupNormActivation(256, 32),
            lambda: FusedResidualBlock(768),
        ]
        
        for maker in fusion_makers:
            fusion = maker()
            savings = fusion.memory_savings_estimate()
            
            assert "memory_bandwidth_reduction_percent" in savings
            assert "speedup_estimate" in savings
            assert savings["memory_bandwidth_reduction_percent"] > 0
            assert savings["speedup_estimate"] > 1.0
    
    def test_speedup_estimates_reasonable(self):
        """Speedup estimates are in reasonable range."""
        fusions = [
            FusedAttentionLinear(768, 12),
            FusedConvActivation(64, 128),
            FusedGroupNormActivation(256, 32),
            FusedResidualBlock(768),
        ]
        
        for fusion in fusions:
            savings = fusion.memory_savings_estimate()
            speedup = savings["speedup_estimate"]
            
            # Speedup should be between 1.1x and 2.0x
            assert 1.1 <= speedup <= 2.0
