"""Test LoRA layer implementations."""

import pytest
import torch
import torch.nn as nn
from aiprod_pipelines.inference.lora_tuning.lora_layers import (
    LoRAWeight,
    LoRALinear,
    LoRAConv2d,
    LoRAAdapter,
    LoRAComposer,
    LoRAMerger,
)
from aiprod_pipelines.inference.lora_tuning.lora_config import (
    LoRAConfig,
    LoRACompositionMode,
)


class TestLoRAWeight:
    """Test LoRA weight representation."""
    
    def test_weight_creation(self):
        """Test weight creation."""
        weight = LoRAWeight(
            name="layer1.weight",
            in_features=768,
            out_features=768,
            rank=8,
            alpha=1.0,
        )
        assert weight.name == "layer1.weight"
        assert weight.in_features == 768
        assert weight.out_features == 768
        assert weight.rank == 8
        assert weight.scaling == 1.0 / 8
    
    def test_weight_scaling(self):
        """Test weight scaling factor."""
        w1 = LoRAWeight("w1", 100, 100, 8, 1.0)
        w2 = LoRAWeight("w2", 100, 100, 8, 2.0)
        
        assert w1.scaling == 1.0 / 8
        assert w2.scaling == 2.0 / 8


class TestLoRALinear:
    """Test LoRA linear layer."""
    
    def test_lora_linear_creation(self):
        """Test LoRA linear layer creation."""
        base_layer = nn.Linear(100, 100)
        lora_layer = LoRALinear(base_layer, rank=8, alpha=1.0)
        
        assert lora_layer.base_layer == base_layer
        assert lora_layer.rank == 8
    
    def test_lora_linear_forward(self):
        """Test LoRA linear forward pass."""
        batch_size = 4
        in_features = 100
        
        base_layer = nn.Linear(in_features, in_features)
        lora_layer = LoRALinear(base_layer, rank=8)
        
        x = torch.randn(batch_size, in_features)
        output = lora_layer(x)
        
        assert output.shape == (batch_size, in_features)
    
    def test_lora_linear_parameters(self):
        """Test LoRA parameters are trainable."""
        base_layer = nn.Linear(100, 100)
        base_layer.eval()  # Freeze base
        
        lora_layer = LoRALinear(base_layer, rank=8)
        
        # Base layer frozen
        for param in base_layer.parameters():
            assert param.requires_grad is False
        
        # LoRA parameters trainable
        for name, param in lora_layer.named_parameters():
            if "lora_" in name:
                assert param.requires_grad is True
    
    def test_lora_linear_output_shape(self):
        """Test output dimensions."""
        base_layer = nn.Linear(10, 20)
        lora_layer = LoRALinear(base_layer, rank=4)
        
        for batch_size in [1, 2, 8]:
            x = torch.randn(batch_size, 10)
            output = lora_layer(x)
            assert output.shape == (batch_size, 20)
    
    def test_lora_linear_scaling(self):
        """Test LoRA scaling factor."""
        base_layer = nn.Linear(100, 100)
        
        # Low alpha: smaller LoRA contribution
        lora_low = LoRALinear(base_layer, rank=8, alpha=0.5)
        
        # High alpha: larger LoRA contribution
        lora_high = LoRALinear(base_layer, rank=8, alpha=2.0)
        
        assert lora_low.scaling == pytest.approx(0.5 / 8)
        assert lora_high.scaling == pytest.approx(2.0 / 8)


class TestLoRAConv2d:
    """Test LoRA convolutional layer."""
    
    def test_lora_conv2d_creation(self):
        """Test LoRA conv2d creation."""
        base_layer = nn.Conv2d(3, 64, kernel_size=3)
        lora_layer = LoRAConv2d(base_layer, rank=8)
        
        assert lora_layer.rank == 8
    
    def test_lora_conv2d_forward(self):
        """Test LoRA conv2d forward pass."""
        base_layer = nn.Conv2d(3, 64, kernel_size=3)
        lora_layer = LoRAConv2d(base_layer, rank=8)
        
        x = torch.randn(2, 3, 32, 32)
        output = lora_layer(x)
        
        assert output.shape[0] == 2  # Batch size
        assert output.shape[1] == 64  # Output channels
    
    def test_lora_conv2d_params_frozen(self):
        """Test base conv params are frozen."""
        base_layer = nn.Conv2d(3, 64, kernel_size=3)
        base_layer.eval()
        
        lora_layer = LoRAConv2d(base_layer, rank=8)
        
        # Base params frozen
        for param in base_layer.parameters():
            assert param.requires_grad is False


class TestLoRAAdapter:
    """Test LoRA adapter managing multiple layers."""
    
    def test_adapter_creation(self):
        """Test adapter creation."""
        adapter = LoRAAdapter(rank=8)
        assert adapter.rank == 8
        assert len(adapter.lora_layers) == 0
    
    def test_add_lora_linear(self):
        """Test adding LoRA linear layers."""
        adapter = LoRAAdapter(rank=8)
        base_layer = nn.Linear(100, 100)
        
        adapter.add_lora_linear("layer1", base_layer)
        
        assert "layer1" in adapter.lora_layers
        assert isinstance(adapter.lora_layers["layer1"], LoRALinear)
    
    def test_add_lora_conv2d(self):
        """Test adding LoRA conv layers."""
        adapter = LoRAAdapter(rank=8)
        base_layer = nn.Conv2d(3, 64, kernel_size=3)
        
        adapter.add_lora_conv2d("conv1", base_layer)
        
        assert "conv1" in adapter.lora_layers
        assert isinstance(adapter.lora_layers["conv1"], LoRAConv2d)
    
    def test_get_parameters(self):
        """Test getting all LoRA parameters."""
        adapter = LoRAAdapter(rank=8)
        adapter.add_lora_linear("layer1", nn.Linear(100, 100))
        adapter.add_lora_linear("layer2", nn.Linear(100, 100))
        
        params = list(adapter.get_parameters())
        assert len(params) > 0
    
    def test_get_parameter_count(self):
        """Test parameter counting."""
        adapter = LoRAAdapter(rank=8)
        adapter.add_lora_linear("layer1", nn.Linear(100, 100))
        
        param_count = adapter.get_parameter_count()
        # 2 matrices per layer (A and B)
        expected = 2 * 8 * 100  # 8 rank, 100 dims
        assert param_count >= expected - 100  # Allow some tolerance
    
    def test_multiple_adapters(self):
        """Test managing multiple adapters."""
        adapter1 = LoRAAdapter(rank=8)
        adapter1.add_lora_linear("layer1", nn.Linear(100, 100))
        
        adapter2 = LoRAAdapter(rank=16)
        adapter2.add_lora_linear("layer1", nn.Linear(100, 100))
        
        params1 = adapter1.get_parameter_count()
        params2 = adapter2.get_parameter_count()
        
        assert params2 > params1  # Rank 16 > Rank 8


class TestLoRAComposer:
    """Test LoRA adapter composition."""
    
    def test_sequential_composition(self):
        """Test sequential composition."""
        composer = LoRAComposer(
            num_adapters=2,
            composition_mode=LoRACompositionMode.sequential,
        )
        
        outputs = [torch.randn(2, 100), torch.randn(2, 100)]
        result = composer.compose_outputs(outputs)
        
        assert result.shape == outputs[0].shape
    
    def test_parallel_composition(self):
        """Test parallel composition."""
        composer = LoRAComposer(
            num_adapters=2,
            composition_mode=LoRACompositionMode.parallel,
        )
        
        outputs = [torch.randn(2, 100), torch.randn(2, 100)]
        result = composer.compose_outputs(outputs)
        
        assert result.shape == outputs[0].shape
    
    def test_gated_composition(self):
        """Test gated (learned weighted) composition."""
        composer = LoRAComposer(
            num_adapters=2,
            composition_mode=LoRACompositionMode.gated,
            hidden_dim=100,
        )
        
        outputs = [torch.randn(2, 100), torch.randn(2, 100)]
        result = composer.compose_outputs(outputs)
        
        assert result.shape == outputs[0].shape
    
    def test_conditional_composition(self):
        """Test conditional composition."""
        composer = LoRAComposer(
            num_adapters=2,
            composition_mode=LoRACompositionMode.conditional,
            hidden_dim=100,
        )
        
        outputs = [torch.randn(2, 100), torch.randn(2, 100)]
        result = composer.compose_outputs(outputs)
        
        assert result.shape == outputs[0].shape
    
    def test_get_total_parameters(self):
        """Test total parameter counting."""
        composer = LoRAComposer(
            num_adapters=2,
            composition_mode=LoRACompositionMode.gated,
            hidden_dim=100,
        )
        
        total_params = composer.get_total_parameters()
        assert total_params > 0


class TestLoRAMerger:
    """Test LoRA merging utility."""
    
    def test_merge_linear(self):
        """Test merging LoRA weights into linear layer."""
        base_layer = nn.Linear(10, 10)
        lora_layer = LoRALinear(base_layer, rank=4)
        
        merger = LoRAMerger(scaling=1.0)
        
        # Create merged layer
        merged_layer = merger.merge_linear(base_layer, lora_layer)
        
        assert merged_layer.weight.shape == base_layer.weight.shape
    
    def test_unmerge_linear(self):
        """Test unmerging LoRA weights."""
        base_layer = nn.Linear(10, 10)
        original_weight = base_layer.weight.data.clone()
        
        lora_layer = LoRALinear(base_layer, rank=4)
        merger = LoRAMerger(scaling=1.0)
        
        # Merge
        merged = merger.merge_linear(base_layer, lora_layer)
        
        # Unmerge (restore original)
        unmerged = merger.unmerge_linear(merged, lora_layer)
        
        # Check restoration
        assert torch.allclose(unmerged.weight, original_weight, atol=1e-5)
    
    def test_merge_preserves_output(self):
        """Test merged layer produces similar outputs."""
        base_layer = nn.Linear(10, 10)
        lora_layer = LoRALinear(base_layer, rank=4)
        
        x = torch.randn(2, 10)
        
        # Output with LoRA
        with torch.no_grad():
            lora_output = lora_layer(x)
        
        # Merge and evaluate
        merger = LoRAMerger(scaling=1.0)
        merged_layer = merger.merge_linear(base_layer, lora_layer)
        
        with torch.no_grad():
            merged_output = merged_layer(x)
        
        # Should be very similar
        assert torch.allclose(lora_output, merged_output, atol=1e-4)


class TestLoRALayersIntegration:
    """Integration tests for layer composition."""
    
    def test_adapter_with_multiple_layers(self):
        """Test adapter managing different layer types."""
        adapter = LoRAAdapter(rank=8)
        
        adapter.add_lora_linear("fc1", nn.Linear(100, 100))
        adapter.add_lora_linear("fc2", nn.Linear(100, 100))
        adapter.add_lora_conv2d("conv1", nn.Conv2d(3, 64, 3))
        
        assert len(adapter.lora_layers) == 3
        params = adapter.get_parameter_count()
        assert params > 0
    
    def test_forward_backward_flow(self):
        """Test forward and backward pass through LoRA layers."""
        base_layer = nn.Linear(10, 10)
        lora_layer = LoRALinear(base_layer, rank=4)
        
        x = torch.randn(2, 10, requires_grad=True)
        output = lora_layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients computed
        for name, param in lora_layer.named_parameters():
            if "lora_" in name:
                assert param.grad is not None
    
    def test_composer_with_adapters(self):
        """Test composing multiple adapters."""
        adapter1 = LoRAAdapter(rank=8)
        adapter1.add_lora_linear("layer", nn.Linear(100, 100))
        
        adapter2 = LoRAAdapter(rank=8)
        adapter2.add_lora_linear("layer", nn.Linear(100, 100))
        
        composer = LoRAComposer(
            num_adapters=2,
            composition_mode=LoRACompositionMode.parallel,
        )
        
        # Simulate outputs from different adapters
        adapters_outputs = []
        for param_set in [adapter1.get_parameters(), adapter2.get_parameters()]:
            adapters_outputs.append(torch.randn(2, 100))
        
        result = composer.compose_outputs(adapters_outputs)
        assert result.shape == (2, 100)
