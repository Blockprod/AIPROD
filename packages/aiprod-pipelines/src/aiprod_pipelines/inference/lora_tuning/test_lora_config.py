"""Test LoRA configuration system."""

import pytest
from aiprod_pipelines.inference.lora_tuning.lora_config import (
    LoRAInitType,
    LoRATarget,
    LoRACompositionMode,
    LoRAConfig,
    LoRAAdapter,
    LoRAMetrics,
    LoRACheckpoint,
    LoRAStrategy,
    LoRAPrecisionConfig,
)


class TestLoRAInitType:
    """Test LoRA initialization strategies."""
    
    def test_init_types_exist(self):
        """Test all init types are defined."""
        assert LoRAInitType.gaussian is not None
        assert LoRAInitType.kaiming is not None
        assert LoRAInitType.zeros is not None


class TestLoRATarget:
    """Test LoRA target selection."""
    
    def test_attention_targets(self):
        """Test attention layer targets."""
        assert LoRATarget.attention_qkv is not None
        assert LoRATarget.attention_out is not None
    
    def test_general_targets(self):
        """Test general layer targets."""
        assert LoRATarget.linear is not None
        assert LoRATarget.conv is not None
    
    def test_model_targets(self):
        """Test model-level targets."""
        assert LoRATarget.transformer is not None
        assert LoRATarget.encoder is not None
        assert LoRATarget.decoder is not None


class TestLoRACompositionMode:
    """Test adapter composition modes."""
    
    def test_composition_modes(self):
        """Test all composition modes."""
        assert LoRACompositionMode.sequential is not None
        assert LoRACompositionMode.parallel is not None
        assert LoRACompositionMode.gated is not None
        assert LoRACompositionMode.conditional is not None


class TestLoRAConfig:
    """Test LoRA configuration creation."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = LoRAConfig()
        assert config.rank == 8
        assert config.alpha == 1.0
        assert config.dropout_rate == 0.1
        assert config.learning_rate == 1e-4
        assert config.freeze_base_model is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LoRAConfig(
            rank=16,
            alpha=2.0,
            learning_rate=1e-3,
        )
        assert config.rank == 16
        assert config.alpha == 2.0
        assert config.learning_rate == 1e-3
    
    def test_targets_configuration(self):
        """Test target modules configuration."""
        config = LoRAConfig(
            target_modules=[
                LoRATarget.attention_qkv,
                LoRATarget.linear,
            ]
        )
        assert len(config.target_modules) == 2
        assert LoRATarget.attention_qkv in config.target_modules
    
    def test_composition_mode_config(self):
        """Test composition mode configuration."""
        config = LoRAConfig(
            composition_mode=LoRACompositionMode.gated
        )
        assert config.composition_mode == LoRACompositionMode.gated
    
    def test_parameter_reduction_ratio(self):
        """Test parameter reduction computation."""
        # Base model with 1M params, LoRA reduces to 10K
        reduction = LoRAConfig(rank=8).compute_parameter_reduction(
            total_params=1000000,
            lora_params=10000,
        )
        assert reduction > 0.99  # >99% reduction


class TestLoRAAdapter:
    """Test LoRA adapter configuration."""
    
    def test_adapter_creation(self):
        """Test adapter creation."""
        config = LoRAConfig(rank=8)
        adapter = LoRAAdapter(
            name="test_adapter",
            config=config,
            weights={},
        )
        assert adapter.name == "test_adapter"
        assert adapter.config.rank == 8
    
    def test_adapter_metadata(self):
        """Test adapter metadata."""
        adapter = LoRAAdapter(
            name="adapter1",
            config=LoRAConfig(),
            weights={"layer1": None},
            metadata={"tags": ["test"]},
        )
        assert adapter.metadata["tags"] == ["test"]


class TestLoRAMetrics:
    """Test training metrics."""
    
    def test_metrics_creation(self):
        """Test metrics object."""
        metrics = LoRAMetrics(
            step=100,
            epoch=1,
            loss=0.5,
            val_loss=0.6,
            learning_rate=1e-4,
        )
        assert metrics.step == 100
        assert metrics.loss == 0.5
        assert metrics.val_loss == 0.6
    
    def test_metrics_gradient(self):
        """Test gradient norm tracking."""
        metrics = LoRAMetrics(
            step=1,
            loss=0.5,
            gradient_norm=0.01,
        )
        assert metrics.gradient_norm == 0.01
    
    def test_metrics_throughput(self):
        """Test throughput tracking."""
        metrics = LoRAMetrics(
            step=1,
            loss=0.5,
            samples_per_second=100,
        )
        assert metrics.samples_per_second == 100


class TestLoRACheckpoint:
    """Test checkpoint system."""
    
    def test_checkpoint_creation(self):
        """Test checkpoint creation."""
        config = LoRAConfig()
        checkpoint = LoRACheckpoint(
            step=100,
            epoch=2,
            adapter_state={"lora_A": None},
            optimizer_state=None,
            config=config,
        )
        assert checkpoint.step == 100
        assert checkpoint.epoch == 2
    
    def test_checkpoint_metrics(self):
        """Test checkpoint metrics storage."""
        checkpoint = LoRACheckpoint(
            step=50,
            epoch=1,
            adapter_state={},
            metrics_history=[
                LoRAMetrics(step=0, loss=1.0),
                LoRAMetrics(step=10, loss=0.8),
            ],
        )
        assert len(checkpoint.metrics_history) == 2
        assert checkpoint.metrics_history[0].loss == 1.0


class TestLoRAStrategy:
    """Test pre-built LoRA strategies."""
    
    def test_resource_constrained(self):
        """Test resource-constrained strategy."""
        config = LoRAStrategy.for_resource_constrained()
        assert config.rank <= 4
        assert len(config.target_modules) <= 2
    
    def test_high_quality(self):
        """Test high-quality strategy."""
        config = LoRAStrategy.for_high_quality()
        assert config.rank >= 16
        assert len(config.target_modules) >= 3
    
    def test_quick_adaptation(self):
        """Test quick adaptation strategy."""
        config = LoRAStrategy.for_quick_adaptation()
        assert config.learning_rate >= 1e-3
        assert config.num_epochs <= 5
    
    def test_multi_task_strategy(self):
        """Test multi-task strategy."""
        config = LoRAStrategy.for_multi_task(num_tasks=4)
        assert config.rank > 8  # Scales with tasks
        assert config.composition_mode == LoRACompositionMode.gated
    
    def test_strategy_parameter_counts(self):
        """Test parameter counts across strategies."""
        low = LoRAStrategy.for_resource_constrained()
        high = LoRAStrategy.for_high_quality()
        
        assert low.rank < high.rank
        assert low.compute_parameter_reduction(1000000, 5000) > \
               high.compute_parameter_reduction(1000000, 50000)


class TestLoRAPrecisionConfig:
    """Test precision configuration."""
    
    def test_fp32_config(self):
        """Test FP32 precision."""
        config = LoRAPrecisionConfig(precision="fp32")
        assert config.precision == "fp32"
        assert config.use_mixed_precision is False
    
    def test_fp16_config(self):
        """Test FP16 precision."""
        config = LoRAPrecisionConfig(precision="fp16")
        assert config.precision == "fp16"
        assert config.use_mixed_precision is True
    
    def test_bf16_config(self):
        """Test BF16 precision."""
        config = LoRAPrecisionConfig(precision="bf16")
        assert config.precision == "bf16"
        assert config.use_mixed_precision is True
    
    def test_automatic_precision(self):
        """Test automatic precision selection."""
        config = LoRAPrecisionConfig(precision="auto")
        assert config.precision == "auto"
        assert config.use_mixed_precision is True
    
    def test_grad_scale(self):
        """Test gradient scaling."""
        config = LoRAPrecisionConfig(
            precision="fp16",
            grad_scale=1024.0,
        )
        assert config.grad_scale == 1024.0
    
    def test_loss_scale_window(self):
        """Test dynamic loss scaling window."""
        config = LoRAPrecisionConfig(
            precision="fp16",
            dynamic_loss_scale=True,
            loss_scale_window=100,
        )
        assert config.dynamic_loss_scale is True
        assert config.loss_scale_window == 100


class TestLoRAConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_strategy_selector(self):
        """Test selecting appropriate strategy."""
        # Low VRAM: use resource constrained
        low_vram_config = LoRAStrategy.for_resource_constrained()
        assert low_vram_config.rank <= 4
        
        # High quality: use full strategy
        high_quality = LoRAStrategy.for_high_quality()
        assert high_quality.rank >= 16
        
        # Quick: use fast strategy
        fast = LoRAStrategy.for_quick_adaptation()
        assert fast.learning_rate > 1e-4
    
    def test_multi_adapter_config(self):
        """Test configuration for multiple adapters."""
        config = LoRAConfig(
            rank=8,
            composition_mode=LoRACompositionMode.gated,
            num_adapters=3,
        )
        assert config.composition_mode == LoRACompositionMode.gated
    
    def test_config_with_precision(self):
        """Test configuration with precision settings."""
        config = LoRAConfig(
            rank=16,
            learning_rate=1e-3,
        )
        precision = LoRAPrecisionConfig(
            precision="fp16",
            dynamic_loss_scale=True,
        )
        
        assert config.rank == 16
        assert precision.use_mixed_precision is True
    
    def test_checkpoint_save_load_config(self):
        """Test configuration persistence via checkpoint."""
        original_config = LoRAConfig(
            rank=16,
            alpha=2.0,
            learning_rate=5e-4,
        )
        
        checkpoint = LoRACheckpoint(
            step=100,
            epoch=5,
            adapter_state={},
            config=original_config,
        )
        
        # Simulate load
        loaded_config = checkpoint.config
        assert loaded_config.rank == original_config.rank
        assert loaded_config.alpha == original_config.alpha
        assert loaded_config.learning_rate == original_config.learning_rate
