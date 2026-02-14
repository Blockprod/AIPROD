"""LoRA (Low-Rank Adaptation) configuration and data structures.

Provides:
- LoRA configuration management
- Adapter specifications
- Training targets
- Composition strategies
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import math

logger = logging.getLogger(__name__)


class LoRAInitType(Enum):
    """Initialization methods for LoRA weights."""
    GAUSSIAN = "gaussian"
    KAIMING = "kaiming"
    ZEROS = "zeros"


class LoRATarget(Enum):
    """Which model components to adapt with LoRA."""
    ATTENTION_QKV = "attention_qkv"      # Query/Key/Value projections
    ATTENTION_OUT = "attention_out"      # Output projection
    ATTENTION_ALL = "attention_all"      # All attention projections
    MLP = "mlp"                          # Feed-forward MLPs
    CONV = "conv"                        # Convolutional layers
    LINEAR = "linear"                    # All linear layers
    TRANSFORMER = "transformer"          # Full transformer blocks
    ENCODER = "encoder"                  # Text encoder
    DECODER = "decoder"                  # VAE decoder


class LoRACompositionMode(Enum):
    """How to compose multiple LoRA adapters."""
    SEQUENTIAL = "sequential"    # Apply sequentially
    PARALLEL = "parallel"        # Apply in parallel (sum outputs)
    GATED = "gated"             # Gated mixture (learned weights)
    CONDITIONAL = "conditional"  # Condition on input/state


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning.
    
    Specifies rank, targets, training hyperparameters, etc.
    """

    # Basic LoRA parameters
    rank: int = 8                          # Low-rank dimension
    alpha: float = 1.0                     # Scaling factor for adapters
    dropout_rate: float = 0.0              # Dropout on LoRA input

    # Target configuration
    target_modules: List[LoRATarget] = field(default_factory=lambda: [LoRATarget.ATTENTION_QKV])
    target_layers: Optional[List[int]] = None  # Specific layers (None = all)
    include_bias: bool = False             # Include bias in LoRA
    
    # Initialization
    init_type: LoRAInitType = LoRAInitType.GAUSSIAN
    init_std: float = 0.02
    
    # Training configuration
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 100
    num_epochs: int = 10
    batch_size: int = 32
    
    # Data configuration
    data_path: Optional[str] = None
    validation_split: float = 0.1
    shuffle: bool = True
    
    # Composition
    composition_mode: LoRACompositionMode = LoRACompositionMode.SEQUENTIAL
    merge_on_save: bool = False            # Merge adapters into base model
    
    # Optimization
    use_lora_only: bool = False            # Only update LoRA parameters
    freeze_base_model: bool = True         # Freeze non-LoRA parameters
    gradient_checkpointing: bool = False   # Reduce memory
    
    def __repr__(self) -> str:
        return (
            f"LoRAConfig(rank={self.rank}, alpha={self.alpha}, "
            f"targets={self.target_modules}, lr={self.learning_rate})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        config_dict = asdict(self)
        # Convert enums to strings
        config_dict["target_modules"] = [t.value for t in self.target_modules]
        config_dict["init_type"] = self.init_type.value
        config_dict["composition_mode"] = self.composition_mode.value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoRAConfig":
        """Create from dictionary."""
        config_dict = config_dict.copy()
        # Convert string enums back
        if "target_modules" in config_dict:
            config_dict["target_modules"] = [
                LoRATarget(t) if isinstance(t, str) else t
                for t in config_dict["target_modules"]
            ]
        if "init_type" in config_dict and isinstance(config_dict["init_type"], str):
            config_dict["init_type"] = LoRAInitType(config_dict["init_type"])
        if "composition_mode" in config_dict and isinstance(config_dict["composition_mode"], str):
            config_dict["composition_mode"] = LoRACompositionMode(config_dict["composition_mode"])
        return cls(**config_dict)
    
    def compute_parameter_reduction(self, original_param_count: int) -> float:
        """
        Compute parameter reduction compared to full fine-tuning.
        
        Args:
            original_param_count: Number of parameters in original model
            
        Returns:
            Reduction factor (0-1, lower = more reduction)
        """
        # Rough estimate: LoRA adds rank * 2 params per target layer
        # Assuming ~100 layers, ~1000 linear layers per transformer
        num_target_layers = 1000 if not self.target_layers else len(self.target_layers)
        lora_params = num_target_layers * self.rank * 2
        
        return lora_params / original_param_count


@dataclass
class LoRAAdapter:
    """Individual LoRA adapter specification."""
    
    name: str                        # Adapter name
    config: LoRAConfig              # Configuration
    adapter_data: Optional[Dict[str, Any]] = None  # Learned weights
    
    def __repr__(self) -> str:
        return f"LoRAAdapter({self.name}, config={self.config})"


@dataclass
class LoRAMetrics:
    """Training metrics for LoRA fine-tuning."""
    
    step: int
    loss: float
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    gradient_norm: Optional[float] = None
    samples_per_sec: float = 0.0
    
    def __repr__(self) -> str:
        val_str = f", val_loss={self.val_loss:.4f}" if self.val_loss else ""
        return f"LoRAMetrics(step={self.step}, loss={self.loss:.4f}{val_str})"


@dataclass
class LoRACheckpoint:
    """Checkpoint for LoRA training."""
    
    step: int
    epoch: int
    adapter_state: Dict[str, Any]   # Adapter weights
    optimizer_state: Optional[Dict[str, Any]] = None
    metrics: Optional[LoRAMetrics] = None
    config: Optional[LoRAConfig] = None
    
    def __repr__(self) -> str:
        return f"LoRACheckpoint(step={self.step}, epoch={self.epoch})"


class LoRAStrategy:
    """Strategy for selecting LoRA configuration based on constraints."""
    
    @staticmethod
    def for_resource_constrained(
        model_size: int = 7_000_000_000,  # 7B parameters
    ) -> LoRAConfig:
        """
        LoRA config for resource-constrained environments.
        
        Args:
            model_size: Model parameter count
            
        Returns:
            Optimized LoRA config
        """
        return LoRAConfig(
            rank=4,
            alpha=1.0,
            target_modules=[LoRATarget.ATTENTION_QKV],
            learning_rate=5e-5,
            batch_size=8,
            gradient_checkpointing=True,
        )
    
    @staticmethod
    def for_high_quality(
        model_size: int = 7_000_000_000,
    ) -> LoRAConfig:
        """
        LoRA config for maximum quality adaptation.
        
        Args:
            model_size: Model parameter count
            
        Returns:
            High-quality LoRA config
        """
        # Higher rank + more targets = better quality but more params
        num_params_ewa = math.log(model_size / 1_000_000)  # Log scale
        rank = min(64, max(16, int(num_params_ewa)))
        
        return LoRAConfig(
            rank=rank,
            alpha=2.0,
            target_modules=[
                LoRATarget.ATTENTION_QKV,
                LoRATarget.ATTENTION_OUT,
                LoRATarget.MLP,
            ],
            learning_rate=1e-4,
            batch_size=16,
            gradient_checkpointing=False,
            freeze_base_model=True,
        )
    
    @staticmethod
    def for_quick_adaptation(
        model_size: int = 7_000_000_000,
    ) -> LoRAConfig:
        """
        LoRA config for fastest training (trade-off quality for speed).
        
        Args:
            model_size: Model parameter count
            
        Returns:
            Fast training LoRA config
        """
        return LoRAConfig(
            rank=8,
            alpha=1.0,
            target_modules=[LoRATarget.ATTENTION_QKV],
            learning_rate=1e-3,
            batch_size=32,
            num_epochs=3,
            gradient_checkpointing=True,
        )
    
    @staticmethod
    def for_multi_task(
        model_size: int = 7_000_000_000,
        num_tasks: int = 5,
    ) -> LoRAConfig:
        """
        LoRA config for multi-task learning with separate adapters.
        
        Args:
            model_size: Model parameter count
            num_tasks: Number of tasks/adapters
            
        Returns:
            Multi-task LoRA config
        """
        rank = max(4, 32 // max(1, num_tasks - 1))  # Scale rank inversely
        
        return LoRAConfig(
            rank=rank,
            alpha=1.5,
            target_modules=[LoRATarget.ATTENTION_ALL, LoRATarget.MLP],
            composition_mode=LoRACompositionMode.GATED,
            learning_rate=5e-4,
            batch_size=16,
            freeze_base_model=True,
        )


class LoRAPrecisionConfig:
    """Mixed precision settings for LoRA training."""
    
    @staticmethod
    def fp32() -> Dict[str, str]:
        """Full precision training."""
        return {"compute": "fp32", "backward": "fp32"}
    
    @staticmethod
    def fp16() -> Dict[str, str]:
        """Half precision training."""
        return {"compute": "fp16", "backward": "fp32"}  # Compute in fp16, grad in fp32
    
    @staticmethod
    def bf16() -> Dict[str, str]:
        """BF16 precision training."""
        return {"compute": "bf16", "backward": "bf16"}
    
    @staticmethod
    def automatic() -> Dict[str, str]:
        """Automatic mixed precision."""
        return {"compute": "auto", "backward": "auto"}
