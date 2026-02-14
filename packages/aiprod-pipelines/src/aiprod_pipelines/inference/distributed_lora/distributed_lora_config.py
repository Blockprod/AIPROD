"""
Distributed LoRA Configuration

Manages configuration for distributed LoRA training with per-tenant,
per-user, and per-model customization.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime


class LoRARank(Enum):
    """LoRA adapter rank presets"""
    SMALL = (4, 8)  # (lora_r, lora_alpha)
    MEDIUM = (8, 16)
    LARGE = (16, 32)
    XLARGE = (32, 64)


class LoRATarget(Enum):
    """Model components to apply LoRA to"""
    ATTENTION_QKV = "attention_qkv"  # Q, K, V in attention
    ATTENTION_OUT = "attention_out"  # Output projection
    ATTENTION_FULL = "attention_full"  # All attention projections
    MLP = "mlp"  # Feed-forward network
    ALL = "all"  # All applicable layers


class LoRAInitialization(Enum):
    """LoRA weight initialization strategies"""
    GAUSSIAN = "gaussian"  # Normal distribution
    UNIFORM = "uniform"  # Uniform distribution
    LEXP = "lexp"  # Log-exponential
    ZERO = "zero"  # Zero initialization


@dataclass
class DistributedLoRAConfig:
    """Configuration for distributed LoRA training"""
    # Model structure
    rank: LoRARank = LoRARank.MEDIUM
    target_modules: LoRATarget = LoRATarget.ATTENTION_QKV
    initialization: LoRAInitialization = LoRAInitialization.GAUSSIAN
    init_std: float = 0.02
    
    # Distributed training
    enable_federated: bool = True
    federated_rounds: int = 10
    federated_local_epochs: int = 3
    federated_sample_fraction: float = 1.0  # Fraction of users per round
    
    # Multi-user settings
    enable_model_merging: bool = True
    enable_model_inheritance: bool = True
    max_models_per_tenant: int = 100
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 500
    max_steps: int = 10000
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    
    # Distributed settings
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_strategy: str = "steps"
    save_steps: int = 500
    resume_from_checkpoint: Optional[str] = None
    
    # Monitoring
    eval_steps: int = 500
    log_steps: int = 100
    
    def get_lora_r_alpha(self) -> tuple:
        """Get LoRA rank and alpha values"""
        return self.rank.value
    
    def get_parallel_size(self) -> int:
        """Get total parallelism degree"""
        return (self.data_parallel_size * 
                self.tensor_parallel_size * 
                self.pipeline_parallel_size)


@dataclass
class UserLoRAPreset:
    """Preset configuration for user LoRA models"""
    preset_id: str
    name: str
    description: str
    rank: LoRARank
    target_modules: LoRATarget
    learning_rate: float
    training_steps: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: Set[str] = field(default_factory=set)
    
    def to_config(self) -> DistributedLoRAConfig:
        """Convert preset to config"""
        return DistributedLoRAConfig(
            rank=self.rank,
            target_modules=self.target_modules,
            learning_rate=self.learning_rate,
            max_steps=self.training_steps
        )


@dataclass
class LoRAModelMetadata:
    """Metadata for a LoRA model"""
    model_id: str
    tenant_id: str
    user_id: Optional[str]
    base_model: str
    preset_id: Optional[str]
    rank: int
    alpha: int
    num_parameters: int
    training_steps: int
    training_loss: float
    eval_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    parent_model_id: Optional[str] = None  # For inherited models
    tags: Set[str] = field(default_factory=set)
    is_shared: bool = False
    sharing_permissions: List[str] = field(default_factory=list)
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "model_id": self.model_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "base_model": self.base_model,
            "rank": self.rank,
            "alpha": self.alpha,
            "num_parameters": self.num_parameters,
            "training_steps": self.training_steps,
            "training_loss": self.training_loss,
            "eval_metrics": self.eval_metrics,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "parent_model_id": self.parent_model_id,
            "tags": list(self.tags),
            "is_shared": self.is_shared,
            "version": self.version
        }


@dataclass
class DatasetConfig:
    """Configuration for LoRA training dataset"""
    dataset_id: str
    name: str
    split: str = "train"
    num_samples: int = 0
    example_inputs: List[str] = field(default_factory=list)
    preprocessing_steps: List[str] = field(default_factory=list)
    tokenizer_name: str = "default"
    max_seq_length: int = 2048
    sampling_strategy: str = "sequential"  # sequential, random, stratified
    
    def get_dataloader_config(self) -> Dict[str, Any]:
        """Get dataloader configuration"""
        return {
            "dataset_id": self.dataset_id,
            "num_samples": self.num_samples,
            "max_seq_length": self.max_seq_length,
            "sampling_strategy": self.sampling_strategy,
            "tokenizer": self.tokenizer_name
        }


class DistributedLoRAConfigManager:
    """Manages distributed LoRA configurations"""
    
    def __init__(self):
        self.configs: Dict[str, DistributedLoRAConfig] = {}
        self.presets: Dict[str, UserLoRAPreset] = {}
        self._initialize_default_presets()
    
    def _initialize_default_presets(self):
        """Initialize default presets"""
        self.presets["quick"] = UserLoRAPreset(
            preset_id="quick",
            name="Quick Training",
            description="Fast LoRA training (small rank)",
            rank=LoRARank.SMALL,
            target_modules=LoRATarget.ATTENTION_QKV,
            learning_rate=1e-4,
            training_steps=1000
        )
        
        self.presets["balanced"] = UserLoRAPreset(
            preset_id="balanced",
            name="Balanced",
            description="Medium rank with good quality/speed",
            rank=LoRARank.MEDIUM,
            target_modules=LoRATarget.ATTENTION_FULL,
            learning_rate=5e-5,
            training_steps=5000,
            tags={"recommended"}
        )
        
        self.presets["quality"] = UserLoRAPreset(
            preset_id="quality",
            name="High Quality",
            description="Large rank for best results",
            rank=LoRARank.LARGE,
            target_modules=LoRATarget.ALL,
            learning_rate=1e-5,
            training_steps=10000
        )
    
    def create_config_from_preset(self, preset_id: str) -> DistributedLoRAConfig:
        """Create config from preset"""
        if preset_id not in self.presets:
            raise ValueError(f"Unknown preset: {preset_id}")
        
        preset = self.presets[preset_id]
        return preset.to_config()
    
    def register_custom_preset(self, preset: UserLoRAPreset):
        """Register custom preset"""
        self.presets[preset.preset_id] = preset
    
    def get_preset(self, preset_id: str) -> Optional[UserLoRAPreset]:
        """Get preset by ID"""
        return self.presets.get(preset_id)
    
    def list_presets(self) -> List[UserLoRAPreset]:
        """List all presets"""
        return list(self.presets.values())
    
    def save_config(self, config_id: str, config: DistributedLoRAConfig):
        """Save configuration"""
        self.configs[config_id] = config
    
    def load_config(self, config_id: str) -> Optional[DistributedLoRAConfig]:
        """Load configuration"""
        return self.configs.get(config_id)
