"""
Distributed LoRA Module

Distributed fine-tuning infrastructure for user-custom models at scale.
Enables federated learning, model composition, and per-user model management.

Key Components:
- Distributed LoRA configuration and presets
- Federated learning with privacy preservation
- LoRA model merging and hierarchical inheritance
- Global model registry and discovery
- Adaptive LoRA selection for inference
- Per-user model management and deduplication
- Training loop integration with tensor parallelism
"""

# Configuration
from .distributed_lora_config import (
    LoRARank,
    LoRATarget,
    LoRAInitialization,
    DistributedLoRAConfig,
    UserLoRAPreset,
    LoRAModelMetadata,
    DatasetConfig,
    DistributedLoRAConfigManager,
)

# Federated Learning
from .federated_training import (
    FederatedAggregationMethod,
    DifferentialPrivacyLevel,
    ClientUpdate,
    ServerState,
    FederatedAggregator,
    DifferentialPrivacyEngine,
    FederatedTrainer,
    FederatedTrainingConfig,
)

# Model Composition
from .lora_merge_engine import (
    MergeStrategy,
    AdapterWeight,
    CompositionPlan,
    LoRAMergeEngine,
    ModelInheritance,
    AdapterComposer,
)

# Model Registry
from .lora_registry import (
    ModelStatus,
    AccessLevel,
    ModelVersion,
    LoRARegistry,
    ModelDiscovery,
    RegistryConfig,
)

# User Model Management
from .user_model_manager import (
    UserModelTier,
    UserModelQuota,
    UserModel,
    UserModelManager,
    ModelDeduplication,
    TenantModelManager,
)

# Distributed Training
from .distributed_lora_trainer import (
    TrainingMode,
    TrainingMetrics,
    TrainingState,
    DistributedLoRATrainer,
    FederatedLoRATrainer,
    LoRAInferenceOptimizer,
    LoRATrainingConfig,
)

__all__ = [
    # Configuration (8 exports)
    "LoRARank",
    "LoRATarget",
    "LoRAInitialization",
    "DistributedLoRAConfig",
    "UserLoRAPreset",
    "LoRAModelMetadata",
    "DatasetConfig",
    "DistributedLoRAConfigManager",
    
    # Federated Learning (8 exports)
    "FederatedAggregationMethod",
    "DifferentialPrivacyLevel",
    "ClientUpdate",
    "ServerState",
    "FederatedAggregator",
    "DifferentialPrivacyEngine",
    "FederatedTrainer",
    "FederatedTrainingConfig",
    
    # Model Composition (6 exports)
    "MergeStrategy",
    "AdapterWeight",
    "CompositionPlan",
    "LoRAMergeEngine",
    "ModelInheritance",
    "AdapterComposer",
    
    # Model Registry (6 exports)
    "ModelStatus",
    "AccessLevel",
    "ModelVersion",
    "LoRARegistry",
    "ModelDiscovery",
    "RegistryConfig",
    
    # User Management (6 exports)
    "UserModelTier",
    "UserModelQuota",
    "UserModel",
    "UserModelManager",
    "ModelDeduplication",
    "TenantModelManager",
    
    # Distributed Training (7 exports)
    "TrainingMode",
    "TrainingMetrics",
    "TrainingState",
    "DistributedLoRATrainer",
    "FederatedLoRATrainer",
    "LoRAInferenceOptimizer",
    "LoRATrainingConfig",
]
