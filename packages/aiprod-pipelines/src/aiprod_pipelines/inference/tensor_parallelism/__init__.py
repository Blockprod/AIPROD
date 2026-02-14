"""
Tensor Parallelism Module

Core infrastructure for distributed training with tensor parallelism support.
Enables scaling large models across multiple GPUs, TPUs, and nodes.

Key Components:
- Sharding strategies (data, tensor, pipeline, hybrid parallelism)
- Distributed communication (allreduce, allgather, broadcast, etc.)
- Checkpoint management for distributed training
- Load balancing across devices
- Gradient accumulation and synchronization
- Model sharding and activation checkpointing
"""

# Sharding Strategies
from .sharding_strategies import (
    ShardingStrategy,
    DataLayout,
    ShardingConfig,
    DeviceMesh,
    TensorShardingPlanner,
    DataParallelPlanner,
    TensorParallelPlanner,
    PipelineParallelPlanner,
    HybridParallelPlanner,
    ShardingPlanner,
    ShardingMetrics,
)

# Communication Primitives
from .communication import (
    CollectiveOp,
    ReduceOp,
    CommunicationBackend,
    CommunicationConfig,
    CommunicationMetrics,
    CollectiveOperation,
    AllReduceOperation,
    AllGatherOperation,
    BroadcastOperation,
    ReduceScatterOperation,
    CommunicationManager,
    OverlappedCommunication,
)

# Distributed Configuration
from .distributed_config import (
    DistributedBackend,
    DistributedConfig,
    GPUMemoryConfig,
    CPUAffinity,
    DistributedEnvironment,
    DistributedInitializer,
    ProcessGroupManager,
    DistributedTrainingState,
    DistributedStateManager,
)

# Checkpoint Management 
from .checkpoint_manager import (
    CheckpointFormat,
    CheckpointMetadata,
    OptimizationState,
    CheckpointManager,
    IncrementalCheckpointing,
    ZeroCheckpointing,
    CheckpointRecoveryStrategy,
)

# Load Balancing
from .load_balancer import (
    LoadBalancingStrategy,
    DeviceWorkload,
    LoadMetrics,
    LoadBalancer,
    AdaptiveLoadBalancer,
    LoadBalancingConfig,
)

# Gradient Accumulation
from .gradient_accumulation import (
    GradientAccumulationMode,
    GradientAccumulationConfig,
    GradientBuffer,
    GradientAccumulator,
    DistributedGradientSync,
    GradientCheckpointing,
    GradientCompressionConfig,
    GradientCompression,
)

# Model Sharding
from .model_sharding import (
    ShardingDimension,
    ShardedTensorSpec,
    ModelShardingStrategy,
    ModelSharder,
    ActivationCheckpointing,
    LayerWisePartitioning,
    SequenceParallelConfig,
    SequenceParallelism,
)

__all__ = [
    # Sharding Strategies (15 exports)
    "ShardingStrategy",
    "DataLayout",
    "ShardingConfig",
    "DeviceMesh",
    "TensorShardingPlanner",
    "DataParallelPlanner",
    "TensorParallelPlanner",
    "PipelineParallelPlanner",
    "HybridParallelPlanner",
    "ShardingPlanner",
    "ShardingMetrics",
    
    # Communication (17 exports)
    "CollectiveOp",
    "ReduceOp",
    "CommunicationBackend",
    "CommunicationConfig",
    "CommunicationMetrics",
    "CollectiveOperation",
    "AllReduceOperation",
    "AllGatherOperation",
    "BroadcastOperation",
    "ReduceScatterOperation",
    "CommunicationManager",
    "OverlappedCommunication",
    
    # Distributed Config (10 exports)
    "DistributedBackend",
    "DistributedConfig",
    "GPUMemoryConfig",
    "CPUAffinity",
    "DistributedEnvironment",
    "DistributedInitializer",
    "ProcessGroupManager",
    "DistributedTrainingState",
    "DistributedStateManager",
    
    # Checkpoint Management (7 exports)
    "CheckpointFormat",
    "CheckpointMetadata",
    "OptimizationState",
    "CheckpointManager",
    "IncrementalCheckpointing",
    "ZeroCheckpointing",
    "CheckpointRecoveryStrategy",
    
    # Load Balancing (6 exports)
    "LoadBalancingStrategy",
    "DeviceWorkload",
    "LoadMetrics",
    "LoadBalancer",
    "AdaptiveLoadBalancer",
    "LoadBalancingConfig",
    
    # Gradient Accumulation (8 exports)
    "GradientAccumulationMode",
    "GradientAccumulationConfig",
    "GradientBuffer",
    "GradientAccumulator",
    "DistributedGradientSync",
    "GradientCheckpointing",
    "GradientCompressionConfig",
    "GradientCompression",
    
    # Model Sharding (8 exports)
    "ShardingDimension",
    "ShardedTensorSpec",
    "ModelShardingStrategy",
    "ModelSharder",
    "ActivationCheckpointing",
    "LayerWisePartitioning",
    "SequenceParallelConfig",
    "SequenceParallelism",
]
