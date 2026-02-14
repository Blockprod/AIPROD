"""
Tensor Parallelism Sharding Strategies

Implements data parallelism, tensor parallelism, pipeline parallelism, and hybrid strategies
for distributed training of large models across multiple devices and nodes.
"""

from enum import Enum
from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math


class ShardingStrategy(Enum):
    """Enum for supported sharding strategies"""
    DATA_PARALLEL = "data_parallel"  # Replicate model, shard data
    TENSOR_PARALLEL = "tensor_parallel"  # Shard model weights, all data
    PIPELINE_PARALLEL = "pipeline_parallel"  # Split model layers across devices
    SEQUENCE_PARALLEL = "sequence_parallel"  # Shard by sequence dimension
    HYBRID = "hybrid"  # Combination of above strategies
    EXPERT_PARALLEL = "expert_parallel"  # For mixture-of-experts


class DataLayout(Enum):
    """Data layout strategies for tensor dimensions"""
    REPLICATE = "replicate"  # Broadcast to all devices
    SHARD_DIM0 = "shard_dim0"  # Shard along dimension 0
    SHARD_DIM1 = "shard_dim1"  # Shard along dimension 1
    SHARD_ROW = "shard_row"  # Row-wise sharding (for linear layers)
    SHARD_COL = "shard_col"  # Column-wise sharding (for linear layers)


@dataclass
class ShardingConfig:
    """Configuration for sharding strategy"""
    strategy: ShardingStrategy
    world_size: int  # Total number of participating devices
    tp_size: int  # Tensor parallelism degree
    pp_size: int  # Pipeline parallelism degree
    dp_size: int  # Data parallelism degree (computed as world_size / (tp_size * pp_size))
    sp_size: int = 1  # Sequence parallelism degree
    enable_gradient_accumulation: bool = True
    microbatch_size: int = 1
    activation_checkpointing: bool = True
    
    def __post_init__(self):
        """Validate sharding configuration"""
        if self.tp_size * self.pp_size * self.sp_size > self.world_size:
            raise ValueError(
                f"Invalid sharding: tp_size({self.tp_size}) * pp_size({self.pp_size}) * "
                f"sp_size({self.sp_size}) > world_size({self.world_size})"
            )
        # Compute data parallelism size
        if self.dp_size == 0:
            self.dp_size = self.world_size // (self.tp_size * self.pp_size)


@dataclass
class DeviceMesh:
    """Represents a multi-dimensional mesh of devices"""
    shape: Tuple[int, ...]  # Shape of mesh (e.g., (4, 8, 2) for 64 devices)
    axis_names: Tuple[str, ...] = field(default_factory=lambda: ("dp", "tp", "pp"))
    device_ids: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize mesh topology"""
        self.size = math.prod(self.shape)
        if not self.device_ids:
            self.device_ids = list(range(self.size))
    
    def get_rank_from_coords(self, coords: Tuple[int, ...]) -> int:
        """Convert multi-dimensional coordinates to linear rank"""
        rank = 0
        multiplier = 1
        for i in range(len(self.shape) - 1, -1, -1):
            rank += coords[i] * multiplier
            multiplier *= self.shape[i]
        return rank
    
    def get_coords_from_rank(self, rank: int) -> Tuple[int, ...]:
        """Convert linear rank to multi-dimensional coordinates"""
        coords = []
        for dim in reversed(self.shape):
            coords.append(rank % dim)
            rank //= dim
        return tuple(reversed(coords))


class TensorShardingPlanner(ABC):
    """Base class for tensor sharding planners"""
    
    @abstractmethod
    def plan_tensor_sharding(self, tensor_shape: Tuple[int, ...], 
                            sharding_spec: str) -> Dict[str, Any]:
        """Plan how to shard a tensor"""
        pass
    
    @abstractmethod
    def get_local_shape(self, global_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get local tensor shape after sharding"""
        pass


class DataParallelPlanner(TensorShardingPlanner):
    """Plans sharding for data parallelism"""
    
    def __init__(self, config: ShardingConfig):
        self.config = config
    
    def plan_tensor_sharding(self, tensor_shape: Tuple[int, ...], 
                            sharding_spec: str = "replicate") -> Dict[str, Any]:
        """All devices replicate model; data is sharded by batch dimension"""
        return {
            "strategy": "data_parallel",
            "tensor_sharding": "replicate",
            "data_sharding": f"shard_dim0_size{self.config.dp_size}",
            "local_batch_size": tensor_shape[0] // self.config.dp_size if tensor_shape else None,
            "requires_allreduce": True
        }
    
    def get_local_shape(self, global_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Local shape = global shape with batch dimension reduced"""
        if not global_shape:
            return global_shape
        local_shape = list(global_shape)
        local_shape[0] = local_shape[0] // self.config.dp_size
        return tuple(local_shape)


class TensorParallelPlanner(TensorShardingPlanner):
    """Plans sharding for tensor (model) parallelism"""
    
    def __init__(self, config: ShardingConfig):
        self.config = config
    
    def plan_tensor_sharding(self, tensor_shape: Tuple[int, ...], 
                            sharding_spec: str) -> Dict[str, Any]:
        """Shard model weights across devices, keep all data"""
        if sharding_spec == "row":
            # Row-wise: shard dimension 0 (output features)
            return {
                "strategy": "tensor_parallel",
                "sharding_type": "row_wise",
                "shard_dim": 0,
                "num_shards": self.config.tp_size,
                "local_shape_dim0": tensor_shape[0] // self.config.tp_size if tensor_shape else None,
                "requires_allgather": True
            }
        elif sharding_spec == "col":
            # Column-wise: shard dimension 1 (input features)
            return {
                "strategy": "tensor_parallel",
                "sharding_type": "col_wise",
                "shard_dim": 1,
                "num_shards": self.config.tp_size,
                "local_shape_dim1": tensor_shape[1] // self.config.tp_size if len(tensor_shape) > 1 else None,
                "requires_allreduce": True
            }
        else:
            raise ValueError(f"Unknown tensor parallelism spec: {sharding_spec}")
    
    def get_local_shape(self, global_shape: Tuple[int, ...], 
                       sharding_spec: str = "row") -> Tuple[int, ...]:
        """Reduce model dimension based on sharding"""
        if not global_shape:
            return global_shape
        local_shape = list(global_shape)
        shard_dim = 0 if sharding_spec == "row" else 1
        if shard_dim < len(local_shape):
            local_shape[shard_dim] = local_shape[shard_dim] // self.config.tp_size
        return tuple(local_shape)


class PipelineParallelPlanner(TensorShardingPlanner):
    """Plans sharding for pipeline parallelism"""
    
    def __init__(self, config: ShardingConfig):
        self.config = config
    
    def plan_tensor_sharding(self, tensor_shape: Tuple[int, ...], 
                            sharding_spec: str = "layer") -> Dict[str, Any]:
        """Assign model layers to different devices in pipeline"""
        return {
            "strategy": "pipeline_parallel",
            "num_pipeline_stages": self.config.pp_size,
            "layers_per_stage": -1,  # To be computed from model
            "activation_checkpointing": self.config.activation_checkpointing,
            "microbatch_size": self.config.microbatch_size,
            "requires_p2p_communication": True
        }
    
    def get_local_shape(self, global_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Shape doesn't change, but subset of layers processed locally"""
        return global_shape


class HybridParallelPlanner(TensorShardingPlanner):
    """Plans sharding for hybrid parallelism (combination of strategies)"""
    
    def __init__(self, config: ShardingConfig):
        self.config = config
        self.dp_planner = DataParallelPlanner(config)
        self.tp_planner = TensorParallelPlanner(config)
        self.pp_planner = PipelineParallelPlanner(config)
    
    def plan_tensor_sharding(self, tensor_shape: Tuple[int, ...], 
                            sharding_spec: str = "dp_tp") -> Dict[str, Any]:
        """Combine multiple sharding strategies"""
        if "dp" in sharding_spec and "tp" in sharding_spec:
            return {
                "strategy": "hybrid",
                "composition": "dp_tp",
                "data_parallel": self.dp_planner.plan_tensor_sharding(tensor_shape),
                "tensor_parallel": self.tp_planner.plan_tensor_sharding(tensor_shape, "row"),
                "total_parallelism": self.config.dp_size * self.config.tp_size
            }
        elif "tp" in sharding_spec and "pp" in sharding_spec:
            return {
                "strategy": "hybrid",
                "composition": "tp_pp",
                "tensor_parallel": self.tp_planner.plan_tensor_sharding(tensor_shape, "row"),
                "pipeline_parallel": self.pp_planner.plan_tensor_sharding(tensor_shape),
                "total_parallelism": self.config.tp_size * self.config.pp_size
            }
        else:
            raise ValueError(f"Unsupported hybrid spec: {sharding_spec}")
    
    def get_local_shape(self, global_shape: Tuple[int, ...], strategy: str = "dp_tp") -> Tuple[int, ...]:
        """Reduce dimensions based on combined strategies"""
        shape = list(global_shape) if global_shape else []
        if "dp" in strategy and shape:
            shape[0] = shape[0] // self.config.dp_size
        if "tp" in strategy and len(shape) > 1:
            shape[1] = shape[1] // self.config.tp_size
        return tuple(shape)


class ShardingPlanner:
    """High-level planner for selecting and managing sharding strategies"""
    
    def __init__(self, config: ShardingConfig):
        self.config = config
        self.planners = {
            ShardingStrategy.DATA_PARALLEL: DataParallelPlanner(config),
            ShardingStrategy.TENSOR_PARALLEL: TensorParallelPlanner(config),
            ShardingStrategy.PIPELINE_PARALLEL: PipelineParallelPlanner(config),
            ShardingStrategy.HYBRID: HybridParallelPlanner(config),
        }
    
    def plan_model_sharding(self, strategy: ShardingStrategy, 
                           tensor_specs: Dict[str, str]) -> Dict[str, Any]:
        """Plan sharding for entire model"""
        planner = self.planners.get(strategy)
        if not planner:
            raise ValueError(f"Unsupported strategy: {strategy}")
        
        tensor_plans = {}
        for tensor_name, spec in tensor_specs.items():
            tensor_plans[tensor_name] = planner.plan_tensor_sharding((1, 1), spec)
        
        return {
            "strategy": strategy.value,
            "world_size": self.config.world_size,
            "tp_size": self.config.tp_size,
            "pp_size": self.config.pp_size,
            "dp_size": self.config.dp_size,
            "tensor_plans": tensor_plans
        }
    
    def get_communication_pattern(self, strategy: ShardingStrategy) -> Dict[str, Any]:
        """Get communication pattern for strategy"""
        patterns = {
            ShardingStrategy.DATA_PARALLEL: {
                "collective_ops": ["allreduce"],
                "synchronization": "epoch",
                "bandwidth_intensive": True
            },
            ShardingStrategy.TENSOR_PARALLEL: {
                "collective_ops": ["allgather", "allreduce"],
                "synchronization": "batch",
                "bandwidth_intensive": True
            },
            ShardingStrategy.PIPELINE_PARALLEL: {
                "collective_ops": ["point_to_point"],
                "synchronization": "pipeline_stage",
                "bandwidth_intensive": False
            },
            ShardingStrategy.HYBRID: {
                "collective_ops": ["allreduce", "allgather", "point_to_point"],
                "synchronization": "mixed",
                "bandwidth_intensive": True
            }
        }
        return patterns.get(strategy, {})


@dataclass
class ShardingMetrics:
    """Metrics for sharding efficiency"""
    computation_efficiency: float  # Ratio of actual compute to peak compute
    communication_efficiency: float  # Ratio of compute to communication time
    memory_efficiency: float  # Ratio of useful memory to total memory
    throughput_tokens_per_sec: float  # Tokens per second throughput
    latency_per_batch_ms: float  # Latency in milliseconds
    
    @property
    def overall_efficiency(self) -> float:
        """Combined efficiency metric"""
        return (self.computation_efficiency + self.communication_efficiency + 
                self.memory_efficiency) / 3.0
