"""
Model Sharding and Activation Checkpointing

Implements model weight sharding strategies and activation recomputation
for memory-efficient distributed training.
"""

from enum import Enum
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
import math


class ShardingDimension(Enum):
    """Dimensions for sharding tensor parameters"""
    ROWS = "rows"  # Shard output dimension (row-wise)
    COLS = "cols"  # Shard input dimension (column-wise)
    SEQUENCE = "sequence"  # Shard sequence dimension
    HEADS = "heads"  # Shard attention heads


@dataclass
class ShardedTensorSpec:
    """Specification for how to shard a tensor"""
    name: str
    shape: Tuple[int, ...]
    shard_dim: ShardingDimension
    num_shards: int
    local_shape: Tuple[int, ...] = None
    
    def __post_init__(self):
        """Compute local shape after sharding"""
        if self.local_shape is None:
            local_shape = list(self.shape)
            dim_map = {
                ShardingDimension.ROWS: 0,
                ShardingDimension.COLS: -1,
                ShardingDimension.SEQUENCE: 1,
                ShardingDimension.HEADS: 0
            }
            
            dim_idx = dim_map.get(self.shard_dim, 0)
            if 0 <= dim_idx < len(local_shape):
                local_shape[dim_idx] = local_shape[dim_idx] // self.num_shards
            
            self.local_shape = tuple(local_shape)


class ModelShardingStrategy(Enum):
    """Overall model sharding strategies"""
    SHARD_ALL_LAYERS = "shard_all_layers"
    SHARD_ATTENTION_ONLY = "shard_attention_only"
    SHARD_MLP_ONLY = "shard_mlp_only"
    SHARD_SELECTIVE = "shard_selective"  # Custom per layer


class ModelSharder:
    """Shards model parameters across devices"""
    
    def __init__(self, strategy: ModelShardingStrategy, tp_size: int):
        self.strategy = strategy
        self.tp_size = tp_size
        self.sharding_specs: Dict[str, ShardedTensorSpec] = {}
    
    def create_sharding_plan(self, model_config: Dict[str, Any]) -> Dict[str, ShardedTensorSpec]:
        """
        Create sharding plan for model based on architecture
        
        Args:
            model_config: Model architecture config (hidden_dim, num_layers, etc.)
        
        Returns:
            Mapping of parameter names to sharding specs
        """
        hidden_dim = model_config.get("hidden_dim", 4096)
        num_layers = model_config.get("num_layers", 32)
        attention_heads = model_config.get("attention_heads", 32)
        
        print(f"[ModelSharder] Creating sharding plan: {hidden_dim}d, {num_layers} layers, {attention_heads} heads, tp_size={self.tp_size}")
        
        specs = {}
        
        # Shard embedding layer
        if self.strategy in [ModelShardingStrategy.SHARD_ALL_LAYERS, ModelShardingStrategy.SHARD_SELECTIVE]:
            vocab_size = model_config.get("vocab_size", 50000)
            specs["embedding.weight"] = ShardedTensorSpec(
                name="embedding.weight",
                shape=(vocab_size, hidden_dim),
                shard_dim=ShardingDimension.COLS,
                num_shards=self.tp_size
            )
        
        # Shard each transformer layer
        for layer_idx in range(num_layers):
            prefix = f"layers.{layer_idx}"
            
            # Attention layers
            if self.strategy in [ModelShardingStrategy.SHARD_ALL_LAYERS, ModelShardingStrategy.SHARD_ATTENTION_ONLY]:
                # Q, K, V projections
                specs[f"{prefix}.attention.q_proj.weight"] = ShardedTensorSpec(
                    name=f"{prefix}.attention.q_proj.weight",
                    shape=(hidden_dim, hidden_dim),
                    shard_dim=ShardingDimension.ROWS,
                    num_shards=self.tp_size
                )
                specs[f"{prefix}.attention.o_proj.weight"] = ShardedTensorSpec(
                    name=f"{prefix}.attention.o_proj.weight",
                    shape=(hidden_dim, hidden_dim),
                    shard_dim=ShardingDimension.COLS,
                    num_shards=self.tp_size
                )
            
            # MLP layers
            if self.strategy in [ModelShardingStrategy.SHARD_ALL_LAYERS, ModelShardingStrategy.SHARD_MLP_ONLY]:
                mlp_hidden = model_config.get("mlp_hidden_dim", hidden_dim * 4)
                specs[f"{prefix}.mlp.fc1.weight"] = ShardedTensorSpec(
                    name=f"{prefix}.mlp.fc1.weight",
                    shape=(mlp_hidden, hidden_dim),
                    shard_dim=ShardingDimension.ROWS,
                    num_shards=self.tp_size
                )
                specs[f"{prefix}.mlp.fc2.weight"] = ShardedTensorSpec(
                    name=f"{prefix}.mlp.fc2.weight",
                    shape=(hidden_dim, mlp_hidden),
                    shard_dim=ShardingDimension.COLS,
                    num_shards=self.tp_size
                )
        
        self.sharding_specs = specs
        return specs
    
    def get_local_parameter_shape(self, param_name: str) -> Optional[Tuple[int, ...]]:
        """Get shape of parameter on local device after sharding"""
        spec = self.sharding_specs.get(param_name)
        return spec.local_shape if spec else None
    
    def get_sharding_info(self, param_name: str) -> Optional[Dict[str, Any]]:
        """Get sharding info for parameter"""
        spec = self.sharding_specs.get(param_name)
        if not spec:
            return None
        
        return {
            "name": spec.name,
            "global_shape": spec.shape,
            "local_shape": spec.local_shape,
            "shard_dim": spec.shard_dim.value,
            "num_shards": spec.num_shards
        }
    
    def compute_reduction_axis(self, param_name: str) -> Optional[int]:
        """
        Compute which dimension needs allreduce
        (for column-wise sharding, need to allreduce output)
        """
        spec = self.sharding_specs.get(param_name)
        if not spec or spec.shard_dim == ShardingDimension.ROWS:
            return None  # Row-wise sharding: no allreduce needed
        
        return 0  # Column-wise: reduce-scatter result


class ActivationCheckpointing:
    """Enables activation checkpointing for memory efficiency"""
    
    def __init__(self, num_checkpoints: int = 1):
        self.num_checkpoints = num_checkpoints
        self.checkpoint_list: List[str] = []
        self.max_memory_saved_mb = 0.0
    
    def select_checkpointing_layers(self, num_layers: int) -> List[int]:
        """
        Select which layers to checkpoint activations for
        
        Returns:
            List of layer indices to checkpoint
        """
        if num_layers <= 1:
            return []
        
        # Checkpoint sqrt(N) layers uniformly distributed
        num_checkpoints = max(1, int(math.sqrt(num_layers)))
        checkpoint_interval = max(1, num_layers // num_checkpoints)
        
        return list(range(checkpoint_interval - 1, num_layers, checkpoint_interval))
    
    def get_memory_savings(self, layer_memory_mb: float, num_layers: int) -> float:
        """
        Estimate memory savings from activation checkpointing
        
        Args:
            layer_memory_mb: Memory per layer activations
            num_layers: Total number of layers
        
        Returns:
            Estimated memory saved in MB
        """
        # Without checkpointing: store all N activation sets
        # With checkpointing: store sqrt(N) + recompute others
        without_checkpointing = layer_memory_mb * num_layers
        with_checkpointing = layer_memory_mb * math.sqrt(num_layers)
        
        return without_checkpointing - with_checkpointing
    
    def estimate_recompute_time(self, forward_time_ms: float, 
                               num_layers: int) -> float:
        """Estimate time overhead from recomputation"""
        num_checkpoints = max(1, int(math.sqrt(num_layers)))
        # Roughly need to recompute sqrt(N) layers
        return forward_time_ms * (num_checkpoints / num_layers)


class LayerWisePartitioning:
    """Partitions layers across devices for pipeline parallelism"""
    
    def __init__(self, num_devices: int, num_layers: int):
        self.num_devices = num_devices
        self.num_layers = num_layers
        self.layers_per_device = num_layers // num_devices
        self.partition_plan = self._create_partition()
    
    def _create_partition(self) -> Dict[int, Tuple[int, int]]:
        """Create partition plan: device -> (start_layer, end_layer)"""
        plan = {}
        for device_id in range(self.num_devices):
            start = device_id * self.layers_per_device
            end = start + self.layers_per_device if device_id < self.num_devices - 1 else self.num_layers
            plan[device_id] = (start, end)
        return plan
    
    def get_layers_for_device(self, device_id: int) -> Tuple[int, int]:
        """Get layer range for device"""
        return self.partition_plan[device_id]
    
    def get_device_for_layer(self, layer_id: int) -> int:
        """Get device that processes layer"""
        for device_id, (start, end) in self.partition_plan.items():
            if start <= layer_id < end:
                return device_id
        return -1
    
    def get_boundary_layers(self) -> List[int]:
        """Get layers at device boundaries"""
        boundaries = []
        for device_id in range(1, self.num_devices):
            boundaries.append(self.partition_plan[device_id][0])
        return boundaries


@dataclass
class SequenceParallelConfig:
    """Configuration for sequence dimension parallelism"""
    enabled: bool = False
    sequence_partition_size: int = 1
    scatter_output: bool = False  # Scatter output or replicate


class SequenceParallelism:
    """Shards sequence dimension for memory efficiency"""
    
    def __init__(self, config: SequenceParallelConfig):
        self.config = config
    
    def get_local_sequence_length(self, global_seq_len: int) -> int:
        """Compute local sequence length after sharding"""
        if not self.config.enabled:
            return global_seq_len
        return global_seq_len // self.config.sequence_partition_size
    
    def scatter_sequence(self, sequence: Any, batch_size: int) -> Tuple[Any, int]:
        """Scatter sequence across devices"""
        local_seq_len = self.get_local_sequence_length(sequence.shape[1] if hasattr(sequence, 'shape') else len(sequence))
        # Returns local sequence portion
        return sequence, local_seq_len
    
    def gather_sequence(self, local_sequences: List[Any]) -> Any:
        """Gather sequences from all devices"""
        # Concatenate along sequence dimension
        return local_sequences
