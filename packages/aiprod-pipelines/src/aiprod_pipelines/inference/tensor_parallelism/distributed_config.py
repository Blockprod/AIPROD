"""
Distributed Configuration and Environment Setup

Manages distributed training environment configuration, CUDA/device setup,
and rank initialization across multiple nodes.
"""

from enum import Enum
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
import os


class DistributedBackend(Enum):
    """Distributed training backends"""
    NCCL = "nccl"  # GPU collective communication
    GLOO = "gloo"  # CPU/GPU generic communication
    MPI = "mpi"  # Message passing interface
    PYTORCH_DEEPSPEED = "pytorch_deepspeed"


@dataclass
class DistributedConfig:
    """Configuration for distributed training environment"""
    backend: DistributedBackend = DistributedBackend.NCCL
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    node_rank: int = 0
    num_nodes: int = 1
    master_addr: str = "localhost"
    master_port: int = 29500
    num_gpus_per_node: int = 8
    debug: bool = False
    enable_cuda_graphs: bool = True
    enable_memory_stats: bool = True
    
    @property
    def is_distributed(self) -> bool:
        """Whether this is a distributed setup"""
        return self.world_size > 1
    
    @property
    def is_main_process(self) -> bool:
        """Whether this is the main/rank 0 process"""
        return self.rank == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "backend": self.backend.value,
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "node_rank": self.node_rank,
            "num_nodes": self.num_nodes,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "num_gpus_per_node": self.num_gpus_per_node,
        }


@dataclass
class GPUMemoryConfig:
    """GPU memory management configuration"""
    max_memory_allocation_percent: float = 0.9  # Use 90% of available GPU memory
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    mixed_precision_dtype: str = "float16"  # float16, bfloat16, float32
    enable_activation_remat: bool = True  # Activation recomputation
    compiled_mode: bool = True  # Enable torch.compile
    
    def get_memory_limit_bytes(self, device_id: int) -> int:
        """Calculate memory limit for device"""
        # In real implementation: query actual GPU memory
        assumed_40gb = 40 * 1024 * 1024 * 1024  # 40GB RTX A100
        return int(assumed_40gb * self.max_memory_allocation_percent)


@dataclass
class CPUAffinity:
    """CPU core affinity configuration"""
    enabled: bool = True
    cores_per_rank: int = 8
    
    def get_cpu_core_range(self, rank: int) -> tuple:
        """Get CPU core range for rank"""
        if not self.enabled:
            return None
        start = rank * self.cores_per_rank
        end = start + self.cores_per_rank
        return (start, end)


@dataclass
class DistributedEnvironment:
    """Represents distributed training environment"""
    config: DistributedConfig
    gpu_memory_config: GPUMemoryConfig = field(default_factory=GPUMemoryConfig)
    cpu_affinity: CPUAffinity = field(default_factory=CPUAffinity)
    local_device_id: int = 0
    global_device_ids: Dict[int, int] = field(default_factory=dict)  # rank -> device_id
    
    @classmethod
    def from_env_vars(cls) -> "DistributedEnvironment":
        """Initialize from environment variables (set by launcher)"""
        config = DistributedConfig(
            backend=DistributedBackend(os.environ.get("DISTRIBUTED_BACKEND", "nccl")),
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            rank=int(os.environ.get("RANK", "0")),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            node_rank=int(os.environ.get("NODE_RANK", "0")),
            num_nodes=int(os.environ.get("NUM_NODES", "1")),
            master_addr=os.environ.get("MASTER_ADDR", "localhost"),
            master_port=int(os.environ.get("MASTER_PORT", "29500")),
        )
        return cls(config=config, local_device_id=config.local_rank)
    
    def set_device(self):
        """Set current device (GPU or CPU)"""
        # In real implementation: set CUDA_VISIBLE_DEVICES
        pass
    
    def get_device_string(self) -> str:
        """Get device string for PyTorch"""
        device_id = self.config.local_rank
        if self.config.backend in [DistributedBackend.NCCL]:
            return f"cuda:{device_id}"
        return "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "config": self.config.to_dict(),
            "local_device_id": self.local_device_id,
            "device_string": self.get_device_string(),
        }


class DistributedInitializer:
    """Handles distributed training initialization"""
    
    def __init__(self, env: DistributedEnvironment):
        self.env = env
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize distributed training"""
        if not self.env.config.is_distributed:
            self.initialized = True
            return True
        
        # In real implementation: call init_process_group with backend
        # torch.distributed.init_process_group(
        #     backend=self.env.config.backend.value,
        #     init_method="env://",
        #     world_size=self.env.config.world_size,
        #     rank=self.env.config.rank,
        # )
        
        self.initialized = True
        return True
    
    def finalize(self):
        """Cleanup distributed setup"""
        if self.initialized:
            # In real implementation: destroy_process_group()
            self.initialized = False
    
    def get_rank(self) -> int:
        """Get global rank"""
        return self.env.config.rank
    
    def get_world_size(self) -> int:
        """Get world size"""
        return self.env.config.world_size
    
    def get_local_rank(self) -> int:
        """Get local rank (within node)"""
        return self.env.config.local_rank
    
    def is_main_process(self) -> bool:
        """Check if main process"""
        return self.env.config.is_main_process
    
    def barrier(self):
        """Synchronization barrier"""
        # In real: torch.distributed.barrier()
        pass


class ProcessGroupManager:
    """Manages process group topology for hierarchical communication"""
    
    def __init__(self, initializer: DistributedInitializer):
        self.initializer = initializer
        self.process_groups = {}
    
    def create_group(self, group_name: str, ranks: List[int]) -> str:
        """Create communication group for subset of ranks"""
        group_id = f"group_{group_name}"
        self.process_groups[group_id] = {
            "name": group_name,
            "ranks": ranks,
            "size": len(ranks),
            "group_handle": None  # Would be actual PyTorch group
        }
        return group_id
    
    def create_hierarchical_groups(self, tensor_parallel_size: int, 
                                   pipeline_parallel_size: int) -> Dict[str, str]:
        """Create hierarchical process groups for tensor + pipeline parallelism"""
        world_size = self.initializer.get_world_size()
        rank = self.initializer.get_rank()
        
        # Create tensor parallel groups (same TP group across different pp/dp ranks)
        tp_groups = {}
        for pp_idx in range(pipeline_parallel_size):
            for dp_idx in range(world_size // (tensor_parallel_size * pipeline_parallel_size)):
                group_ranks = [
                    dp_idx * tensor_parallel_size * pipeline_parallel_size +
                    pp_idx * tensor_parallel_size + tp_idx
                    for tp_idx in range(tensor_parallel_size)
                ]
                group_id = self.create_group(
                    f"tp_group_pp{pp_idx}_dp{dp_idx}",
                    group_ranks
                )
                tp_groups[f"tp_{pp_idx}_{dp_idx}"] = group_id
        
        # Create pipeline parallel groups
        pp_groups = {}
        for tp_idx in range(tensor_parallel_size):
            group_ranks = [
                i * tensor_parallel_size + tp_idx
                for i in range(world_size // tensor_parallel_size)
            ]
            group_id = self.create_group(f"pp_group_tp{tp_idx}", group_ranks)
            pp_groups[f"pp_{tp_idx}"] = group_id
        
        return {
            "tensor_parallel": tp_groups,
            "pipeline_parallel": pp_groups
        }
    
    def get_group_info(self, group_id: str) -> Dict[str, Any]:
        """Get information about communication group"""
        return self.process_groups.get(group_id, {})


@dataclass
class DistributedTrainingState:
    """Global state for distributed training"""
    step: int = 0
    epoch: int = 0
    total_samples_processed: int = 0
    total_time_seconds: float = 0.0
    is_checkpoint_step: bool = False
    
    def increment_step(self):
        """Increment training step"""
        self.step += 1
    
    def increment_epoch(self):
        """Increment epoch"""
        self.epoch += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state"""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "total_samples_processed": self.total_samples_processed,
            "total_time_seconds": self.total_time_seconds,
        }


class DistributedStateManager:
    """Manages synchronized state across all ranks"""
    
    def __init__(self, initializer: DistributedInitializer):
        self.initializer = initializer
        self.state = DistributedTrainingState()
    
    def get_state(self) -> DistributedTrainingState:
        """Get current training state"""
        return self.state
    
    def broadcast_state(self, state: Optional[DistributedTrainingState] = None):
        """Broadcast state from rank 0 to all ranks"""
        if state is None:
            state = self.state
        
        # In real: torch.distributed.broadcast_object_list
        # All ranks get same state from rank 0
        if self.initializer.is_main_process():
            self.state = state
    
    def synchronize_step(self):
        """Ensure all ranks are synchronized at same step"""
        self.initializer.barrier()
    
    def log_step(self, step_info: Dict[str, Any]):
        """Log information about training step"""
        if self.initializer.is_main_process():
            self.state.step += 1
            self.state.total_samples_processed += step_info.get("num_samples", 0)
