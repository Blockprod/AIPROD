"""
Distributed Gradient Accumulation and Synchronization

Implements gradient accumulation across micro-batches with distributed
synchronization patterns for different parallelism strategies.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time


class GradientAccumulationMode(Enum):
    """Gradient accumulation synchronization modes"""
    IMMEDIATE = "immediate"  # Sync after every batch
    DELAYED = "delayed"  # Sync after N micro-batches
    LAZY = "lazy"  # Lazy synchronization with optional sync
    OVERLAPPED = "overlapped"  # Overlap gradient compute with communication


@dataclass
class GradientAccumulationConfig:
    """Configuration for gradient accumulation"""
    mode: GradientAccumulationMode = GradientAccumulationMode.DELAYED
    num_accumulation_steps: int = 4  # Accumulate over N micro-batches
    sync_freq: int = 1  # Sync after every sync_freq accumulated steps
    checkpointing_enabled: bool = True
    dtype: str = "float32"
    
    @property
    def effective_batch_size_multiplier(self) -> int:
        """Effective batch size multiplier"""
        return self.num_accumulation_steps


@dataclass
class GradientBuffer:
    """Buffers gradients for accumulation"""
    gradients: Dict[str, List[Any]] = field(default_factory=dict)
    accumulation_step: int = 0
    synchronized: bool = False
    
    def add_gradient(self, param_name: str, gradient: Any):
        """Add gradient to buffer"""
        if param_name not in self.gradients:
            self.gradients[param_name] = []
        self.gradients[param_name].append(gradient)
        self.accumulation_step += 1
    
    def get_accumulated_gradient(self, param_name: str) -> Optional[Any]:
        """Get accumulated gradient for parameter"""
        if param_name not in self.gradients or not self.gradients[param_name]:
            return None
        
        grads = self.gradients[param_name]
        # Simple averaging (in real implementation: fused kernel)
        return sum(grads) / len(grads) if len(grads) > 0 else None
    
    def clear(self):
        """Clear gradient buffer"""
        self.gradients.clear()
        self.accumulation_step = 0
        self.synchronized = False


class GradientAccumulator:
    """Manages gradient accumulation across micro-batches"""
    
    def __init__(self, config: GradientAccumulationConfig):
        self.config = config
        self.buffer = GradientBuffer()
        self.sync_counter = 0
        self.total_sync_operations = 0
        self.accumulated_steps = 0
    
    def accumulate_gradient(self, param_name: str, gradient: Any) -> bool:
        """
        Accumulate gradient from backward pass
        
        Returns:
            True if synchronization is needed, False otherwise
        """
        self.buffer.add_gradient(param_name, gradient)
        self.accumulated_steps += 1
        self.sync_counter += 1
        
        # Check if we should synchronize
        should_sync = self._should_synchronize()
        
        if should_sync:
            self.total_sync_operations += 1
            self.sync_counter = 0
        
        return should_sync
    
    def _should_synchronize(self) -> bool:
        """Determine if synchronization is needed"""
        if self.config.mode == GradientAccumulationMode.IMMEDIATE:
            return True
        elif self.config.mode == GradientAccumulationMode.DELAYED:
            return self.buffer.accumulation_step >= self.config.num_accumulation_steps
        elif self.config.mode == GradientAccumulationMode.LAZY:
            return False  # Manual sync required
        else:
            return self.buffer.accumulation_step >= self.config.num_accumulation_steps
    
    def sync_gradients(self, comm_manager: Optional[Any] = None):
        """Synchronize accumulated gradients across devices"""
        if comm_manager:
            # Perform all-reduce on accumulated gradients
            for param_name in self.buffer.gradients.keys():
                grad = self.buffer.get_accumulated_gradient(param_name)
                if grad is not None:
                    # In real implementation: comm_manager.allreduce(grad)
                    pass
        
        self.buffer.synchronized = True
    
    def get_synchronized_gradients(self) -> Dict[str, Any]:
        """Get all synchronized gradients"""
        return {
            param_name: self.buffer.get_accumulated_gradient(param_name)
            for param_name in self.buffer.gradients.keys()
        }
    
    def reset(self):
        """Reset accumulator for next round"""
        self.buffer.clear()
        self.accumulated_steps = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get accumulation statistics"""
        return {
            "accumulated_steps": self.accumulated_steps,
            "total_sync_operations": self.total_sync_operations,
            "buffer_size": len(self.buffer.gradients),
            "is_synchronized": self.buffer.synchronized
        }


class DistributedGradientSync:
    """Synchronizes gradients across devices/nodes"""
    
    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.allreduce_time_ms = 0
        self.num_allreduce_ops = 0
    
    def allreduce_gradients(self, gradients: Dict[str, Any], 
                           async_op: bool = False,
                           group: Optional[int] = None) -> Dict[str, Any]:
        """All-reduce gradients across all devices"""
        start = time.perf_counter()
        
        # In real implementation: use torch.distributed.all_reduce
        synchronized = {name: grad for name, grad in gradients.items()}
        
        self.allreduce_time_ms += (time.perf_counter() - start) * 1000
        self.num_allreduce_ops += 1
        
        return synchronized
    
    def reduce_scatter_gradients(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Reduce-scatter gradients (tensor parallelism)"""
        # Each device gets portion of gradients
        num_params = len(gradients)
        scattered = {}
        
        for i, (name, grad) in enumerate(gradients.items()):
            if i % self.world_size == self.rank:
                scattered[name] = grad
        
        return scattered
    
    def get_sync_stats(self) -> Dict[str, float]:
        """Get synchronization statistics"""
        avg_reduce_time = (self.allreduce_time_ms / max(1, self.num_allreduce_ops))
        return {
            "total_allreduce_time_ms": self.allreduce_time_ms,
            "num_allreduce_ops": self.num_allreduce_ops,
            "avg_allreduce_time_ms": avg_reduce_time
        }


class GradientCheckpointing:
    """Enables gradient checkpointing for memory efficiency"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.checkpoint_segments: Dict[str, Any] = {}
    
    def save_checkpoint(self, segment_id: str, activations: Any):
        """Save activations for gradient checkpoint"""
        if self.enabled:
            self.checkpoint_segments[segment_id] = activations
    
    def load_checkpoint(self, segment_id: str) -> Optional[Any]:
        """Load activations from checkpoint"""
        if self.enabled:
            return self.checkpoint_segments.get(segment_id)
        return None
    
    def get_memory_savings(self, total_activations_mb: float) -> float:
        """Estimate memory savings from checkpointing"""
        if self.enabled:
            # Rough estimate: save sqrt(N) activations instead of N
            return total_activations_mb * (1 - (1 / (len(self.checkpoint_segments) ** 0.5)))
        return 0.0


@dataclass
class GradientCompressionConfig:
    """Configuration for gradient compression"""
    enabled: bool = False
    compression_ratio: float = 0.1  # Keep top 10% gradients
    compression_threshold: float = 1e-5
    decompress_on_sync: bool = True


class GradientCompression:
    """Compresses gradients for efficient communication"""
    
    def __init__(self, config: GradientCompressionConfig):
        self.config = config
        self.compression_stats = {
            "total_gradients_before": 0,
            "total_gradients_after": 0,
            "compression_ratio": 1.0
        }
    
    def compress_gradients(self, gradients: Dict[str, Any]) -> Dict[str, Any]:
        """Compress gradients"""
        if not self.config.enabled:
            return gradients
        
        compressed = {}
        total_before = 0
        total_after = 0
        
        for name, grad in gradients.items():
            # Count elements
            total_before += self._count_elements(grad)
            
            # Keep top K gradients (sparsify)
            # In real implementation: use actual gradient compression
            compressed[name] = grad
            total_after += self._count_elements(compressed[name])
        
        self.compression_stats["total_gradients_before"] = total_before
        self.compression_stats["total_gradients_after"] = total_after
        if total_before > 0:
            self.compression_stats["compression_ratio"] = total_after / total_before
        
        return compressed
    
    def decompress_gradients(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """Decompress gradients"""
        # In real implementation: restore from compressed form
        return compressed
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return self.compression_stats
    
    def _count_elements(self, grad: Any) -> int:
        """Count elements in gradient"""
        if hasattr(grad, '__len__'):
            try:
                return len(grad)
            except:
                return 1
        return 1
