"""
Distributed Communication Primitives

Implements collective operations (allreduce, allgather, broadcast, scatter) and
point-to-point communication patterns for distributed tensor parallelism.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time


class CollectiveOp(Enum):
    """Supported collective operations"""
    ALLREDUCE = "allreduce"
    ALLGATHER = "allgather"
    BROADCAST = "broadcast"
    SCATTER = "scatter"
    REDUCE_SCATTER = "reduce_scatter"
    REDUCE = "reduce"


class ReduceOp(Enum):
    """Reduction operators"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    PRODUCT = "product"


class CommunicationBackend(Enum):
    """Supported communication backends"""
    NCCL = "nccl"  # NVIDIA Collective Communications Library
    GLOO = "gloo"  # Generic collective communication
    MPI = "mpi"  # Message Passing Interface
    UCC = "ucc"  # Unified Collective Communication


@dataclass
class CommunicationConfig:
    """Configuration for distributed communication"""
    backend: CommunicationBackend
    world_size: int
    rank: int
    master_addr: str = "localhost"
    master_port: int = 29500
    timeout_seconds: int = 30
    enable_gradient_compression: bool = False
    compress_threshold_bytes: int = 1024 * 1024  # 1MB
    enable_overlap: bool = True  # Overlap compute and communication
    
    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1


@dataclass
class CommunicationMetrics:
    """Metrics for communication performance"""
    op: str
    collective_op: CollectiveOp
    num_elements: int
    bytes_transferred: int
    duration_ms: float
    throughput_gbps: float  # Gigabytes per second
    latency_ns: float  # Nanoseconds
    
    @property
    def efficiency(self) -> float:
        """Communication efficiency (higher is better)"""
        # Ideal throughput depends on interconnect
        theoretical_max_gbps = 600  # RTX A100 NVLink
        return min(1.0, self.throughput_gbps / theoretical_max_gbps)


class CollectiveOperation(ABC):
    """Base class for collective operations"""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
    
    @abstractmethod
    def execute(self, data: Any, *args, **kwargs) -> Any:
        """Execute collective operation"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> CommunicationMetrics:
        """Get communication metrics"""
        pass


class AllReduceOperation(CollectiveOperation):
    """Reduce operation across all processes, then broadcast result"""
    
    def __init__(self, config: CommunicationConfig):
        super().__init__(config)
        self.last_metrics = None
    
    def execute(self, data: Any, reduce_op: ReduceOp = ReduceOp.SUM,
                async_op: bool = False, group: Optional[List[int]] = None) -> Any:
        """
        All-reduce operation
        
        Args:
            data: Tensor data to reduce
            reduce_op: Reduction operator (SUM, AVG, etc.)
            async_op: Non-blocking operation
            group: Communication group ranks
        
        Returns:
            Reduced tensor (or handle if async)
        """
        start_time = time.perf_counter()
        
        # Simulate allreduce: gather from all, reduce, broadcast
        if not self.config.is_distributed:
            result = data
        else:
            # In real implementation, would use NCCL/Gloo
            result = self._reduce_across_ranks(data, reduce_op)
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Record metrics
        num_elements = self._count_elements(data)
        bytes_transferred = num_elements * 4  # Assume float32
        self.last_metrics = CommunicationMetrics(
            op="allreduce",
            collective_op=CollectiveOp.ALLREDUCE,
            num_elements=num_elements,
            bytes_transferred=bytes_transferred * self.config.world_size,  # Each rank contributes
            duration_ms=duration_ms,
            throughput_gbps=(bytes_transferred * self.config.world_size / 1e9) / (duration_ms / 1000),
            latency_ns=duration_ms * 1e6
        )
        
        return result
    
    def get_metrics(self) -> CommunicationMetrics:
        return self.last_metrics or CommunicationMetrics(
            op="allreduce", collective_op=CollectiveOp.ALLREDUCE,
            num_elements=0, bytes_transferred=0, duration_ms=0,
            throughput_gbps=0, latency_ns=0
        )
    
    def _reduce_across_ranks(self, data: Any, reduce_op: ReduceOp) -> Any:
        # Simulated reduction (in real system: actual distributed reduction)
        return data
    
    def _count_elements(self, data: Any) -> int:
        # Count elements in tensor/array
        if isinstance(data, (list, tuple)):
            return len(data)
        return getattr(data, 'numel', lambda: 1)()


class AllGatherOperation(CollectiveOperation):
    """Gather data from all processes and distribute to all"""
    
    def __init__(self, config: CommunicationConfig):
        super().__init__(config)
        self.last_metrics = None
    
    def execute(self, data: Any, async_op: bool = False,
                group: Optional[List[int]] = None) -> List[Any]:
        """
        All-gather operation
        
        Returns:
            List of gathered data from all ranks
        """
        start_time = time.perf_counter()
        
        if not self.config.is_distributed:
            gathered = [data]
        else:
            gathered = [data] * self.config.world_size  # Simulated gather
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        num_elements = self._count_elements(data)
        bytes_per_rank = num_elements * 4
        self.last_metrics = CommunicationMetrics(
            op="allgather",
            collective_op=CollectiveOp.ALLGATHER,
            num_elements=num_elements * self.config.world_size,
            bytes_transferred=bytes_per_rank * self.config.world_size * self.config.world_size,
            duration_ms=duration_ms,
            throughput_gbps=(bytes_per_rank * self.config.world_size * self.config.world_size / 1e9) / (duration_ms / 1000),
            latency_ns=duration_ms * 1e6
        )
        
        return gathered
    
    def get_metrics(self) -> CommunicationMetrics:
        return self.last_metrics or CommunicationMetrics(
            op="allgather", collective_op=CollectiveOp.ALLGATHER,
            num_elements=0, bytes_transferred=0, duration_ms=0,
            throughput_gbps=0, latency_ns=0
        )
    
    def _count_elements(self, data: Any) -> int:
        if isinstance(data, (list, tuple)):
            return len(data)
        return getattr(data, 'numel', lambda: 1)()


class BroadcastOperation(CollectiveOperation):
    """Send data from one process to all others"""
    
    def __init__(self, config: CommunicationConfig):
        super().__init__(config)
        self.last_metrics = None
    
    def execute(self, data: Any, src_rank: int = 0,
                async_op: bool = False, group: Optional[List[int]] = None) -> Any:
        """Broadcast from source rank to all"""
        start_time = time.perf_counter()
        
        result = data  # In distributed: broadcast from src_rank
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        num_elements = self._count_elements(data)
        bytes_transferred = num_elements * 4
        
        self.last_metrics = CommunicationMetrics(
            op="broadcast",
            collective_op=CollectiveOp.BROADCAST,
            num_elements=num_elements,
            bytes_transferred=bytes_transferred * (self.config.world_size - 1),
            duration_ms=duration_ms,
            throughput_gbps=(bytes_transferred * (self.config.world_size - 1) / 1e9) / (duration_ms / 1000),
            latency_ns=duration_ms * 1e6
        )
        
        return result
    
    def get_metrics(self) -> CommunicationMetrics:
        return self.last_metrics or CommunicationMetrics(
            op="broadcast", collective_op=CollectiveOp.BROADCAST,
            num_elements=0, bytes_transferred=0, duration_ms=0,
            throughput_gbps=0, latency_ns=0
        )
    
    def _count_elements(self, data: Any) -> int:
        if isinstance(data, (list, tuple)):
            return len(data)
        return getattr(data, 'numel', lambda: 1)()


class ReduceScatterOperation(CollectiveOperation):
    """Reduce operation followed by scatter"""
    
    def __init__(self, config: CommunicationConfig):
        super().__init__(config)
        self.last_metrics = None
    
    def execute(self, data: Any, reduce_op: ReduceOp = ReduceOp.SUM,
                async_op: bool = False, group: Optional[List[int]] = None) -> Any:
        """Reduce-scatter: reduce then scatter pieces to each rank"""
        start_time = time.perf_counter()
        
        # Reduce and scatter
        reduced_size = len(data) // self.config.world_size if isinstance(data, (list, tuple)) else 1
        result = data[:reduced_size] if isinstance(data, (list, tuple)) else data
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        num_elements = self._count_elements(data)
        
        self.last_metrics = CommunicationMetrics(
            op="reduce_scatter",
            collective_op=CollectiveOp.REDUCE_SCATTER,
            num_elements=num_elements,
            bytes_transferred=(num_elements * 4 * self.config.world_size),
            duration_ms=duration_ms,
            throughput_gbps=(num_elements * 4 * self.config.world_size / 1e9) / (duration_ms / 1000),
            latency_ns=duration_ms * 1e6
        )
        
        return result
    
    def get_metrics(self) -> CommunicationMetrics:
        return self.last_metrics or CommunicationMetrics(
            op="reduce_scatter", collective_op=CollectiveOp.REDUCE_SCATTER,
            num_elements=0, bytes_transferred=0, duration_ms=0,
            throughput_gbps=0, latency_ns=0
        )
    
    def _count_elements(self, data: Any) -> int:
        if isinstance(data, (list, tuple)):
            return len(data)
        return getattr(data, 'numel', lambda: 1)()


class CommunicationManager:
    """Manages distributed communication operations"""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.operations = {
            CollectiveOp.ALLREDUCE: AllReduceOperation(config),
            CollectiveOp.ALLGATHER: AllGatherOperation(config),
            CollectiveOp.BROADCAST: BroadcastOperation(config),
            CollectiveOp.REDUCE_SCATTER: ReduceScatterOperation(config),
        }
        self.communication_stats = {
            "total_operations": 0,
            "total_bytes_transferred": 0,
            "total_duration_ms": 0
        }
    
    def allreduce(self, data: Any, reduce_op: ReduceOp = ReduceOp.SUM,
                 async_op: bool = False) -> Any:
        """All-reduce operation"""
        result = self.operations[CollectiveOp.ALLREDUCE].execute(data, reduce_op, async_op)
        self._update_stats(CollectiveOp.ALLREDUCE)
        return result
    
    def allgather(self, data: Any, async_op: bool = False) -> List[Any]:
        """All-gather operation"""
        result = self.operations[CollectiveOp.ALLGATHER].execute(data, async_op)
        self._update_stats(CollectiveOp.ALLGATHER)
        return result
    
    def broadcast(self, data: Any, src_rank: int = 0, async_op: bool = False) -> Any:
        """Broadcast operation"""
        result = self.operations[CollectiveOp.BROADCAST].execute(data, src_rank, async_op)
        self._update_stats(CollectiveOp.BROADCAST)
        return result
    
    def reduce_scatter(self, data: Any, reduce_op: ReduceOp = ReduceOp.SUM,
                      async_op: bool = False) -> Any:
        """Reduce-scatter operation"""
        result = self.operations[CollectiveOp.REDUCE_SCATTER].execute(data, reduce_op, async_op)
        self._update_stats(CollectiveOp.REDUCE_SCATTER)
        return result
    
    def barrier(self):
        """Synchronization barrier"""
        pass
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get accumulated communication statistics"""
        return {
            **self.communication_stats,
            "avg_bytes_per_op": (
                self.communication_stats["total_bytes_transferred"] / 
                max(1, self.communication_stats["total_operations"])
            ),
            "avg_duration_ms": (
                self.communication_stats["total_duration_ms"] / 
                max(1, self.communication_stats["total_operations"])
            )
        }
    
    def reset_stats(self):
        """Reset communication statistics"""
        self.communication_stats = {
            "total_operations": 0,
            "total_bytes_transferred": 0,
            "total_duration_ms": 0
        }
    
    def _update_stats(self, op: CollectiveOp):
        """Update statistics after operation"""
        self.communication_stats["total_operations"] += 1
        op_instance = self.operations[op]
        metrics = op_instance.get_metrics()
        if metrics:
            self.communication_stats["total_bytes_transferred"] += metrics.bytes_transferred
            self.communication_stats["total_duration_ms"] += metrics.duration_ms


class OverlappedCommunication:
    """Enables computation-communication overlap"""
    
    def __init__(self, comm_manager: CommunicationManager):
        self.comm_manager = comm_manager
        self.pending_ops: List[Dict[str, Any]] = []
    
    def submit_async_allreduce(self, data: Any, reduce_op: ReduceOp = ReduceOp.SUM) -> str:
        """Submit non-blocking allreduce, continue with computation"""
        op_id = f"allreduce_{len(self.pending_ops)}"
        self.pending_ops.append({
            "id": op_id,
            "op": CollectiveOp.ALLREDUCE,
            "data": data,
            "reduce_op": reduce_op,
            "completed": False
        })
        return op_id
    
    def wait_for_op(self, op_id: str) -> Any:
        """Wait for pending operation to complete"""
        for op in self.pending_ops:
            if op["id"] == op_id:
                # Execute if not done
                if not op["completed"]:
                    result = self.comm_manager.allreduce(op["data"], op["reduce_op"])
                    op["completed"] = True
                    op["result"] = result
                return op.get("result")
        raise ValueError(f"Unknown operation: {op_id}")
    
    def wait_all(self) -> Dict[str, Any]:
        """Wait for all pending operations"""
        results = {}
        for op in self.pending_ops:
            if not op["completed"]:
                result = self.comm_manager.allreduce(op["data"], op.get("reduce_op", ReduceOp.SUM))
                op["completed"] = True
                op["result"] = result
            results[op["id"]] = op.get("result")
        return results
