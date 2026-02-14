"""
Distributed Checkpoint Management and State Persistence

Handles saving and loading distributed model checkpoints with state synchronization,
recovery, and version tracking across multiple nodes.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import json
import time
from pathlib import Path


class CheckpointFormat(Enum):
    """Checkpoint storage formats"""
    CONSOLIDATED = "consolidated"  # Single file per rank
    DISTRIBUTED = "distributed"  # Sharded across storage
    FULLY_SHARDED = "fully_sharded"  # FSDP format
    SAFETENSORS = "safetensors"  # SafeTensors format


@dataclass
class CheckpointMetadata:
    """Metadata for checkpoint"""
    checkpoint_id: str
    timestamp: float
    step: int
    epoch: int
    world_size: int
    tp_size: int
    pp_size: int
    dp_size: int
    model_hidden_dim: int
    num_layers: int
    total_params: int
    sharding_strategy: str
    format: CheckpointFormat = CheckpointFormat.DISTRIBUTED
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "step": self.step,
            "epoch": self.epoch,
            "world_size": self.world_size,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "dp_size": self.dp_size,
            "model_hidden_dim": self.model_hidden_dim,
            "num_layers": self.num_layers,
            "total_params": self.total_params,
            "sharding_strategy": self.sharding_strategy,
            "format": self.format.value,
            "tags": self.tags
        }


@dataclass
class OptimizationState:
    """State of optimizer across distributed training"""
    step: int
    adam_m: Optional[Dict[str, Any]] = None  # First moment (Adam)
    adam_v: Optional[Dict[str, Any]] = None  # Second moment (Adam)
    lr_scheduler_state: Optional[Dict[str, Any]] = None
    scaler_state: Optional[float] = None  # Mixed precision scalar


class CheckpointManager:
    """Manages distributed checkpointing"""
    
    def __init__(self, checkpoint_dir: str, rank: int = 0, world_size: int = 1):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.world_size = world_size
        self.checkpoints: Dict[str, CheckpointMetadata] = {}
        self._load_checkpoint_registry()
    
    def save_checkpoint(self, model_state: Dict[str, Any], optimizer_state: OptimizationState,
                       metadata: CheckpointMetadata, format: CheckpointFormat = CheckpointFormat.DISTRIBUTED) -> str:
        """Save distributed checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{metadata.checkpoint_id}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model state (rank-sharded)
        model_shard_path = checkpoint_path / f"model_rank{self.rank}.bin"
        self._save_shard(model_state, model_shard_path)
        
        # Save optimizer state
        opt_shard_path = checkpoint_path / f"optimizer_rank{self.rank}.bin"
        opt_state_dict = {
            "step": optimizer_state.step,
            "adam_m": optimizer_state.adam_m,
            "adam_v": optimizer_state.adam_v,
        }
        self._save_shard(opt_state_dict, opt_shard_path)
        
        # Rank 0 saves metadata and index
        if self.rank == 0:
            metadata_path = checkpoint_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Create checkpoint index
            index = {
                "format": format.value,
                "metadata": metadata.to_dict(),
                "shards": [f"model_rank{i}.bin" for i in range(self.world_size)],
                "optimizer_shards": [f"optimizer_rank{i}.bin" for i in range(self.world_size)]
            }
            index_path = checkpoint_path / "index.json"
            with open(index_path, 'w') as f:
                json.dump(index, f, indent=2)
            
            self.checkpoints[metadata.checkpoint_id] = metadata
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_id: str) -> Tuple[Dict[str, Any], OptimizationState, CheckpointMetadata]:
        """Load distributed checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}"
        
        # Load metadata (from rank 0)
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
            metadata = CheckpointMetadata(**metadata_dict)
        
        # Load model shard
        model_shard_path = checkpoint_path / f"model_rank{self.rank}.bin"
        model_state = self._load_shard(model_shard_path)
        
        # Load optimizer shard
        opt_shard_path = checkpoint_path / f"optimizer_rank{self.rank}.bin"
        opt_dict = self._load_shard(opt_shard_path)
        optimizer_state = OptimizationState(
            step=opt_dict.get("step", 0),
            adam_m=opt_dict.get("adam_m"),
            adam_v=opt_dict.get("adam_v")
        )
        
        return model_state, optimizer_state, metadata
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get ID of most recent checkpoint"""
        if not self.checkpoints:
            return None
        return max(self.checkpoints.items(), key=lambda x: x[1].timestamp)[0]
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints"""
        return list(self.checkpoints.keys())
    
    def delete_old_checkpoints(self, keep_last_n: int = 3):
        """Delete old checkpoints except last N"""
        sorted_ckpts = sorted(self.checkpoints.items(), 
                             key=lambda x: x[1].timestamp, reverse=True)
        
        for ckpt_id, _ in sorted_ckpts[keep_last_n:]:
            ckpt_path = self.checkpoint_dir / f"checkpoint_{ckpt_id}"
            import shutil
            if ckpt_path.exists():
                shutil.rmtree(ckpt_path)
            del self.checkpoints[ckpt_id]
    
    def _save_shard(self, state: Dict[str, Any], path: Path):
        """Save state shard"""
        # In real implementation: use torch.save or safetensors
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def _load_shard(self, path: Path) -> Dict[str, Any]:
        """Load state shard"""
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _load_checkpoint_registry(self):
        """Load metadata for all existing checkpoints"""
        for ckpt_dir in self.checkpoint_dir.glob("checkpoint_*"):
            metadata_path = ckpt_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                    ckpt_id = ckpt_dir.name.replace("checkpoint_", "")
                    self.checkpoints[ckpt_id] = CheckpointMetadata(**metadata_dict)


class IncrementalCheckpointing:
    """Saves only weights that changed since last checkpoint"""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.last_checkpoint_state: Optional[Dict[str, Any]] = None
    
    def save_incremental(self, model_state: Dict[str, Any], 
                        optimizer_state: OptimizationState,
                        metadata: CheckpointMetadata) -> Tuple[str, float]:
        """Save only changed weights
        
        Returns:
            (checkpoint_path, compression_ratio)
        """
        changed_state = {}
        total_params = 0
        changed_params = 0
        
        if self.last_checkpoint_state:
            for key, value in model_state.items():
                total_params += self._count_parameters(value)
                # Simple change detection (in real: use hash or delta)
                if key not in self.last_checkpoint_state or self.last_checkpoint_state[key] is not value:
                    changed_state[key] = value
                    changed_params += self._count_parameters(value)
        else:
            changed_state = model_state
            total_params = changed_params = self._count_parameters(model_state)
        
        compression_ratio = changed_params / max(1, total_params)
        
        path = self.checkpoint_manager.save_checkpoint(
            changed_state, optimizer_state, metadata
        )
        self.last_checkpoint_state = model_state
        
        return path, compression_ratio
    
    def _count_parameters(self, state: Any) -> int:
        """Count parameters in state"""
        if isinstance(state, dict):
            return sum(self._count_parameters(v) for v in state.values())
        elif hasattr(state, '__len__'):
            try:
                return len(state)
            except:
                return 1
        return 1


class ZeroCheckpointing:
    """ZeRO stage-1, 2, 3 checkpoint consolidation"""
    
    def __init__(self, checkpoint_manager: CheckpointManager, world_size: int):
        self.checkpoint_manager = checkpoint_manager
        self.world_size = world_size
    
    def consolidate_zero_stage1(self, optimizer_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate ZeRO Stage 1 (optimizer state sharding)"""
        consolidated = {}
        for rank_id, state in enumerate(optimizer_states):
            consolidated[f"rank_{rank_id}"] = state
        return consolidated
    
    def consolidate_zero_stage2(self, model_shards: List[Dict[str, Any]], 
                               optimizer_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate ZeRO Stage 2 (gradient + optimizer state sharding)"""
        consolidated_model = {}
        for rank_id, shard in enumerate(model_shards):
            for key, value in shard.items():
                if key not in consolidated_model:
                    consolidated_model[key] = []
                consolidated_model[key].append(value)
        
        return {
            "model": consolidated_model,
            "optimizer_states": {f"rank_{i}": state for i, state in enumerate(optimizer_states)}
        }
    
    def consolidate_zero_stage3(self, model_shards: List[Dict[str, Any]], 
                               optimizer_states: List[Dict[str, Any]],
                               gradient_shards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate ZeRO Stage 3 (all parameters sharded)"""
        consolidated_model = {}
        for rank_id, shard in enumerate(model_shards):
            for key, value in shard.items():
                if key not in consolidated_model:
                    consolidated_model[key] = []
                consolidated_model[key].append(value)
        
        return {
            "model": consolidated_model,
            "optimizer_states": {f"rank_{i}": state for i, state in enumerate(optimizer_states)},
            "gradients": {f"rank_{i}": grad for i, grad in enumerate(gradient_shards)}
        }


@dataclass
class CheckpointRecoveryStrategy:
    """Strategy for checkpoint recovery after failure"""
    enable_async_checkpoint: bool = True
    checkpoint_interval_steps: int = 1000
    checkpoint_interval_seconds: int = 3600  # 1 hour
    max_checkpoints_to_keep: int = 5
    enable_incremental: bool = True
    
    def should_checkpoint(self, step: int, elapsed_seconds: float) -> bool:
        """Determine if checkpoint should be saved"""
        return (step % self.checkpoint_interval_steps == 0 or 
                elapsed_seconds >= self.checkpoint_interval_seconds)
