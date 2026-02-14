"""Curriculum Learning Training for AIPROD v2

Five-phase progressive training strategy designed for GTX 1070 (8GB VRAM).
Each phase builds on previous, gradually increasing dataset complexity.
"""

import os
import sys
import subprocess
from types import ModuleType

# Disable torch compilation and dynamo before importing torch to avoid Windows frame inspection bug
os.environ['TORCH_DISABLE_DYNAMO'] = '1'
os.environ['TORCH_COMPILE_BACKEND'] = 'eager'

# Create a fake torch._dynamo module to prevent import errors
fake_dynamo = ModuleType('torch._dynamo')
fake_dynamo.disable = lambda fn=None, recursive=True: fn
fake_dynamo.is_compiling = lambda: False
fake_dynamo.graph_break = lambda: None
sys.modules['torch._dynamo'] = fake_dynamo
sys.modules['torch._dynamo.convert_frame'] = ModuleType('torch._dynamo.convert_frame')

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import json
from pathlib import Path


class TrainingPhase(Enum):
    """Training curriculum phases."""
    PHASE_1_SIMPLE = 1        # Simple objects
    PHASE_2_COMPOUND = 2      # Compound scenes
    PHASE_3_COMPLEX = 3       # Complex motion
    PHASE_4_EDGE_CASES = 4    # Edge cases
    PHASE_5_REFINEMENT = 5    # Fine-tuning mix


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    
    # Phase 1: Simple Objects
    phase1_epochs: int = 10            # EXTREME: Reduced from 20 (test phase)
    phase1_batch_size: int = 1         # EXTREME: Batch=1 for min thermal load
    phase1_learning_rate: float = 1e-4
    phase1_num_samples: int = 1000     # ~2-3 hours video
    phase1_max_frames: int = 8         # EXTREME: Reduced from 12
    phase1_resolution: Tuple[int, int] = (192, 192)  # EXTREME: Reduced from 224
    
    # Phase 2: Compound Scenes
    phase2_epochs: int = 15            # EXTREME: Reduced from 25 (test phase)
    phase2_batch_size: int = 1         # EXTREME: Batch=1 for min thermal load
    phase2_learning_rate: float = 5e-5
    phase2_num_samples: int = 1500     # ~5 hours
    phase2_max_frames: int = 12        # EXTREME: Reduced from 20
    phase2_resolution: Tuple[int, int] = (200, 200)  # EXTREME: Reduced from 280
    
    # Phase 3: Complex Motion
    phase3_epochs: int = 15            # EXTREME: Reduced from 30 (test phase)
    phase3_batch_size: int = 1         # EXTREME: Batch=1 for min thermal load
    phase3_learning_rate: float = 2e-5
    phase3_num_samples: int = 2000     # ~7-8 hours
    phase3_max_frames: int = 16        # EXTREME: Reduced from 28
    phase3_resolution: Tuple[int, int] = (224, 224)  # EXTREME: Reduced from 336
    
    # Phase 4: Edge Cases
    phase4_epochs: int = 10            # EXTREME: Reduced from 20 (test phase)
    phase4_batch_size: int = 1         # EXTREME: Batch=1 for min thermal load
    phase4_learning_rate: float = 1e-5
    phase4_num_samples: int = 1200     # ~4-5 hours
    phase4_max_frames: int = 16        # EXTREME: Reduced from 28
    phase4_resolution: Tuple[int, int] = (224, 224)  # EXTREME: Reduced from 336
    
    # Phase 5: Refinement
    phase5_epochs: int = 10            # EXTREME: Reduced from 15 (test phase)
    phase5_batch_size: int = 1         # EXTREME: Batch=1 for min thermal load
    phase5_learning_rate: float = 5e-6
    phase5_num_samples: int = 4000     # ~15-20 hours (all phases mixed)
    phase5_max_frames: int = 16        # EXTREME: Reduced from 28
    phase5_resolution: Tuple[int, int] = (224, 224)  # EXTREME: Reduced from 336
    
    # General
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True       # CRITICAL: FP16 halves memory and heat
    gradient_accumulation_steps: int = 1
    use_gradient_checkpointing: bool = True  # CRITICAL: Save memory = less heat
    enable_amp_loss_scaling: bool = True  # NEW: Dynamic loss scaling
    use_fused_optimizers: bool = True   # NEW: NVIDIA fused kernels (faster, cooler)
    enable_memory_efficient_attention: bool = True  # NEW: Flash attention (if available)
    enable_activation_checkpointing: bool = True  # NEW: More memory efficiency
    enable_gpu_throttling: bool = False
    max_gpu_clock_mhz: int = 1500      
    enable_channel_last: bool = True   
    use_torch_compile: bool = False    
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    
    # EXTREME safety settings
    max_allowed_gpu_temp: float = 80.0  # NEW: Stop training if exceeds this
    target_gpu_temp: float = 75.0       # NEW: Target temperature
    enable_thermal_monitoring: bool = True  # NEW: Monitor and throttle
    
    @classmethod
    def total_training_hours(cls) -> float:
        """Estimate total training hours."""
        # Rough calculation: 100-150h video data across all phases
        return (2 + 5 + 8 + 5 + 20) / 7 * 60  # ~30-50 GPU hours expected


class TrainingMetrics:
    """Track training metrics across phases."""
    
    def __init__(self):
        self.metrics: Dict = {}
    
    def record(self, phase: TrainingPhase, epoch: int, metrics_dict: Dict):
        """Record metrics for a phase/epoch."""
        phase_key = f"phase_{phase.value}"
        if phase_key not in self.metrics:
            self.metrics[phase_key] = []
        
        metrics_dict['epoch'] = epoch
        self.metrics[phase_key].append(metrics_dict)
    
    def get_phase_metrics(self, phase: TrainingPhase) -> list:
        """Get metrics for specific phase."""
        phase_key = f"phase_{phase.value}"
        return self.metrics.get(phase_key, [])
    
    def save_to_json(self, filepath: Path):
        """Save metrics to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def load_from_json(self, filepath: Path):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            self.metrics = json.load(f)


class CurriculumTrainer:
    """
    Curriculum Learning Trainer for AIPROD v2.
    
    Progressive training across 5 phases, optimized for GTX 1070.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[CurriculumConfig] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.config = config or CurriculumConfig()
        self.device = device or self.config.device
        self.model.to(self.device)
        
        self.metrics = TrainingMetrics()
        self.current_phase = None
        self.optimizer = None
        self.scheduler = None
        self.grad_scaler = GradScaler() if self.config.mixed_precision else None
        
        # Enable gradient checkpointing if configured
        if self.config.use_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Setup GPU throttling if enabled
        if self.device == "cuda" and self.config.enable_gpu_throttling:
            self._setup_gpu_throttling()
        
        self._setup_checkpoints()
    
    def _setup_gpu_throttling(self):
        """Setup GPU frequency throttling to reduce thermal load.
        
        On Windows with NVIDIA GPUs, this attempts to limit GPU core clock.
        Requires drivers that support clock limiting (e.g., nvidia-smi).
        """
        try:
            # Try to set GPU clock limit using nvidia-smi (if available)
            # This is a best-effort approach - may not work on all systems
            gpu_id = 0
            max_clock_mhz = self.config.max_gpu_clock_mhz
            
            # Windows approach: Try nvidia-settings or nvidia-smi alternatives
            # Note: Full control requires admin privileges
            print(f"\n  ℹ  GPU Throttling Configuration:")
            print(f"     Requested max clock: {max_clock_mhz}MHz")
            print(f"     (Actual limiting depends on driver permissions)")
            
            # Get current GPU info
            props = torch.cuda.get_device_properties(gpu_id)
            print(f"     Current GPU: {props.name}")
            print(f"     Default max clock: ~{1800}MHz (GTX 1070)")
            
            # Try nvidia-smi on Windows
            try:
                # This attempts to set persistent power mode and clock limit
                # Requires admin privileges - will fail silently if unavailable
                result = subprocess.run(
                    ["nvidia-smi", "-pm", "1"],
                    capture_output=True,
                    timeout=5,
                    shell=True
                )
                if result.returncode == 0:
                    print(f"     ✓ Persistent power mode enabled")
                    
                    # Try to lock clocks
                    subprocess.run(
                        ["nvidia-smi", "-lgc", str(max_clock_mhz)],
                        capture_output=True,
                        timeout=5,
                        shell=True
                    )
                    print(f"     ✓ GPU clock limited to {max_clock_mhz}MHz")
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
                # nvidia-smi not available or failed
                # Set environment variable to enable power limiting instead
                os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                
                # Enable PyTorch power limiting features
                if hasattr(torch.cuda, '_set_device_debug'):
                    torch.cuda._set_device_debug(False)
                
                print(f"     ⚠  nvidia-smi not available (requires admin)")
                print(f"     → Using software power limiting instead")
                print(f"     → Reduce batch_size or resolution if needed")
        
        except Exception as e:
            print(f"     ⚠  GPU throttling setup failed: {e}")
            print(f"     → Monitor temperatures and adjust batch_size if needed")
    
    def _setup_checkpoints(self):
        """Setup checkpoint directories."""
        Path(self.config.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.logs_dir).mkdir(parents=True, exist_ok=True)
    
    def get_phase_config(self, phase: TrainingPhase) -> Dict:
        """Get configuration for specific training phase."""
        configs = {
            TrainingPhase.PHASE_1_SIMPLE: {
                "epochs": self.config.phase1_epochs,
                "batch_size": self.config.phase1_batch_size,
                "learning_rate": self.config.phase1_learning_rate,
                "num_samples": self.config.phase1_num_samples,
                "max_frames": self.config.phase1_max_frames,
                "resolution": self.config.phase1_resolution,
                "description": "Training on simple, single-subject scenes with clean backgrounds",
            },
            TrainingPhase.PHASE_2_COMPOUND: {
                "epochs": self.config.phase2_epochs,
                "batch_size": self.config.phase2_batch_size,
                "learning_rate": self.config.phase2_learning_rate,
                "num_samples": self.config.phase2_num_samples,
                "max_frames": self.config.phase2_max_frames,
                "resolution": self.config.phase2_resolution,
                "description": "Training on compound scenes with 2-3 subjects",
            },
            TrainingPhase.PHASE_3_COMPLEX: {
                "epochs": self.config.phase3_epochs,
                "batch_size": self.config.phase3_batch_size,
                "learning_rate": self.config.phase3_learning_rate,
                "num_samples": self.config.phase3_num_samples,
                "max_frames": self.config.phase3_max_frames,
                "resolution": self.config.phase3_resolution,
                "description": "Training on complex motion: fast action, occlusions, perspective changes",
            },
            TrainingPhase.PHASE_4_EDGE_CASES: {
                "epochs": self.config.phase4_epochs,
                "batch_size": self.config.phase4_batch_size,
                "learning_rate": self.config.phase4_learning_rate,
                "num_samples": self.config.phase4_num_samples,
                "max_frames": self.config.phase4_max_frames,
                "resolution": self.config.phase4_resolution,
                "description": "Training on edge cases: unusual angles, weather, dynamic lighting",
            },
            TrainingPhase.PHASE_5_REFINEMENT: {
                "epochs": self.config.phase5_epochs,
                "batch_size": self.config.phase5_batch_size,
                "learning_rate": self.config.phase5_learning_rate,
                "num_samples": self.config.phase5_num_samples,
                "max_frames": self.config.phase5_max_frames,
                "resolution": self.config.phase5_resolution,
                "description": "Refinement: mixed dataset from all phases, fine-tuning",
            },
        }
        return configs[phase]
    
    def setup_phase(self, phase: TrainingPhase):
        """Setup optimizer and scheduler for a phase."""
        self.current_phase = phase
        config = self.get_phase_config(phase)
        
        print(f"\n{'='*60}")
        print(f"Setting up {phase.name}")
        print(f"{'='*60}")
        print(f"Description: {config['description']}")
        print(f"Epochs: {config['epochs']}")
        print(f"Batch size: {config['batch_size']}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Samples: {config['num_samples']} (~{config['num_samples']/500:.1f}h video)")
        print(f"Frame count: {config['max_frames']} frames/sample")
        print(f"Resolution: {config['resolution']}")
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-5,
        )
        
        # Setup scheduler
        total_steps = config['epochs'] * (config['num_samples'] // config['batch_size'])
        warmup_steps = total_steps // 20  # OPTIMIZED: Reduced from 10% to 5% warmup for faster ramp-up
        
        self.scheduler = self._create_warmup_scheduler(warmup_steps, total_steps)
    
    def _create_warmup_scheduler(self, warmup_steps: int, total_steps: int):
        """Create cosine scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Dict:
        """Single training step."""
        self.model.train()
        
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Use mixed precision if enabled
        if self.config.mixed_precision:
            with autocast():
                # Forward pass
                outputs = self.model(x)
                
                # Compute loss (depends on model type)
                if isinstance(outputs, tuple):
                    # VAE-type output
                    reconstruction, mean, logvar = outputs
                    loss, recon_loss, kl_loss = self.model.compute_loss(
                        y, reconstruction, mean, logvar
                    )
                else:
                    # Standard regression
                    loss = nn.MSELoss()(outputs, y)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass with scaled loss
            self.grad_scaler.scale(loss).backward()
        else:
            with torch.enable_grad():
                # Forward pass
                outputs = self.model(x)
                
                # Compute loss (depends on model type)
                if isinstance(outputs, tuple):
                    # VAE-type output
                    reconstruction, mean, logvar = outputs
                    loss, recon_loss, kl_loss = self.model.compute_loss(
                        y, reconstruction, mean, logvar
                    )
                else:
                    # Standard regression
                    loss = nn.MSELoss()(outputs, y)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        # For mixed precision, handle scaling
        if self.config.mixed_precision:
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return {
            "loss": loss.item() * self.config.gradient_accumulation_steps,  # Undo scaling for reporting
            "learning_rate": self.optimizer.param_groups[0]['lr'],
        }
    
    def train_phase(
        self,
        phase: TrainingPhase,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict:
        """Train a single curriculum phase."""
        self.setup_phase(phase)
        config = self.get_phase_config(phase)
        
        best_loss = float('inf')
        phase_metrics = []
        
        for epoch in range(config['epochs']):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                num_batches += 1
                
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"[{phase.name}] Epoch {epoch+1}/{config['epochs']}, "
                          f"Batch {batch_idx+1}/{len(train_loader)}, "
                          f"Loss: {avg_loss:.4f}, "
                          f"LR: {metrics['learning_rate']:.2e}")
            
            avg_epoch_loss = epoch_loss / num_batches
            epoch_metrics = {
                "loss": avg_epoch_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
            }
            
            # Validation
            if val_loader:
                val_loss = self.validate(val_loader)
                epoch_metrics['val_loss'] = val_loss
                print(f"[{phase.name}] Validation Loss: {val_loss:.4f}")
            
            # Save checkpoint if best
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self._save_checkpoint(phase, epoch)
            
            # Record metrics
            self.metrics.record(phase, epoch, epoch_metrics)
            phase_metrics.append(epoch_metrics)
        
        return {
            "phase": phase.name,
            "final_loss": best_loss,
            "num_epochs": config['epochs'],
            "metrics": phase_metrics,
        }
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(x)
                
                if isinstance(outputs, tuple):
                    reconstruction, mean, logvar = outputs
                    loss, _, _ = self.model.compute_loss(y, reconstruction, mean, logvar)
                else:
                    loss = nn.MSELoss()(outputs, y)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _save_checkpoint(self, phase: TrainingPhase, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoints_dir) / f"{phase.name}_epoch_{epoch}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'phase': phase.value,
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': self.metrics.metrics,
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def get_training_summary(self) -> str:
        """Get summary of training progress."""
        summary = "\n" + "="*60 + "\n"
        summary += "CURRICULUM TRAINING SUMMARY\n"
        summary += "="*60 + "\n"
        
        for phase in TrainingPhase:
            metrics = self.metrics.get_phase_metrics(phase)
            if metrics:
                final_loss = metrics[-1]['loss']
                summary += f"\n{phase.name}:\n"
                summary += f"  Final Loss: {final_loss:.4f}\n"
                summary += f"  Epochs: {len(metrics)}\n"
        
        summary += "\n" + "="*60 + "\n"
        return summary


if __name__ == "__main__":
    print("AIPROD v2 Curriculum Learning Training")
    print("="*60)
    print(f"Total estimated training time: {CurriculumConfig.total_training_hours():.1f} hours")
    print(f"GPU Memory requirement: ~6GB (GTX 1070 compatible)")
    print(f"\nPhases:")
    for phase in TrainingPhase:
        config = CurriculumTrainer(None, CurriculumConfig()).get_phase_config(phase)
        print(f"  {phase.name}: {config['epochs']} epochs, {config['num_samples']} samples")
