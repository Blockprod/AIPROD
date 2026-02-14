#!/usr/bin/env python3
"""
AIPROD v2 Phase 1: Curriculum Learning Training

Main training orchestrator for the complete 5-phase curriculum.
Run this script to begin training models on GTX 1070.

Usage:
    python train.py [--resume-phase N] [--device cuda/cpu]
"""

import os
import sys
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

import argparse
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import HybridBackbone, VideoVAE, MultilingualTextEncoder
from training import CurriculumTrainer, TrainingPhase, CurriculumConfig
from data import VideoDataLoader


class Phase1MLTraining:
    """Orchestrator for Phase 1 ML training."""
    
    def __init__(self, checkpoint_dir: Path = None, device: str = None):
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints/phase1")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"\n{'='*60}")
        print(f"AIPROD v2 Phase 1: ML Training Initialization")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        self._build_models()
        self._setup_training()
    
    def _build_models(self):
        """Build core models."""
        print("\nBuilding models...")
        
        # Hybrid Backbone
        self.backbone = HybridBackbone(
            dim=768,
            num_attention_layers=30,
            num_cnn_layers=18,
            num_heads=8,
            vocab_size=32000,
            max_seq_length=4096,
        )
        backbone_params = self.backbone.get_num_params() / 1e6
        print(f"  ✓ HybridBackbone: {backbone_params:.1f}M parameters")
        
        # VideoVAE
        self.vae = VideoVAE(latent_dim=256, beta=0.1)
        vae_params = sum(p.numel() for p in self.vae.parameters()) / 1e6
        print(f"  ✓ VideoVAE: {vae_params:.1f}M parameters")
        
        # Multilingual Text Encoder
        self.text_encoder = MultilingualTextEncoder(output_dim=768)
        text_params = sum(p.numel() for p in self.text_encoder.parameters()) / 1e6
        print(f"  ✓ MultilingualTextEncoder: {text_params:.1f}M parameters")
        
        total_params = backbone_params + vae_params + text_params
        print(f"\nTotal parameters: {total_params:.1f}M")
        
        # For Phase 1, we'll train VAE first (most memory intensive)
        self.current_model = self.vae
    
    def _setup_training(self):
        """Setup training configuration."""
        print("\nSetting up training...")
        
        self.config = CurriculumConfig(device=self.device)
        self.trainer = CurriculumTrainer(
            model=self.current_model,
            config=self.config,
            device=self.device,
        )
        
        print("  ✓ Trainer configured")
        print(f"  ✓ Checkpoints: {self.checkpoint_dir}")
        print(f"  ✓ Total training hours: {CurriculumConfig.total_training_hours():.1f}h estimated")
    
    def create_data_loader(self, phase: TrainingPhase):
        """Create data loader for phase."""
        config = self.trainer.get_phase_config(phase)
        
        loader = VideoDataLoader.create_loader(
            video_dir=Path("data/videos"),
            phase=phase.value,
            batch_size=config['batch_size'],
            num_frames=config['max_frames'],
            resolution=config['resolution'],
            num_samples=config['num_samples'],
            shuffle=True,
        )
        
        return loader
    
    def train_curriculum(self, start_phase: int = 1, end_phase: int = 5):
        """Train complete curriculum."""
        print(f"\n{'='*60}")
        print(f"Starting Curriculum Training (Phases {start_phase}-{end_phase})")
        print(f"{'='*60}")
        
        all_results = {}
        
        for phase_num in range(start_phase, end_phase + 1):
            phase = TrainingPhase(phase_num)
            
            print(f"\n{'='*60}")
            print(f"Phase {phase_num}: {phase.name}")
            print(f"{'='*60}")
            
            # Create data loader
            train_loader = self.create_data_loader(phase)
            val_loader = None  # Optional validation set
            
            # Train phase
            result = self.trainer.train_phase(
                phase=phase,
                train_loader=train_loader,
                val_loader=val_loader,
            )
            
            all_results[f"phase_{phase_num}"] = result
            
            print(f"\nPhase {phase_num} Complete!")
            print(f"  Final Loss: {result['final_loss']:.4f}")
            print(f"  Epochs: {result['num_epochs']}")
        
        # Save training summary
        self._save_training_summary(all_results)
        
        return all_results
    
    def _save_training_summary(self, results: dict):
        """Save training results to JSON."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "device": self.device,
            "model": "VideoVAE",
            "phases": results,
            "total_training_time": "~6-8 weeks on GTX 1070",
        }
        
        summary_path = self.checkpoint_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining summary saved to: {summary_path}")
    
    def demo_inference(self):
        """Demonstrate model inference on sample data."""
        print(f"\n{'='*60}")
        print(f"Demo: Model Inference")
        print(f"{'='*60}")
        
        with torch.no_grad():
            # Demo 1: VAE encoding
            print("\nDemo 1: VAE Encoding")
            x = torch.randn(1, 4, 3, 256, 256).to(self.device)  # 4 frames
            z, mean, logvar = self.vae.encode(x)
            print(f"  Input shape: {x.shape}")
            print(f"  Latent shape: {z.shape}")
            print(f"  Latent mean: {mean.mean().item():.4f}")
            
            # Demo 2: Text encoding
            print("\nDemo 2: Text Encoding (Multilingual)")
            texts = [
                "Wide cinematic pan across a desert landscape",
                "Primer plano de un personaje con movimiento lento",
                "Gros plan d'une explosion avec flammes",
            ]
            
            for text in texts:
                embeddings, pooled = self.text_encoder(text, include_video_vocab=True)
                print(f"  Text: '{text[:50]}...'")
                print(f"    Embedding shape: {pooled.shape}")
    
    def print_phase_summary(self):
        """Print training phase summary."""
        print(f"\n{'='*60}")
        print(f"Curriculum Training Phases")
        print(f"{'='*60}")
        
        phases_info = [
            (1, "Simple Objects", "Single-subject, clean backgrounds", "~2-3h video"),
            (2, "Compound Scenes", "2-3 subjects with gentle motion", "~5h video"),
            (3, "Complex Motion", "Fast action, occlusions, perspectives", "~8h video"),
            (4, "Edge Cases", "Unusual angles, weather, dynamic lighting", "~5h video"),
            (5, "Refinement", "Mixed phases, fine-tuning", "~20h video"),
        ]
        
        for phase_num, name, description, data in phases_info:
            config = self.trainer.get_phase_config(TrainingPhase(phase_num))
            print(f"\nPhase {phase_num}: {name}")
            print(f"  Description: {description}")
            print(f"  Data requirement: {data}")
            print(f"  Epochs: {config['epochs']} | Batch: {config['batch_size']} | "
                  f"LR: {config['learning_rate']:.0e}")


def main():
    parser = argparse.ArgumentParser(
        description="AIPROD v2 Phase 1 Curriculum Learning Training"
    )
    parser.add_argument("--start-phase", type=int, default=1,
                       help="Starting phase (1-5)")
    parser.add_argument("--end-phase", type=int, default=5,
                       help="Ending phase (1-5)")
    parser.add_argument("--resume-phase", type=int, default=None,
                       help="Resume from specific phase (overrides start)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu)")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo inference only")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Phase1MLTraining(device=args.device)
    
    # Show phases
    trainer.print_phase_summary()
    
    if args.demo:
        # Run demo only
        trainer.demo_inference()
    else:
        # Run training
        start_phase = args.resume_phase or args.start_phase
        results = trainer.train_curriculum(start_phase=start_phase, end_phase=args.end_phase)
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(trainer.trainer.get_training_summary())


if __name__ == "__main__":
    main()
