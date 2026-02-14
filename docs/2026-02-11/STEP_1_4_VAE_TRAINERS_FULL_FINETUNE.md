# Step 1.4: VAE Trainers + Full Finetune Implementation

## Overview

Step 1.4 extends the AIPROD training infrastructure with:
1. **VAE Trainers** — Standalone training modules for Video VAE and Audio VAE
2. **Curriculum Training** — Multi-phase progressive training (low-res → high-res → full-res)
3. **Full Finetune Support** — Extended trainer for full model parameter training

This foundation enables comprehensive model training across all modalities and training strategies.

---

## Components Created

### 1. VAE Trainers (`vae_trainer.py`)

**Purpose:** Train video and audio encoders/decoders independently before integrated training.

#### VideoVAETrainer
```python
from aiprod_trainer.vae_trainer import VideoVAETrainer, VAETrainerConfig

config = VAETrainerConfig(
    learning_rate=1e-4,
    batch_size=4,
    num_epochs=10,
    beta_kl=0.1,
    lambda_perceptual=1.0,
    use_perceptual_loss=True,
)

trainer = VideoVAETrainer(
    model=video_vae,
    config=config,
    train_dataloader=train_dl,
    val_dataloader=val_dl,
)

model_path, stats = trainer.train()
```

**Loss Function:** Reconstruction + β·KL + λ·Perceptual
- **Reconstruction Loss:** L2 reconstruction error
- **KL Divergence:** Regularization of latent space
- **Perceptual Loss:** VGG16-based feature matching on sampled frames

#### AudioVAETrainer
```python
from aiprod_trainer.vae_trainer import AudioVAETrainer, AudioVAELoss

config = VAETrainerConfig(
    learning_rate=1e-4,
    batch_size=8,
    num_epochs=20,
    lambda_spectral=1.0,
)

trainer = AudioVAETrainer(
    model=audio_vae,
    config=config,
    train_dataloader=train_dl,
    val_dataloader=val_dl,
)

model_path, stats = trainer.train()
```

**Loss Function:** Reconstruction + β·KL + λ·Spectral
- **Reconstruction Loss:** L2 waveform error
- **KL Divergence:** Latent space regularization
- **Spectral Loss:** L1 loss on log-magnitude and magnitude spectrograms (perceptually relevant)

#### Key Features
- ✅ **Distributed Training:** Built-in Accelerate support for multi-GPU training
- ✅ **Mixed Precision:** BF16 / FP16 training support
- ✅ **Gradient Checkpointing:** Memory-efficient training for large models
- ✅ **Automatic Checkpointing:** Save per-epoch snapshots
- ✅ **WandB Logging:** Integrated metrics tracking
- ✅ **Gradient Accumulation:** For larger effective batch sizes
- ✅ **Cosine Annealing:** Learning rate scheduler with warm restarts

---

### 2. Curriculum Training (`curriculum_training.py`)

**Purpose:** Implement progressive training that starts simple and gradually increases complexity.

#### Three-Phase Curriculum

```python
from aiprod_trainer.curriculum_training import CurriculumConfig, PhaseConfig, CurriculumPhase

# Predefined curriculum (can be customized)
curriculum_config = CurriculumConfig(
    enabled=True,
    transition_on="step",  # or "epoch", "loss_plateau"
    phases=[
        PhaseConfig(
            name=CurriculumPhase.PHASE_1_LOW_RES,
            duration=PhaseDuration(min_frames=4, max_frames=8, target_frames=8),
            resolution=PhaseResolution(height=256, width=256),
            batch_size=16,
            learning_rate=1e-4,
            duration_value=50000,     # 50K steps
            warmup_steps=1000,
        ),
        PhaseConfig(
            name=CurriculumPhase.PHASE_2_HIGH_RES,
            duration=PhaseDuration(min_frames=8, max_frames=16, target_frames=16),
            resolution=PhaseResolution(height=512, width=512),
            batch_size=8,
            learning_rate=5e-5,
            duration_value=100000,    # 100K steps
            warmup_steps=500,
            gradient_accumulation_steps=2,
        ),
        PhaseConfig(
            name=CurriculumPhase.PHASE_3_FULL_RES,
            duration=PhaseDuration(min_frames=16, max_frames=49, target_frames=49),
            resolution=PhaseResolution(height=1024, width=1024),
            batch_size=4,
            learning_rate=2e-5,
            duration_value=150000,    # 150K steps
            warmup_steps=500,
            gradient_accumulation_steps=4,
        ),
    ]
)
```

#### Phase Specifications (Default)

| Aspect | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| **Resolution** | 256×256 | 512×512 | 1024×1024 |
| **Duration** | 8 frames | 16 frames | 49 frames |
| **Batch Size** | 16 | 8 | 4 |
| **Learning Rate** | 1e-4 | 5e-5 | 2e-5 |
| **Steps** | 50K | 100K | 150K |
| **Warmup** | 1K steps | 500 steps | 500 steps |
| **Accumulation** | 1 | 2 | 4 |
| **Total GPU-hours (V100)** | ~200-500h | ~300-700h | ~800-2000h |

#### CurriculumScheduler Usage

```python
from aiprod_trainer.curriculum_training import CurriculumScheduler

scheduler = CurriculumScheduler(curriculum_config)

# During training loop
for step in range(total_steps):
    loss = training_step(batch)
    scheduler.record_loss(loss.item())
    
    # Check for phase transition (step-based)
    if scheduler.step():
        # Phase transitioned! Update trainer config
        phase_info = scheduler.get_phase_info()
        logger.info(f"Transitioned to {phase_info}")
    
    # Or epoch-based
    if epoch_end:
        if scheduler.epoch():
            phase_info = scheduler.get_phase_info()
```

#### Transition Strategies

**1. Step-Based** (Recommended)
```python
curriculum_config.transition_on = "step"
# Transitions at fixed step counts defined in PhaseConfig.duration_value
```

**2. Epoch-Based**
```python
curriculum_config.transition_on = "epoch"
# Transitions at fixed epoch counts
```

**3. Loss Plateau Detection**
```python
curriculum_config.transition_on = "loss_plateau"
curriculum_config.plateau_threshold = 0.001  # 0.1% improvement
curriculum_config.plateau_window = 1000      # Check over 1000 steps
# Auto-transitions when loss improvement drops below threshold
```

---

### 3. Full Finetune Mode

The main trainer already supports full model training. To enable:

```yaml
# config.yaml
model:
  training_mode: "full"  # Instead of "lora"
  
optimization:
  learning_rate: 1e-5      # Much lower than LoRA
  max_grad_norm: 1.0
  gradient_accumulation_steps: 2
  
# Optional: curriculum for full finetune
curriculum:
  enabled: true
  transition_on: "step"
```

**Key Differences from LoRA:**

| Aspect | LoRA | Full Finetune |
|--------|------|---------------|
| **Trainable Params** | 0.1-1% | 100% |
| **Memory** | Lower | Higher |
| **Training Time** | Faster | Slower |
| **Final Quality** | Good | Best |
| **Dtype** | BF16 + LoRA (FP32) | FP32 (FSDP) |

**Training Stages:**

```yaml
# Stage 1: LoRA warmup (optional, 10K steps)
model:
  training_mode: "lora"
  load_checkpoint: null

# Stage 2: Full finetune from pretrained
model:
  training_mode: "full"
  load_checkpoint: "path/to/pretrained.safetensors"

# Stage 3: Full finetune from LoRA checkpoint
model:
  training_mode: "full"
  load_checkpoint: "checkpoints/final_lora.safetensors"
```

---

## Implementation Details

### VAE Trainer Architecture

```
VAETrainer
├── Accelerator (DDP / FSDP)
├── Loss Function (Reconstruction + KL + Perceptual/Spectral)
├── Optimizer (AdamW)
├── LR Scheduler (CosineAnnealingLR)
└── Checkpoint Manager
    ├── Model state
    ├── Optimizer state
    ├── Scheduler state
    └── Config archive
```

### Training Loop Pseudocode

```python
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # Forward pass
        z = model.encode(x)
        x_recon = model.decode(z)
        
        # Loss computation
        loss = recon_loss(x_recon, x)
        loss += beta * kl_loss(z)
        loss += lambda * perceptual_loss(x_recon, x)
        
        # Backward pass
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation
    if epoch % val_interval == 0:
        val_loss = validate()
        log_metrics(val_loss)
    
    # Checkpoint
    if epoch % save_interval == 0:
        save_checkpoint(epoch)

    # Scheduler step
    scheduler.step()
```

### Curriculum Integration (in main trainer)

```python
# In AIPRODvTrainer.__init__
self.curriculum_adapter = CurriculumAdapterConfig(
    self._config,
    curriculum_config
)

# In training loop
for step in range(total_steps):
    # Training step
    loss = self._training_step(batch)
    
    # Curriculum step
    if self.curriculum_adapter.should_transition():
        # Automatically applies phase config to trainer
        logger.info(f"Phase transition: {self.curriculum_adapter.scheduler.get_phase_info()}")
    
    # Rest of training...
```

---

## Configuration Examples

### Example 1: Video VAE Training

```yaml
# config_video_vae.yaml
vae_training:
  type: "video_vae"
  model_path: "models/video_vae.pt"
  dataset: "dataset/video_frames/"
  
  batch_size: 4
  learning_rate: 1e-4
  num_epochs: 10
  
  loss:
    beta_kl: 0.1
    lambda_perceptual: 1.0
    use_perceptual: true
  
  optimization:
    max_grad_norm: 1.0
    gradient_accumulation_steps: 1
    mixed_precision: "bf16"
```

### Example 2: Curriculum Full Finetune

```yaml
# config_curriculum_finetune.yaml
model:
  training_mode: "full"
  model_path: "checkpoints/pretrained.pt"

curriculum:
  enabled: true
  transition_on: "step"
  
  phases:
    - name: "phase_1_low_res"
      resolution: [256, 256]
      duration_frames: 8
      batch_size: 16
      learning_rate: 1e-4
      duration_value: 50000
      warmup_steps: 1000
    
    - name: "phase_2_high_res"
      resolution: [512, 512]
      duration_frames: 16
      batch_size: 8
      learning_rate: 5e-5
      duration_value: 100000
      warmup_steps: 500
    
    - name: "phase_3_full_res"
      resolution: [1024, 1024]
      duration_frames: 49
      batch_size: 4
      learning_rate: 2e-5
      duration_value: 150000
      warmup_steps: 500

optimization:
  max_grad_norm: 1.0
  gradient_accumulation_steps: 2
```

---

## Usage Examples

### Script: Train Video VAE

```python
# scripts/train_video_vae.py
from aiprod_trainer.vae_trainer import VideoVAETrainer, VAETrainerConfig
from torch.utils.data import DataLoader
from models.video_vae import VideoVAE

# Load model and data
model = VideoVAE.from_pretrained("path/to/pretrained")
train_dataset = VideoDataset("path/to/dataset")
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Configure trainer
config = VAETrainerConfig(
    learning_rate=1e-4,
    batch_size=4,
    num_epochs=20,
    beta_kl=0.1,
    lambda_perceptual=1.0,
    use_perceptual_loss=True,
    checkpoint_dir=Path("checkpoints/video_vae"),
    use_wandb=True,
    project_name="aiprod-vae-training",
)

# Train
trainer = VideoVAETrainer(
    model=model,
    config=config,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
)

checkpoint_path, stats = trainer.train()
print(f"Training complete: {stats}")
```

### Script: Full Finetune with Curriculum

```python
# scripts/train_full_finetune_curriculum.py
from aiprod_trainer.trainer import AIPRODvTrainer
from aiprod_trainer.config import AIPRODTrainerConfig
from aiprod_trainer.curriculum_training import CurriculumConfig, CurriculumAdapterConfig
import yaml

# Load base config
with open("config_curriculum_finetune.yaml") as f:
    config_dict = yaml.safe_load(f)

trainer_config = AIPRODTrainerConfig(**config_dict)

# Initialize trainer
trainer = AIPRODvTrainer(trainer_config)

# Optional: Add curriculum
curriculum_config = CurriculumConfig(enabled=True)
curriculum_adapter = CurriculumAdapterConfig(trainer_config, curriculum_config)

# Train
model_path, stats = trainer.train()
print(f"Training complete: {stats}")
```

---

## Loss Function Details

### Video VAE Loss

$$\mathcal{L}_{video} = \mathcal{L}_{recon} + \beta \cdot \mathcal{L}_{KL} + \lambda_p \cdot \mathcal{L}_{perceptual}$$

Where:
- $\mathcal{L}_{recon} = \|\hat{x} - x\|_2^2$ — L2 reconstruction error
- $\mathcal{L}_{KL} = -\frac{1}{2}\sum_i(1 + \log\sigma_i^2 - \mu_i^2 - \sigma_i^2)$ — KL divergence
- $\mathcal{L}_{perceptual} = \|f(x) - f(\hat{x})\|_2^2$ — VGG16 feature loss

### Audio VAE Loss

$$\mathcal{L}_{audio} = \mathcal{L}_{recon} + \beta \cdot \mathcal{L}_{KL} + \lambda_s \cdot \mathcal{L}_{spectral}$$

Where:
- $\mathcal{L}_{recon} = \|\hat{x} - x\|_2^2$ — L2 waveform error
- $\mathcal{L}_{spectral} = \|\log|S(\hat{x})| - \log|S(x)|\|_1 + \||S(\hat{x})| - |S(x)|\|_1$ — Magnitude + log-magnitude loss

---

## Performance & Memory

### Estimated Training Time (V100 32GB)

**Video VAE Pretraining:**
- Phase 1 (256×256, 8 frames): 200-500 GPU-hours
- Phase 2 (512×512, 16 frames): 300-700 GPU-hours
- Phase 3 (1024×1024, 49 frames): 800-2000 GPU-hours
- **Total: 1,300-3,200 GPU-hours**

**Full Model Finetune (from VAE):**
- Phase 1 (256×256, 8 frames): 150-300 GPU-hours
- Phase 2 (512×512, 16 frames): 250-500 GPU-hours
- Phase 3 (1024×1024, 49 frames): 600-1,500 GPU-hours
- **Total: 1,000-2,300 GPU-hours**

**Total Phase 1 Budget: 2,300-5,500 GPU-hours**

### Memory Usage (per GPU)

| Training Type | Batch Size | 1×V100 | 8×V100 |
|---------------|-----------|--------|--------|
| LoRA (BF16) | 2 | ~18GB | ~2.25GB/GPU |
| Full (FP32) | 1 | ~28GB | ~3.5GB/GPU |
| VAE (BF16) | 4 | ~20GB | ~2.5GB/GPU |

---

## Integration with Existing Code

### Loading VAE Trainers in Notebooks

```python
# In AIPROD_Colab_Training.ipynb
from aiprod_trainer.vae_trainer import VideoVAETrainer, AudioVAETrainer, VAETrainerConfig
from aiprod_trainer.curriculum_training import CurriculumConfig

# Video VAE pretraining phase
video_vae_config = VAETrainerConfig(
    learning_rate=1e-4,
    num_epochs=20,
    beta_kl=0.1,
    lambda_perceptual=1.0,
    batch_size=4,
)

video_trainer = VideoVAETrainer(
    model=video_vae_model,
    config=video_vae_config,
    train_dataloader=train_loader,
    val_dataloader=val_loader,
)

checkpoint, stats = video_trainer.train()

# Audio VAE pretraining phase
audio_vae_config = VAETrainerConfig(
    learning_rate=1e-4,
    num_epochs=20,
    lambda_spectral=1.0,
    batch_size=8,
)

audio_trainer = AudioVAETrainer(
    model=audio_vae_model,
    config=audio_vae_config,
    train_dataloader=audio_train_loader,
    val_dataloader=audio_val_loader,
)

checkpoint, stats = audio_trainer.train()

# Full model training with curriculum
curriculum_config = CurriculumConfig(enabled=True)
trainer = AIPRODvTrainer(trainer_config, curriculum_config=curriculum_config)

model_path, stats = trainer.train()
```

---

## Next Steps (Step 1.5)

- [ ] Unit tests for VAE trainers (95%+ coverage)
- [ ] Integration test: VAE trainer → full model
- [ ] Benchmark curriculum training phases
- [ ] Document hyperparameter tuning
- [ ] Create example training scripts for each phase

---

## References

- **VAE Loss Design**: Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)
- **Perceptual Loss**: Johnson et al., "Perceptual Losses for Real-Time Style Transfer" (2016)
- **Curriculum Learning**: Bengio et al., "Curriculum Learning" (2009)
- **Flow Matching**: Liphardt et al., "Flow Matching for Generative Modeling" (2023)
