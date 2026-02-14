# AIPROD - Internal Documentation

> **⚠️ PRIVATE PROJECT - CONFIDENTIAL**  
> This repository contains proprietary code and innovations. Do not share externally.

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage Examples](#-usage-examples)
- [Available Pipelines](#-available-pipelines)
- [Packages](#-packages)
- [Troubleshooting](#-troubleshooting)

## 💻 System Requirements

### Minimum
- **GPU**: NVIDIA GPU with 24GB VRAM (RTX 3090, RTX 4090, A5000)
- **RAM**: 32GB system memory
- **Storage**: 100GB+ free space
- **OS**: Linux (Ubuntu 20.04+), Windows 10/11
- **Python**: 3.11.x (required)
- **CUDA**: 11.8+ or 12.1+

### Recommended
- **GPU**: NVIDIA H100, A100 (80GB), RTX 6000 Ada (48GB)
- **RAM**: 64GB+ system memory
- **Storage**: NVMe SSD

## 🚀 Installation

### Standard Installation

```bash
# Clone repository
git clone <private-repo-url>
cd AIPROD

# Create virtual environment (Python 3.11 required)
python3.11 -m venv .venv_311
source .venv_311/bin/activate  # Linux/Mac
# OR
.venv_311\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install packages in editable mode
pip install -e packages/aiprod-core
pip install -e packages/aiprod-pipelines
pip install -e packages/aiprod-trainer
```

### Optional Optimizations

```bash
# xFormers (recommended for most GPUs)
pip install xformers

# Flash Attention 3 (H100/H200 only)
pip install flash-attn --no-build-isolation
```

## ⚙️ Configuration

### 1. Environment Setup

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Model paths
MODEL_DIR=/path/to/your/models
OUTPUT_DIR=/path/to/outputs

# GPU settings
CUDA_VISIBLE_DEVICES=0

# Optional: HuggingFace token
HF_TOKEN=your_token_here

# Optional: WandB for training
WANDB_API_KEY=your_key_here
WANDB_PROJECT=aiprod-training
```

### 2. Download Models

Download required models from HuggingFace:
- AIPROD checkpoint (choose one): dev, dev-fp8, distilled, distilled-fp8
- Spatial upscaler: AIPROD-spatial-upscaler-x2-1.0.safetensors
- Gemma text encoder: gemma-3-12b-it-qat-q4_0-unquantized
- Optional LoRAs for specific controls

Place models in `models/` directory:
```
models/
├── aiprod2/
│   ├── AIPROD-19b-dev-fp8.safetensors
│   └── AIPROD-spatial-upscaler-x2-1.0.safetensors
└── gemma-3-12b-it/
    └── (gemma model files)
```

### 3. Verify Installation

```bash
python -c "import aiprod_core; import aiprod_pipelines; print('✓ OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📝 Usage Examples

### Text-to-Video

```python
from aiprod_pipelines import TI2VidTwoStagesPipeline

pipeline = TI2VidTwoStagesPipeline(
    ckpt_dir="models/aiprod2",
    text_encoder_dir="models/gemma-3-12b-it",
    fp8_transformer=True
)

video = pipeline(
    prompt="Your video description here",
    height=480,
    width=832,
    num_frames=121,
    guidance_scale=3.0,
    seed=42
)

pipeline.save_video(video, "output.mp4", fps=24)
```

### Image-to-Video

```python
from PIL import Image

image = Image.open("input.jpg")

video = pipeline(
    prompt="Describe the animation",
    image=image,
    num_frames=121,
    guidance_scale=3.0
)

pipeline.save_video(video, "animated.mp4", fps=24)
```

### Fast Generation

```python
from aiprod_pipelines import DistilledPipeline

pipeline = DistilledPipeline(
    ckpt_dir="models/aiprod2",
    text_encoder_dir="models/gemma-3-12b-it",
    fp8_transformer=True
)

video = pipeline(
    prompt="Your prompt",
    height=480,
    width=832,
    num_frames=121
)
```

## 🔧 Available Pipelines

- **TI2VidTwoStagesPipeline** - Two-stage generation with upsampling
- **TI2VidOneStagePipeline** - Single-stage for prototyping
- **DistilledPipeline** - Fast inference
- **ICLoraPipeline** - Video-to-video transformations
- **KeyframeInterpolationPipeline** - Keyframe interpolation

See package documentation for detailed usage.

## 📦 Packages

```
packages/
├── aiprod-core/         # Core model implementation
├── aiprod-pipelines/    # Pipeline implementations
└── aiprod-trainer/      # Training tools
```

Each package has its own README:
- [aiprod-core/README.md](packages/aiprod-core/README.md)
- [aiprod-pipelines/README.md](packages/aiprod-pipelines/README.md)
- [aiprod-trainer/README.md](packages/aiprod-trainer/README.md)

## 🔧 Troubleshooting

### CUDA Out of Memory

```python
# Enable FP8 mode
pipeline = TI2VidTwoStagesPipeline(
    ckpt_dir="models/aiprod2",
    fp8_transformer=True  # Reduces VRAM usage
)

# Reduce resolution
video = pipeline(
    prompt="...",
    height=384,  # Lower resolution
    width=640,
    num_frames=61  # Fewer frames
)
```

### Import Errors

```bash
# Verify Python version
python --version  # Must be 3.11.x

# Reinstall packages
pip install -e packages/aiprod-core
pip install -e packages/aiprod-pipelines
pip install -e packages/aiprod-trainer

# Test
python -c "from aiprod_core import model; print('✓')"
```

### Model Loading Issues

```bash
# Check model files
ls -lh models/aiprod2/*.safetensors

# Verify paths in .env
cat .env | grep MODEL_DIR
```

### Slow Performance

```python
# Use fast pipeline
from aiprod_pipelines import DistilledPipeline

# Enable FP8
pipeline = DistilledPipeline(fp8_transformer=True)

# Reduce steps
video = pipeline(
    prompt="...",
    num_inference_steps=20  # Faster
)
```

### Windows Permission Issues

```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate environment
.venv_311\Scripts\Activate.ps1
```

## 📚 Additional Documentation

- **Configuration**: [config/README.md](config/README.md)
- **Deployment**: [deploy/README.md](deploy/README.md)
- **Scripts**: [scripts/README.md](scripts/README.md)
- **Training**: [packages/aiprod-trainer/docs/](packages/aiprod-trainer/docs/)

## 🔒 Confidentiality Notice

This project contains proprietary algorithms and optimizations. All code, documentation, and generated outputs are confidential. Do not:
- Share code or documentation externally
- Discuss technical details publicly
- Upload models or code to public repositories
- Share performance metrics or benchmarks

For questions or issues, contact the internal team only.

---

**Last Updated**: February 10, 2026  
**For Internal Use Only**
