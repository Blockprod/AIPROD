# AIPROD Trainer

> **⚠️ PROPRIETARY - STRICTLY CONFIDENTIAL**  
> © 2026 Blockprod. All rights reserved.  
> Unauthorized access, copying, or distribution is strictly prohibited.

This package provides tools and scripts for training and fine-tuning
the **AIPROD** audio-video generation model. It enables LoRA training, full
fine-tuning, and training of video-to-video transformations (IC-LoRA) on custom datasets.

---

## 📖 Documentation

All detailed guides and technical documentation are in the [docs](./docs/) directory:

- [⚡ Quick Start Guide](docs/quick-start.md)
- [🎬 Dataset Preparation](docs/dataset-preparation.md)
- [🛠️ Training Modes](docs/training-modes.md)
- [⚙️ Configuration Reference](docs/configuration-reference.md)
- [🚀 Training Guide](docs/training-guide.md)
- [🧪 Inference Guide](../AIPROD-pipelines/README.md)
- [🔧 Utility Scripts](docs/utility-scripts.md)
- [📚 AIPROD-Core Documentation](../AIPROD-core/README.md)
- [🛡️ Troubleshooting Guide](docs/troubleshooting.md)

---

## 🔧 Requirements

- **AIPROD Model Checkpoint** - Local `.safetensors` file
- **AIPROD Text Encoder** - Local AIPROD text encoder model directory (required for AIPROD)
- **Linux with CUDA** - CUDA 13+ recommended for optimal performance
- **Nvidia GPU with 80GB+ VRAM** - Recommended for the standard config. For GPUs with 32GB VRAM (e.g., RTX 5090),
  use the [low VRAM config](configs/AIPROD2_av_lora_low_vram.yaml) which enables INT8 quantization and other
  memory optimizations

---

## 📝 License

**Proprietary** - © 2026 Blockprod. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or modification is strictly prohibited.
