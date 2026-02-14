# AIPROD Pipelines

> **⚠️ PROPRIETARY - STRICTLY CONFIDENTIAL**  
> © 2026 Blockprod. All rights reserved.  
> This package contains trade secrets and proprietary technology. Unauthorized access, copying, or distribution is strictly prohibited and may result in legal action.

Production-ready inference pipelines for the AIPROD generation system.

## 🔧 Installation

```bash
pip install -e packages/aiprod-pipelines
```

## 📦 Available Pipelines

| Pipeline | Description |
|----------|-------------|
| `ti2vid_two_stages` | Two-stage generation (recommended) |
| `ti2vid_one_stage` | Single-stage generation |
| `distilled` | Fast inference pipeline |
| `ic_lora` | Video-to-video with LoRA |
| `keyframe_interpolation` | Keyframe-based generation |

## 🚀 CLI Usage

```bash
python -m AIPROD_pipelines.<pipeline_name> --help
```

## 🔗 Related Projects

- [AIPROD-core](../aiprod-core/) — Core library
- [AIPROD-trainer](../aiprod-trainer/) — Training tools

---

*© 2026 Blockprod. All rights reserved. Proprietary and confidential.*

- **[AIPROD-Core](../AIPROD-core/)** - Core model implementation and inference components (schedulers, guiders, noisers, patchifiers)
- **[AIPROD-Trainer](../AIPROD-trainer/)** - Training and fine-tuning tools
