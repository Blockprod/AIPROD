# Getting Started - Copy & Paste Commands

## Step 1: Activate Your Development Environment

### Option A: Windows Batch File (Easiest)
```batch
.\activate.bat
```

### Option B: PowerShell Manual
```powershell
. .venv_311\Scripts\Activate.ps1
```

### Option C: Command Prompt Manual
```cmd
.venv_311\Scripts\activate.bat
```

---

## Step 2: Verify GPU is Detected

```bash
python -c "import torch; print('GPU:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected output**:
```
GPU: True | NVIDIA GeForce GTX 1070
```

---

## Step 3: Download Models (One-Time Setup)

See [DEVELOPMENT_GUIDE.md#model-downloads](DEVELOPMENT_GUIDE.md) for detailed HuggingFace download links.

**Quick summary**:
- Download to `models/aiprod2/` (~8 GB)
- Download to `models/gemma-3/` (~6 GB)

---

## Step 4a: Run First Video Generation

```bash
python examples/quickstart.py --prompt "A beautiful sunset over mountains, cinematic, 4K"
```

**Parameters**:
- `--prompt TEXT` - Video description (required)
- `--pipeline {two_stages,one_stage,distilled}` - Pipeline choice (default: two_stages)
- `--resolution {480,720,1080}` - Output resolution (default: 720)
- `--frames N` - Number of frames (default: 48)
- `--guidance FLOAT` - Guidance scale (default: 7.5)
- `--seed N` - Random seed for reproducibility (default: random)

**Examples**:
```bash
# Fast generation (5-10 min)
python examples/quickstart.py --prompt "Ocean waves" --pipeline distilled --resolution 480 --frames 24

# Quality generation (20-30 min)
python examples/quickstart.py --prompt "Mountain landscape, sunset" --pipeline two_stages --resolution 720 --frames 48

# Reproduce specific result
python examples/quickstart.py --prompt "Ocean waves" --seed 42
```

---

## Step 4b: Monitor GPU (Optional, Separate Terminal)

```bash
# Activate venv in new terminal first
. .venv_311\Scripts\Activate.ps1

# Then start monitoring
python scripts/monitor_gpu.py
```

**What you'll see**:
```
Time        GPU Memory      Utilization    Temp       Status
10:45:23    6.2 / 8.0 GB   95%            65°C       🔴 HIGH LOAD
10:45:24    6.2 / 8.0 GB   93%            66°C       🔴 HIGH LOAD
```

---

## Iterative Testing

### Test Different Prompts
```bash
python examples/quickstart.py --prompt "Forest waterfall"
python examples/quickstart.py --prompt "City traffic at night"
python examples/quickstart.py --prompt "Astronaut on moon"
```

### Test Different Pipelines
```bash
# Fastest (lower quality)
python examples/quickstart.py --prompt "Your prompt" --pipeline distilled

# Balanced (recommended)
python examples/quickstart.py --prompt "Your prompt" --pipeline one_stage

# Highest quality (slowest)
python examples/quickstart.py --prompt "Your prompt" --pipeline two_stages
```

### Test Different Resolutions
```bash
# Fastest
python examples/quickstart.py --prompt "Your prompt" --resolution 480 --frames 24

# Balanced
python examples/quickstart.py --prompt "Your prompt" --resolution 720 --frames 48

# Highest quality
python examples/quickstart.py --prompt "Your prompt" --resolution 1080 --frames 48
```

### Reproduce Results
```bash
# Save a seed when you like a result
python examples/quickstart.py --prompt "Your prompt" --seed 42

# Regenerate with same seed = same output
python examples/quickstart.py --prompt "Your prompt" --seed 42
```

---

## Troubleshooting

### GPU Not Detected
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```
Expected: `True` and `12.1`

If False, see [PHASE_1_GPU_CONFIG_COMPLETE.md](PHASE_1_GPU_CONFIG_COMPLETE.md)

### Out of Memory (OOM)
Use distilled pipeline and lower resolution:
```bash
python examples/quickstart.py --prompt "Your prompt" --pipeline distilled --resolution 480 --frames 24
```

### Models Not Found
```bash
# Check what files are in models directory
dir models\aiprod2\
dir models\gemma-3\
```

Then download missing files from HuggingFace (see DEVELOPMENT_GUIDE.md)

### Very Slow Generation
Check GPU monitor:
```bash
# In separate terminal
python scripts/monitor_gpu.py
```

If GPU utilization < 50%, something is wrong. If GPU at 90%+, it's just slow (expected on GTX 1070 for high-res).

---

## Where Did My Video Go?

Output is saved to:
```
outputs/
  video_TIMESTAMP.mp4
  metadata_TIMESTAMP.json
```

Check recent files:
```powershell
# PowerShell
Get-ChildItem outputs\ -File | Sort-Object LastWriteTime -Descending | Select-Object -First 3

# Command Prompt
dir outputs\ /O-D /T:W
```

---

## What's Next?

1. ✅ Activate environment (`activate.bat`)
2. ✅ Verify GPU (`python -c "import torch..."`)
3. 📥 Download models (HuggingFace links in DEVELOPMENT_GUIDE.md)
4. 🎬 Run first video (`python examples/quickstart.py`)
5. 📊 Monitor GPU (`python scripts/monitor_gpu.py`)
6. 🎨 Iterate on prompts
7. 📈 When satisfied: Plan Phase 2 (HF Spaces with H100)

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `.\activate.bat` | Activate venv + verify GPU |
| `python examples/quickstart.py --prompt "..."` | Generate video |
| `python scripts/monitor_gpu.py` | Monitor GPU in real-time |
| `dir outputs\` | See generated videos |
| `python -c "import torch; print(torch.cuda.memory_allocated() / 1e9, 'GB')"` | Check VRAM usage |
| `python -c "import torch; print(torch.__version__)"` | Check PyTorch version |

---

## Performance Expectations (GTX 1070)

| Configuration | Time |
|---------------|------|
| distilled, 480p, 24fps | 5-10 min |
| one_stage, 720p, 48fps | 15-25 min |
| two_stages, 720p, 48fps | 25-45 min |
| two_stages, 1080p, 48fps | 40-60 min |

*Times depend on prompt complexity and system state*

---

**Generated**: 2025-01-17  
**Ready For**: Copy & Paste Commands
