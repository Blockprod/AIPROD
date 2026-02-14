# GPU THERMAL CRISIS - EMERGENCY MEASURES

**Status**: Hardware throttling FAILED - Software-only optimizations in effect

---

## 🔴 ISSUE

- GPU temperature: **90°C at IDLE** (should be ~70°C)
- GPU clock limiting: **FAILED** (still 1544 MHz, target was 1500 MHz)
- Power limiting: **FAILED** ([N/A])
- Hardware configuration: **NOT APPLIED**

Root cause: nvidia-smi lacks sufficient privileges to apply hardware-level constraints on this system.

---

## ✅ SOLUTION: AGGRESSIVE SOFTWARE OPTIMIZATION

Since hardware throttling didn't work, we've aggressively cut:

1. **Batch Sizes** (drastically reduced):
   - Phase 1: 8 → **4** (smaller = cooler)
   - Phase 2: 6 → **3**
   - Phase 3: 4 → **2**
   - Phase 4: 4 → **2**
   - Phase 5: 4 → **2**

2. **Resolutions** (lower = less computation = less heat):
   - Phase 1: 256×256 → **224×224**
   - Phase 2: 320×320 → **280×280**
   - Phase 3: 384×384 → **336×336**
   - Phase 4: 384×384 → **336×336**
   - Phase 5: 384×384 → **336×336**

3. **Frame Count** (fewer frames = less memory):
   - Phase 2: 24 → **20**
   - Phase 3: 32 → **28**
   - Phase 4: 32 → **28**
   - Phase 5: 32 → **28**

4. **Software Optimizations** (always active):
   - Mixed precision: **ON**
   - Gradient checkpointing: **ON**
   - Channel-last layout: **ON** (better cache)
   - Torch compile: **OFF** (overhead)

---

## 📊 Expected Results

```
BEFORE (with failed hardware config):
- GPU: 90°C at idle, 100°C+ during training
- Training: UNSTABLE, thermal throttling
- Speed: SLOW (due to throttling)

AFTER (software-only approach):
- GPU: ~75-80°C during training (aggressive cuts)
- Training: STABLE (smaller batches, lower res)
- Speed: SLOWER but RELIABLE
```

---

## 🚀 How to Proceed

### 1. Clear GPU Cache
```powershell
# Stop all Python processes
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force

# Wait for GPU to cool
Start-Sleep -Seconds 30

# Verify temperature dropped
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
# Should show ~60-70°C, not 90°C
```

### 2. Start Training
```powershell
python packages/aiprod-core/src/training/train.py --start-phase 1
```

### 3. Monitor Temperatures (NEW window)
```powershell
python scripts/gpu_thermal_monitor.py --duration 3600
```

Watch for:
- **OK** if temps stay 75-85°C
- **WARNING** if temps spike above 85°C
- **BAD** if temps exceed 90°C (stop training)

---

## ⚠️ If Still Too Hot

### Immediate Actions:
1. **Reduce batch size FURTHER**:
   - Edit: `packages/aiprod-core/src/training/curriculum.py`
   - Change: `phase1_batch_size: int = 4` → `2`

2. **Reduce resolution FURTHER**:
   - Change: `phase1_resolution: Tuple[int, int] = (224, 224)` → `(192, 192)`

3. **Reduce frame count**:
   - Change: `phase1_max_frames: int = 12` → `8`

4. **Run shorter epochs**:
   - Change: `phase1_epochs: int = 20` → `10`

### Hardware Checks:
1. Clean GPU fans (dust buildup?)
2. Check thermal paste condition (may be dried)
3. Ensure proper case ventilation
4. Consider external fan or better airflow

### Last Resort:
```powershell
# Only use on GTX 1070 if temps still >85°C
# This is risky but might help:
nvidia-smi -pl 100  # Limit to 100W (very conservative)
nvidia-smi -i 0 -lgc 1200  # Lock clocks even lower

# WARNING: Will impact training speed significantly
```

---

## 📈 New Configuration Summary

| Phase | Batch | Resolution | Max Frames | Status |
|-------|-------|-----------|-----------|--------|
| Phase 1 | 4 | 224×224 | 12 | Conservative |
| Phase 2 | 3 | 280×280 | 20 | Conservative |
| Phase 3 | 2 | 336×336 | 28 | Conservative |
| Phase 4 | 2 | 336×336 | 28 | Conservative |
| Phase 5 | 2 | 336×336 | 28 | Conservative |

**Impact on training time**: +40-50% slower than original, but STABLE and SAFE

---

## 🔔 Exit Criteria (STOP TRAINING if):
- Temperature exceeds **90°C** persistently
- Thermal throttling detected (clock drops mid-epoch)
- GPU memory OOM errors
- Loss stops improving across epochs

If any of these occur:
1. Stop training immediately: `Ctrl+C`
2. Wait 60 seconds for GPU to cool
3. Reduce batch size in config
4. Restart training

---

## 📝 Notes

- These settings are **TEMPORARY** until hardware can be fixed
- Original aggressive configs (batch=8, res=384) were designed for better data centers
- GTX 1070 in this environment needs lighter workloads
- Future: Replace thermal paste, improve case cooling, or upgrade GPU

---

**Last updated**: February 11, 2026 11:30 AM  
**Configuration file**: `packages/aiprod-core/src/training/curriculum.py`  
**Status**: Ready to test with software-only optimizations
