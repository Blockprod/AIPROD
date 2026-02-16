# AIPROD Captioning — Qwen2.5-Omni-7B

Audio-visual captioning model for dataset preparation.

## Provisioning

```bash
python scripts/download_models.py --model captioning
```

**Source:** `Qwen/Qwen2.5-Omni-7B` (Apache 2.0)
**Size:** ~15 GB
**Usage:** `aiprod_trainer.captioning.QwenOmniCaptioner`

## Notes

- Loaded with `local_files_only=True` — no network access at runtime
- Used during dataset preprocessing to generate video captions
- 8-bit loading supported via `BitsAndBytesConfig(load_in_8bit=True)`
- Not required at inference time — only for training pipeline
