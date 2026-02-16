# AIPROD Text Encoder

Base weights for the AIPROD text encoder (LLMBridge).

## Provisioning

```bash
python scripts/download_models.py --model text-encoder
```

**Source:** `google/gemma-3-1b-pt` (Apache 2.0)
**Size:** ~2 GB
**Usage:** `aiprod_core.model.text_encoder.LLMBridge`

## Notes

- Loaded with `local_files_only=True` — no network access at runtime
- Used as base weights for AIPROD sovereign fine-tuning (Phase D)
- 8-bit loading available via `aiprod_trainer.text_encoder_8bit.load_8bit_text_encoder()`
