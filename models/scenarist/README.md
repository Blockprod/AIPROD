# AIPROD Scenarist — Mistral-7B-Instruct

LLM weights for the AIPROD storyboard/scenario generation engine.

## Provisioning

```bash
python scripts/download_models.py --model scenarist
```

**Source:** `mistralai/Mistral-7B-Instruct-v0.3` (Apache 2.0)
**Size:** ~14 GB
**Usage:** `aiprod_pipelines.inference.scenarist.ScenaristLLM`

## Notes

- Loaded with `local_files_only=True` — no network access at runtime
- Generates creative storyboards from user prompts
- Supports structured JSON output for scene graphs
