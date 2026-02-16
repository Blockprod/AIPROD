# AIPROD CLIP — ViT-L/14

CLIP model for semantic QA scoring (image-text alignment).

## Provisioning

```bash
python scripts/download_models.py --model clip
```

**Source:** `openai/clip-vit-large-patch14` (MIT)
**Size:** ~1.7 GB
**Usage:** `aiprod_pipelines.api.qa_semantic_local`

## Notes

- Loaded with `local_files_only=True` — no network access at runtime
- Used for automated quality assurance of generated videos
- Computes cosine similarity between prompt embeddings and frame embeddings
